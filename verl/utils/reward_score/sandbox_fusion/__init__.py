# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import traceback

from .utils import check_correctness, SUPPORTED_LANGUAGES

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

# Language mapping from dataset format to sandbox_fusion format
LANGUAGE_MAPPING = {
    'py3': 'python',
    'py2': 'python',
    'python3': 'python',
    'python2': 'python',
    'c++': 'cpp',
    'c++17': 'cpp',
    'c++14': 'cpp',
    'c++11': 'cpp',
}


def convert_leetcode_format(leetcode_test_cases):
    """
    Convert LeetCode format to a format compatible with check_correctness.

    LeetCode format contains:
        - entry_point: String expression to get the function (e.g., "Solution().minCost")
        - import_prefix: Helper imports and classes to execute before user code
        - test_code: Test function definition with assertions

    This is converted to a format that check_correctness can handle:
        - Single test case with empty stdin input
        - Special fields (import_prefix, entry_point, test_code) preserved for code assembly
        - No expected output (success determined by assertion pass/fail)

    Args:
        leetcode_test_cases: Dictionary with entry_point, import_prefix, test_code keys

    Returns:
        Dictionary compatible with check_correctness input format
    """
    logger.info(f"Converting LeetCode format with entry_point: {leetcode_test_cases.get('entry_point')}")

    return {
        "inputs": [""],  # Single test case with empty stdin
        "outputs": [None],  # No output comparison needed (assertions determine success)
        "import_prefix": leetcode_test_cases.get("import_prefix", ""),
        "entry_point": leetcode_test_cases.get("entry_point", ""),
        "test_code": leetcode_test_cases.get("test_code", "")
    }


def compute_score(
    sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_cases, continuous=False, timeout=200
):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    solution = completion
    detected_language = "python"  # Default language

    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
        detected_language = "python"
    elif "```" in completion:
        # Handle cases like ```cpp\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Extract potential language specifier
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                first_line_stripped = first_line.strip().lower()

                # Check if it's a valid language specifier
                if first_line_stripped.replace('+', '').replace('#', '').isalnum():
                    # Apply language mapping
                    mapped_language = LANGUAGE_MAPPING.get(first_line_stripped, first_line_stripped)

                    # Verify if supported
                    if mapped_language in SUPPORTED_LANGUAGES:
                        detected_language = mapped_language
                        solution = rest
                        logger.info(f"Detected language: {first_line_stripped} -> {detected_language}")
                    else:
                        # Language not supported, fallback to python but keep the first line
                        logger.warning(f"Unsupported language '{first_line_stripped}', using Python as fallback")
                        detected_language = "python"
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        # Priority 1: LeetCode format (entry_point + test_code)
        if test_cases is not None and "entry_point" in test_cases and "test_code" in test_cases:
            logger.info("Detected LeetCode test format")
            test_cases = convert_leetcode_format(test_cases)
        # Priority 2: Assert case format
        elif test_cases is not None and "assert_case" in test_cases and isinstance(test_cases.get("assert_case"), list):
            assert_cases = test_cases.get("assert_case")
            test_cases.setdefault("inputs", ["" for _ in assert_cases])
            test_cases.setdefault("outputs", [None for _ in assert_cases])
        # Priority 3: Standard format validation
        elif not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            language=detected_language,  # Pass detected language
        )

        # Calculate score
        if not res_list:  # If there are no results (e.g., invalid input)
            return 0.0, metadata_list

        if continuous:
            # Calculate pass rate for the first N (e.g., 10) test cases
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                score = 0.0
            else:
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                score = passed_count / num_to_consider
            # Return all metadata, even if score is based on the first N
            final_metadata = metadata_list
        else:
            # Calculate pass rate for all test cases
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list

    except Exception as e:
        logger.error(f"Error during compute_score: {e}")
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Return dict instead of tuple for consistency with other scorers
    error_msg = ""
    if isinstance(final_metadata, list) and final_metadata:
        # Extract detailed error information from metadata
        error_parts = []

        for i, m in enumerate(final_metadata):
            if not isinstance(m, dict):
                continue

            status = m.get("status", "")

            # Skip successful test cases
            if status == "success":
                continue

            # Build error message based on error type
            case_prefix = f"Test {i+1}" if len(final_metadata) > 1 else ""

            if status == "compile_error" or status == "compile_timeout":
                # Compilation error - include full compile_stderr
                compile_stderr = m.get("compile_stderr", "")
                if compile_stderr:
                    error_parts.append(f"{case_prefix} compile_error: {compile_stderr}")
                else:
                    error_parts.append(f"{case_prefix} {status}")

            elif status in ["runtime_error", "timeout"]:
                # Runtime error - include full stderr and exit code
                stderr = m.get("stderr", "")
                exit_code = m.get("exit_code")

                details = []
                if exit_code is not None and exit_code != 0:
                    details.append(f"exit_code={exit_code}")

                if stderr:
                    details.append(f"stderr: {stderr}")

                if details:
                    error_parts.append(f"{case_prefix} {status}: {' | '.join(details)}")
                else:
                    error_parts.append(f"{case_prefix} {status}")

            elif status == "wrong_answer":
                # Wrong answer - include full stdout and expected output
                stdout = m.get("stdout", "")
                expected = m.get("expected_output", "")

                if stdout or expected:
                    error_parts.append(f"{case_prefix} wrong_answer: got '{stdout}' expected '{expected}'")
                else:
                    error_parts.append(f"{case_prefix} wrong_answer")

            elif status in ["api_error", "sandbox_error"]:
                # API/Sandbox error - include full error message
                api_error = m.get("api_request_error", "")
                if api_error:
                    error_parts.append(f"{case_prefix} {status}: {api_error}")
                else:
                    error_parts.append(f"{case_prefix} {status}")

            else:
                # Other errors - include status
                error_parts.append(f"{case_prefix} {status}")

            # Include first 5 errors (avoid limiting too much)
            if len(error_parts) >= 5:
                remaining = len([m for m in final_metadata[i+1:] if m.get("status") != "success"])
                if remaining > 0:
                    error_parts.append(f"... and {remaining} more errors")
                break

        # Combine error messages with separator
        if error_parts:
            error_msg = " || ".join(error_parts)

    return {
        "score": float(score),
        "error": error_msg,
        "reward_think": 0.0,
        "reward_fmt": 0.0,
        "reward_correct": 0.0,
        "reward_length": 0.0,
    }
