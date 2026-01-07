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

"""
CodeV scorer for Verilog code generation.
Performs equivalence checking between generated code and golden reference
using Sandbox Fusion for Verilog simulation.

Supports both XML (<think>, <answer>) and GPT OSS (<|channel|>analysis, <|channel|>final) formats.
"""

import json
import logging
import pickle
import re
import threading
from itertools import combinations, product
from typing import Optional

from .codev_eval_toolkit import eda_tools, extract_verilog
from .sandbox_fusion.utils import call_sandbox_api
from .format_handlers import detect_format, get_format_handler

logger = logging.getLogger(__name__)


def check_format(output, format_type="auto"):
    """
    Check if the output has proper format with thinking and answer sections.

    Supports:
    - XML format: <think>...</think> and <answer>...</answer> tags
    - GPT OSS format: <|channel|>analysis and <|channel|>final blocks

    Args:
        output: String containing the model output
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        bool: True if format is valid, False otherwise

    Examples:
        XML format:
        >>> check_format("<think>reasoning</think><answer>```verilog\\nmodule...\\n```</answer>")
        True

        GPT OSS format:
        >>> check_format("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n<|start|>assistant<|channel|>final<|message|>```verilog\\nmodule...\\n```<|return|>")
        True
    """
    if format_type == "auto":
        format_type = detect_format(output)

    if format_type == "gpt_oss":
        # For GPT OSS, check that we have analysis and final channels
        handler = get_format_handler(format_type)
        has_analysis = handler.has_analysis(output)
        has_final = handler.has_final_response(output)

        # Must have at least final channel (analysis is optional)
        return has_final
    else:
        # XML format: <think> tags are required, <answer> tags are optional (Qwen3 compatibility)
        # Pattern 1 (traditional): <think>...</think>\n<answer>...</answer>
        # Pattern 2 (Qwen3): <think>...</think>\n[plain text with code]

        # Must have exactly one pair of <think> tags
        if output.count("<think>") != 1 or output.count("</think>") != 1:
            return False

        # Check if has <answer> tags (traditional format)
        if output.count("<answer>") == 1 and output.count("</answer>") == 1:
            # Validate tag order: <think>, </think>, <answer>, </answer>
            tags = ["<think>", "</think>", "<answer>", "</answer>"]
            positions = [output.find(tag) for tag in tags]
            return positions[0] < positions[1] < positions[2] < positions[3]

        # Or allow plain text after </think> (Qwen3 format)
        # Just verify <think> tags are properly positioned
        return (output.find("<think>") < output.find("</think>") and
                output.strip().startswith("<think>"))


def assemble_verilog_code(golden_code, dut_code, testbench_code):
    """
    Assemble golden code, DUT code, and testbench into a single Verilog file.

    Args:
        golden_code: Golden reference Verilog code with _gold suffix
        dut_code: Device Under Test Verilog code with _gate suffix
        testbench_code: Testbench module code

    Returns:
        str: Complete Verilog program
    """
    return f"""
// ========== GOLDEN REFERENCE CODE ==========
{golden_code}

// ========== DEVICE UNDER TEST CODE ==========
{dut_code}

// ========== TESTBENCH CODE ==========
{testbench_code}
"""


def verify_verilog_via_sandbox(
    golden_code,
    dut_code,
    golden_top,
    gate_top,
    port_info,
    sandbox_fusion_url,
    concurrent_semaphore=None,
    compile_timeout=30,
    run_timeout=60,
    random_seq_steps=1000,
    random_seq_num=100
):
    """
    Verify Verilog code equivalence using Sandbox Fusion.

    Args:
        golden_code: Golden reference Verilog code
        dut_code: Device Under Test Verilog code
        golden_top: Top module name for golden code
        gate_top: Top module name for DUT code
        port_info: Tuple of (input_port_width, output_port_width, clock_port_polarity, reset_port_polarity_sync)
        sandbox_fusion_url: URL of Sandbox Fusion service
        concurrent_semaphore: Optional semaphore for concurrency control
        compile_timeout: Compilation timeout in seconds
        run_timeout: Run timeout in seconds
        random_seq_steps: Number of random test steps per sequence
        random_seq_num: Number of random test sequences

    Returns:
        dict: Verification result with 'correct' boolean and optional error info
    """
    try:
        # Initialize EDA tools
        v = eda_tools(
            golden_suffix="_gold",
            gate_suffix="_gate",
            random_seq_steps=random_seq_steps,
            random_seq_num=random_seq_num,
            quiet=True
        )

        # Process Verilog code (add suffixes to module names)
        renamed_golden_code = v.process_verilog(golden_code, "_gold")
        renamed_gate_code = v.process_verilog(dut_code, "_gate")

        # Extract port information
        input_port_width, output_port_width, clock_port_polarity, reset_port_polarity_sync = port_info

        # Generate testbench
        testbench_code = v.generate_testbench(
            input_port_width=input_port_width,
            output_port_width=output_port_width,
            clock_port_polarity=clock_port_polarity,
            reset_port_polarity_sync=reset_port_polarity_sync,
            golden_top=golden_top,
            gate_top=gate_top
        )

        # Assemble complete Verilog program
        full_verilog = assemble_verilog_code(renamed_golden_code, renamed_gate_code, testbench_code)

        # Call Sandbox Fusion API
        logger.info("Calling Sandbox Fusion for Verilog simulation")
        api_response, error_msg = call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url,
            code=full_verilog,
            stdin="",  # Testbench self-generates inputs
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
            memory_limit_mb=2048,
            language="verilog"
        )

        if error_msg:
            # Only log ERROR for non-timeout issues (timeouts are expected)
            if "timeout" not in error_msg.lower():
                logger.error(f"Sandbox API error: {error_msg}")
            return {"correct": False, "api_error": error_msg}

        if not api_response:
            logger.error("No API response received")
            return {"correct": False, "api_error": "No response from Sandbox"}

        # Check response status
        api_status = api_response.get("status")
        if api_status != "Success":
            logger.debug(f"API returned status: {api_status}")
            compile_result = api_response.get("compile_result", {})
            run_result = api_response.get("run_result", {})

            error_info = {
                "correct": False,
                "api_status": api_status,
                "compile_stderr": compile_result.get("stderr") if compile_result else None,
                "run_stderr": run_result.get("stderr") if run_result else None
            }
            return error_info

        # Check run result
        run_result = api_response.get("run_result", {})
        stdout = run_result.get("stdout", "")
        stderr = run_result.get("stderr", "")

        # Parse error rate from stdout
        error_rate_pattern = r"Error rate:\s*(\d+\.\d+)"
        error_rate_match = re.search(error_rate_pattern, stdout)
        error_rate = float(error_rate_match.group(1)) if error_rate_match else 1.0

        # Check if all tests passed
        if "All tests passed." in stdout:
            logger.debug("Verification passed: All tests passed")
            return {
                "correct": True,
                "error_rate": error_rate,
                "stdout": stdout,  # Full output
                "stderr": stderr if stderr else None
            }
        else:
            logger.debug(f"Functional mismatch: {error_rate*100:.1f}% of test cases failed (code compiled and ran successfully)")
            return {
                "correct": False,
                "error_rate": error_rate,
                "stdout": stdout,  # Full output
                "stderr": stderr if stderr else None
            }

    except Exception as e:
        logger.error(f"Exception during Verilog verification: {e}", exc_info=True)
        return {"correct": False, "exception": str(e)}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: bytes,
    extra_info: dict = None,
    sandbox_fusion_url: str = "http://localhost:8080/run_code",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    format_type: str = "auto",
    **kwargs
):
    """
    Compute score for CodeV Verilog generation task.

    Supports both XML (<think>, <answer>) and GPT OSS (<|channel|>analysis, <|channel|>final) formats.

    Args:
        data_source: Data source identifier (should be 'codev')
        solution_str: Generated solution string (may contain thinking and answer sections)
        ground_truth: Pickled ground truth data containing golden Verilog code and port info
        extra_info: Optional extra information
        sandbox_fusion_url: URL of Sandbox Fusion service
        concurrent_semaphore: Optional semaphore for concurrency control
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)
        **kwargs: Additional arguments

    Returns:
        dict: Score dictionary with 'score', 'reward_fmt', 'reward_think' keys
    """
    try:
        # Progress tracking: Start
        logger.info(f"Starting CodeV scoring (format: {format_type})")
        # Parse ground truth from JSON or pickle (backward compatibility)
        if isinstance(ground_truth, bytes):
            # Check if bytes are actually pickle data (magic bytes: \x80\x03 or \x80\x04)
            # Pickle protocol 3 and 4 start with these magic bytes
            if len(ground_truth) >= 2 and ground_truth[:2] in [b'\x80\x03', b'\x80\x04']:
                # Legacy format: pickled bytes
                gts = pickle.loads(ground_truth)
            else:
                # Not pickle data - likely corrupted bytes from base64 decode of JSON
                # Try to decode as UTF-8 and parse as JSON
                logger.warning("Bytes do not have pickle magic bytes, attempting UTF-8 decode + JSON parse")
                try:
                    ground_truth = ground_truth.decode('utf-8')
                    gts = json.loads(ground_truth)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    error_msg = f"Ground truth bytes are not valid pickle or JSON: {e}"
                    logger.debug(error_msg)
                    return {
                        "score": 0.0,
                        "reward_fmt": 0.0,
                        "reward_think": 0.0,
                        "error": error_msg,
                        "error_category": "system_error",
                        "error_summary": "Invalid ground truth format (failed to decode as pickle or JSON)"
                    }
        elif isinstance(ground_truth, str):
            # New format: JSON string
            gts = json.loads(ground_truth)
        else:
            error_msg = f"Unexpected ground_truth type: {type(ground_truth)}"
            logger.debug(error_msg)
            return {
                "score": 0.0,
                "reward_fmt": 0.0,
                "reward_think": 0.0,
                "error": error_msg,
                "error_category": "system_error",
                "error_summary": f"Ground truth must be bytes or str, got {type(ground_truth).__name__}"
            }

        # Convert lists back to sets for port info (JSON doesn't support sets or tuples)
        # This applies to both JSON string and bytes-decoded-to-JSON paths
        if isinstance(gts, dict):
            for variant in gts.values():
                if isinstance(variant, dict):
                    for key in ['input_port_width', 'output_port_width',
                               'clock_port_polarity', 'reset_port_polarity_sync']:
                        if key in variant and isinstance(variant[key], list):
                            # Convert inner lists to tuples, then create set
                            # Original: {('port', width), ...} → JSON: [['port', width], ...] → Back: {('port', width), ...}
                            variant[key] = set(
                                tuple(item) if isinstance(item, list) else item
                                for item in variant[key]
                            )

        # Extract assistant response from chat template wrapper (format-aware)
        # This handles Llama, Qwen, GPT OSS, and raw formats
        if format_type == "auto":
            format_type = detect_format(solution_str)

        handler = get_format_handler(format_type)
        solution_str = handler.extract_assistant_response(solution_str, model_type="auto")

        # Check format (must have proper tags/channels)
        if not check_format(solution_str, format_type=format_type):
            logger.debug("Invalid format: missing or incorrect tags/channels")
            return {
                "score": 0.0,
                "reward_fmt": 0.0,
                "reward_think": 0.0,
                "error": "Invalid format",
                "error_category": "format_error",
                "error_summary": "Response missing required <think>/<answer> tags or <|channel|> blocks"
            }

        # Extract Verilog code from answer block
        extracted_modules = extract_verilog(solution_str)  # Now returns list or None
        if not extracted_modules:
            logger.debug("No Verilog code extracted from answer")
            return {
                "score": 0.0,
                "reward_fmt": 1.0,  # Format is correct, but no code
                "reward_think": 1.0,
                "error": "No Verilog code found",
                "error_category": "extraction_error",
                "error_summary": "No Verilog code blocks found in answer section"
            }

        # Ensure extracted_modules is a list (backward compatibility)
        if isinstance(extracted_modules, str):
            extracted_modules = [extracted_modules]

        logger.debug(f"Extracted {len(extracted_modules)} module(s) from response")

        # Test against all ground truth variants
        # Some problems may have multiple acceptable solutions
        rewards = []
        verification_results = []

        for variant_key, gt_variant in gts.items():
            logger.debug(f"Testing against ground truth variant: {variant_key}")

            # Extract golden code and port information
            golden_code = gt_variant.get('code')
            if not golden_code:
                logger.error(f"No golden code in variant {variant_key}")
                continue

            port_info = (
                gt_variant.get('input_port_width', set()),
                gt_variant.get('output_port_width', set()),
                gt_variant.get('clock_port_polarity', set()),
                gt_variant.get('reset_port_polarity_sync', set())
            )

            # Parse golden code
            try:
                v = eda_tools(quiet=True)
                golden_top = v.auto_top(golden_code)
            except Exception as e:
                logger.error(f"Failed to parse GOLDEN Verilog code: {e}")
                logger.error(f"Golden code (first 300 chars): {golden_code[:300]!r}...")

                verification_results.append({
                    "correct": False,
                    "parse_error": f"Golden code parsing failed: {str(e)}",
                    "golden_code": golden_code if golden_code else None,  # Full code
                })
                rewards.append(0.0)
                continue

            # NEW: Test each extracted module individually
            variant_passed = False
            module_results = []

            for module_idx, module_code in enumerate(extracted_modules):
                logger.debug(f"Testing module {module_idx + 1}/{len(extracted_modules)} against variant {variant_key}")

                # Parse this specific module
                try:
                    gate_top = v.auto_top(module_code)
                except Exception as e:
                    logger.debug(f"Module {module_idx + 1} parsing failed: {e}")
                    logger.debug(f"Module code (first 300 chars): {module_code[:300]!r}...")
                    module_results.append({
                        "module_idx": module_idx,
                        "correct": False,
                        "parse_error": f"Module parsing failed: {str(e)}",
                        "module_code_preview": module_code[:300] if module_code else None
                    })
                    continue

                # Verify this module via Sandbox Fusion
                try:
                    result = verify_verilog_via_sandbox(
                        golden_code=golden_code,
                        dut_code=module_code,  # Test this specific module
                        golden_top=golden_top,
                        gate_top=gate_top,
                        port_info=port_info,
                        sandbox_fusion_url=sandbox_fusion_url,
                        concurrent_semaphore=concurrent_semaphore
                    )

                    result["module_idx"] = module_idx
                    module_results.append(result)

                    # If any module passes, consider this variant passed
                    if result.get("correct", False):
                        logger.debug(f"✓ Module {module_idx + 1} passed for variant {variant_key}")
                        variant_passed = True
                        break  # Early exit: one passing module is enough

                except Exception as e:
                    # Log unexpected exception and continue to next module
                    logger.error(f"Unexpected exception verifying module {module_idx + 1}: {e}", exc_info=True)
                    module_results.append({
                        "module_idx": module_idx,
                        "correct": False,
                        "unexpected_exception": str(e)
                    })
                    continue

            # Store results for this variant
            verification_results.append({
                "variant": variant_key,
                "passed": variant_passed,
                "num_modules_tested": len(module_results),
                "module_results": module_results[:3]  # Keep first 3 for debugging
            })

            rewards.append(1.0 if variant_passed else 0.0)

            # Early exit if we found a passing variant
            if variant_passed:
                logger.debug(f"✓ Found correct solution with variant {variant_key}")
                break

        # Compute final score (max across all variants)
        final_score = max(rewards) if rewards else 0.0

        # Check if we should raise an exception for actual errors (not functional mismatch)
        if final_score == 0.0 and verification_results:
            # Distinguish between functional mismatch (normal) and actual errors
            has_functional_mismatch = False
            error_msgs = []

            for variant_result in verification_results:
                # Check module_results within each variant
                module_results = variant_result.get("module_results", [])

                for module_result in module_results:
                    if not module_result.get("correct", False):
                        # Functional mismatch: code ran successfully but answer is wrong
                        # (has error_rate but no api_error, exception, or parse_error)
                        if ("error_rate" in module_result and
                            "exception" not in module_result and
                            "api_error" not in module_result and
                            "parse_error" not in module_result):
                            has_functional_mismatch = True
                            break

                        # Collect actual error messages
                        if "api_error" in module_result:
                            error_msgs.append(module_result["api_error"])
                        elif "exception" in module_result:
                            error_msgs.append(module_result["exception"])
                        elif "parse_error" in module_result:
                            error_msgs.append(module_result["parse_error"])
                        elif module_result.get("api_status") and module_result["api_status"] != "Success":
                            api_status = module_result.get("api_status")
                            compile_stderr = module_result.get("compile_stderr", "")
                            run_stderr = module_result.get("run_stderr", "")

                            # Build detailed error message with full stderr (no length limit)
                            error_parts = [f"API status: {api_status}"]
                            if compile_stderr:
                                error_parts.append(f"Compile error: {compile_stderr}")
                            if run_stderr:
                                error_parts.append(f"Run error: {run_stderr}")

                            error_msgs.append(" | ".join(error_parts))

                if has_functional_mismatch:
                    break

            # Return error if all modules failed with actual errors (not just wrong answers)
            if not has_functional_mismatch and error_msgs:
                error_msg = f"Verilog verification failed: {error_msgs[0]}"
                logger.debug(error_msg)
                return {
                    "score": 0.0,
                    "reward_fmt": 1.0,  # Format was valid (we extracted modules)
                    "reward_think": 1.0,  # Thinking was present
                    "error": error_msg,
                    "error_category": "verification_error",
                    "error_summary": f"All {len(extracted_modules)} module(s) failed with system errors (compile/API failure)",
                    "num_modules_extracted": len(extracted_modules),
                    "num_variants_tested": len(rewards),
                    "num_variants_passed": 0,
                    "verification_results": verification_results[:3]
                }

        # Progress tracking: End
        logger.info(
            f"CodeV scoring complete: score={final_score:.2f}, "
            f"modules_extracted={len(extracted_modules)}, "
            f"variants_passed={sum(rewards)}/{len(rewards)}"
        )

        result = {
            "score": final_score,
            "reward_fmt": 1.0,  # Format is correct if we got here
            "reward_think": 1.0,  # Thinking structure is present
            "num_modules_extracted": len(extracted_modules),
            "num_variants_tested": len(rewards),
            "num_variants_passed": sum(rewards),
            "verification_results": verification_results[:3]  # Include first 3 for debugging
        }

        # Add error categorization if all modules failed
        if final_score == 0.0:
            result["error_category"] = "verification_error"
            result["error_summary"] = f"All {len(extracted_modules)} module(s) failed verification across {len(rewards)} variant(s) (see verification_results)"

        return result

    except Exception as e:
        logger.error(f"Error in compute_score: {e}", exc_info=True)
        return {
            "score": 0.0,
            "reward_fmt": 0.0,
            "reward_think": 0.0,
            "error": f"Exception: {str(e)}",
            "error_category": "system_error",
            "error_summary": f"Unexpected exception during scoring: {type(e).__name__}"
        }
