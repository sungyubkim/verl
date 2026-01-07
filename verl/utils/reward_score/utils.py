import regex as re
import ast
import json
import os
from typing import Tuple, Optional, List, Any, Union, Dict
from collections import Counter

from .format_handlers import detect_format, get_format_handler


FORMATS_REGEX = [
    [r"\\boxed\{", "}"],
]


def parse_think(text: str, format_type: str = "auto") -> Tuple[str, bool]:
    """
    Remove thinking/reasoning section from text.

    This function is format-aware and supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats.

    Args:
        text: Full response text
        format_type: Format type ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        Tuple of (text_without_thinking, success)
        - text_without_thinking: Text with thinking section removed
        - success: True if thinking section was properly formatted (or absent)

    Examples:
        >>> parse_think("<think>reasoning</think>\\nThe answer is 42")
        ('The answer is 42', True)

        >>> parse_think("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n<|start|>assistant<|channel|>final<|message|>42<|return|>")
        ('<|start|>assistant<|channel|>final<|message|>42<|return|>', True)
    """
    if format_type == "auto":
        format_type = detect_format(text)

    handler = get_format_handler(format_type)
    thinking_content, success = handler.extract_thinking(text)

    if thinking_content is not None:
        # Remove thinking section from text
        text_without_thinking = handler.remove_thinking(text)
        return text_without_thinking, success
    else:
        # No thinking section found - return original text
        return text, success


def parse_answer(text: str, format_type: str = "auto") -> Tuple[str, int]:
    """
    Extract final answer from text.

    This function is format-aware and supports:
    - XML format: Looks for \\boxed{answer} or <response>/<answer> tags
    - GPT OSS format: Looks for <|channel|>final content

    Args:
        text: Full response text
        format_type: Format type ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        Tuple of (answer, format_index)
        - answer: Extracted answer or original text if no format matched
        - format_index: Index of matched format (-1 if no format matched, 0 for \\boxed, 1 for final channel)

    Examples:
        >>> parse_answer("The answer is \\\\boxed{42}")
        ('42', 0)

        >>> parse_answer("<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>")
        ('The answer is 42', 1)
    """
    if not isinstance(text, str):
        return text, -1

    # Detect format if auto
    if format_type == "auto":
        format_type = detect_format(text)

    # For GPT OSS format, try to extract from final channel first
    if format_type == "gpt_oss":
        handler = get_format_handler(format_type)
        final_response = handler.extract_final_response(text)
        if final_response:
            # Still try to extract \boxed{} from final response if present
            for i, pattern in enumerate(FORMATS_REGEX):
                match = extract_answer_recursive(final_response, pattern[0], pattern[1])
                if match:
                    return match, i
            # Return final channel content
            return final_response, 1

    # Try standard answer extraction patterns (e.g., \boxed{})
    last_match = None
    last_index = -1

    for i, pattern in enumerate(FORMATS_REGEX):
        match = extract_answer_recursive(text, pattern[0], pattern[1])
        if match:
            last_match = match
            last_index = i

    if last_match:
        return last_match, last_index

    # For XML format, try to extract from <response> or <answer> tags
    if format_type == "xml":
        handler = get_format_handler(format_type)
        final_response = handler.extract_final_response(text)
        if final_response:
            return final_response, 1

    return text, -1


def normalize_ground_truth(
    ground_truth: Union[str, Dict, List],
    key: str = "answer"
) -> Any:
    """
    Normalize ground truth from various formats to a single value.

    Handles common ground truth formats across different scorers:
    - Dict with key: {"answer": "value"} → "value"
    - List with single element: ["value"] → "value"
    - List with dict: [{"answer": "value"}] → "value"
    - Plain value: "value" → "value"

    Args:
        ground_truth: Ground truth in any supported format
        key: Dictionary key to extract (default: "answer")

    Returns:
        Extracted ground truth value

    Examples:
        >>> normalize_ground_truth({"answer": "Paris"})
        'Paris'

        >>> normalize_ground_truth("Paris")
        'Paris'

        >>> normalize_ground_truth(["Paris"])
        'Paris'

        >>> normalize_ground_truth([{"answer": "Paris"}])
        'Paris'
    """
    if isinstance(ground_truth, dict):
        return ground_truth.get(key, "")
    elif isinstance(ground_truth, list):
        if not ground_truth:
            return ""
        first = ground_truth[0]
        if isinstance(first, dict):
            return first.get(key, "")
        return first
    else:
        return ground_truth


def extract_answer_recursive(text: str, start_pattern: str, end_pattern: str) -> str:

    def find_matching_brace(s: str, start_idx: int) -> int:
        count = 0
        for i in range(start_idx, len(s)):
            if s[i] in ["(", "{", "["]:
                count += 1
            elif s[i] in [")", "}", "]"]:
                count -= 1
                if count == 0:
                    return i
        return -1

    matches = list(re.finditer(start_pattern, text))
    if not matches:
        return None

    results = []
    for match in matches:
        start_paren = match.end() - 1
        end_paren = find_matching_brace(text, start_paren)

        if end_paren != -1:
            after_paren = text[end_paren:]
            if after_paren.startswith(end_pattern):
                extracted = text[start_paren + 1 : end_paren]
                results.append(extracted)

    return results[-1] if results else None


def extract_content_from_tags(text: str, tag_name: str = "answer") -> Tuple[str, bool]:
    """
    Extract content from XML-like tags.

    Args:
        text: Response text potentially containing tags
        tag_name: Tag name (default: "answer")

    Returns:
        Tuple of (extracted_content, success)
        - extracted_content: Content from the last matching tag pair
        - success: True if tag pair was found

    Examples:
        >>> extract_content_from_tags("<answer>42</answer>")
        ('42', True)

        >>> extract_content_from_tags("<answer>first</answer>\\n<answer>second</answer>", "answer")
        ('second', True)

        >>> extract_content_from_tags("No tags here", "answer")
        ('', False)
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))

    if matches:
        # Take the last match (in case there are multiple)
        final_content = matches[-1].group(1).strip()
        return final_content, True

    return "", False


def parse_json_with_fallback(content: str, expected_type: Optional[str] = None) -> Any:
    """
    Parse JSON/Python literals with fallback strategy.

    This function tries multiple parsing strategies in order:
    1. ast.literal_eval (safer for Python literals)
    2. json.loads (for JSON strings)

    Args:
        content: String to parse
        expected_type: Optional type hint ('list', 'dict', 'str', 'array')

    Returns:
        Parsed object or None if parsing fails

    Examples:
        >>> parse_json_with_fallback('["a", "b", "c"]')
        ['a', 'b', 'c']

        >>> parse_json_with_fallback('{"key": "value"}')
        {'key': 'value'}

        >>> parse_json_with_fallback('plain text', expected_type='str')
        'plain text'
    """
    if not content:
        return None

    # For string type, just return the stripped content
    if expected_type == 'str':
        return content.strip()

    # Try ast.literal_eval first (safer for Python literals)
    try:
        parsed = ast.literal_eval(content)
        return parsed
    except (ValueError, SyntaxError):
        pass

    # Fallback to json.loads
    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        pass

    return None


def normalize_text(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Text to normalize
        lowercase: Convert to lowercase (default: True)
        strip: Strip whitespace (default: True)

    Returns:
        Normalized text

    Examples:
        >>> normalize_text("  Hello World  ")
        'hello world'

        >>> normalize_text("Hello World", lowercase=False)
        'Hello World'
    """
    if not isinstance(text, str):
        text = str(text)

    if strip:
        text = text.strip()
    if lowercase:
        text = text.lower()

    return text


def normalize_numeric(value: Union[int, float, str], precision: int = 2) -> Optional[float]:
    """
    Normalize numeric values for comparison.

    Args:
        value: Numeric value (int, float, or string)
        precision: Number of decimal places to round to (default: 2)

    Returns:
        Normalized float value or None if conversion fails

    Examples:
        >>> normalize_numeric("42.12345", precision=2)
        42.12

        >>> normalize_numeric(42)
        42.0

        >>> normalize_numeric("not a number")
        None
    """
    try:
        # Remove common formatting characters
        if isinstance(value, str):
            value = value.replace(',', '').replace('%', '').strip()

        result = float(value)
        return round(result, precision)
    except (ValueError, TypeError):
        return None


def compare_sets_unordered(pred: List, gt: List, normalize: bool = True) -> float:
    """
    Compare two lists as sets (order-independent).

    Computes the Jaccard similarity (intersection over union) between two lists
    treated as sets.

    Args:
        pred: Predicted list
        gt: Ground truth list
        normalize: Apply text normalization before comparison (default: True)

    Returns:
        Accuracy score between 0.0 and 1.0 (Jaccard similarity)

    Examples:
        >>> compare_sets_unordered(['a', 'b', 'c'], ['c', 'b', 'a'])
        1.0

        >>> compare_sets_unordered(['a', 'b'], ['b', 'c', 'd'])
        0.25  # 1 intersection / 4 union
    """
    # Normalize items if requested
    if normalize:
        pred = [normalize_text(str(x)) for x in pred]
        gt = [normalize_text(str(x)) for x in gt]
    else:
        pred = [str(x) for x in pred]
        gt = [str(x) for x in gt]

    pred_set = set(pred)
    gt_set = set(gt)

    # Handle empty ground truth
    if not gt_set:
        return 1.0 if not pred_set else 0.0

    # Handle empty prediction
    if not pred_set:
        return 0.0

    # Compute Jaccard similarity
    intersection = pred_set & gt_set
    union = pred_set | gt_set

    return len(intersection) / len(union) if union else 1.0


def match_score(list1: List, list2: List, strict: bool = False) -> float:
    """
    Compute a similarity score considering element frequency, ignoring order.

    Used by ToolRL for comparing tool calls and parameters.
    Supports strict matching mode for REFINEDREWARD environment variable.

    Args:
        list1: First list of elements
        list2: Second list of elements
        strict: If True, return 0.0 for any mismatch (REFINEDREWARD mode)

    Returns:
        float: Similarity score between 0.0 and 1.0

    Examples:
        >>> match_score(['a', 'b', 'c'], ['a', 'b', 'c'])
        1.0

        >>> match_score(['a', 'b'], ['b', 'c'])
        0.5

        >>> match_score(['a', 'b'], ['c', 'd'], strict=True)
        0.0
    """
    if list1 == list2:
        return 1.0

    # Strict matching mode (REFINEDREWARD)
    if strict or os.getenv("REFINEDREWARD", "0") == "1":
        return 0.0 if list1 != list2 else 1.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable.

    Supports common boolean representations: "1"/"0", "true"/"false", "yes"/"no".

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        bool: Boolean value from environment

    Examples:
        >>> os.environ["TEST_VAR"] = "1"
        >>> get_env_bool("TEST_VAR")
        True

        >>> get_env_bool("MISSING_VAR", default=False)
        False
    """
    value = os.getenv(key)
    if value is None:
        return default

    return value.lower() in ("1", "true", "yes", "on")
