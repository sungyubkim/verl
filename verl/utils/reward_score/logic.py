"""
Logic domain scoring for reasoning tasks including ordering puzzles, zebra puzzles,
graph problems, and ARC-AGI tasks.

Supports both XML and GPT-OSS formats:
- XML: <think>...</think> and optionally <answer>...</answer>
- GPT-OSS: <|channel|>analysis and <|channel|>final

Implements cascade reward system following the math.py pattern:
1. reward_think: Validates thinking section formatting (optional)
2. reward_fmt: Validates answer section formatting and extraction
3. score: Compares extracted answer with ground truth

Supported data sources:
- ordering_puzzle: List ordering with sequence matching
- zebra_puzzle: Structured grid validation
- graph_logical: String matching for graph problems
- arcagi1, arcagi2, barc: 2D array comparison
"""

import re
import ast
import json
from typing import Union, List, Dict, Any, Tuple
import numpy as np

from .utils import parse_think


def extract_answer_from_tags(text: str) -> Tuple[str, bool]:
    """
    Extract content from <answer>...</answer> tags, GPT-OSS final channel, or plain text.

    Supports multiple formats:
    - XML: <answer>...</answer> tags
    - GPT-OSS: <|start|>assistant<|channel|>final<|message|>...<|return|>
    - Qwen3: Plain text without tags

    Args:
        text: Response text potentially containing answer tags or plain text

    Returns:
        (extracted_content, success)

    Examples:
        With <answer> tags:
        >>> extract_answer_from_tags('<answer>["A", "B", "C"]</answer>')
        ('["A", "B", "C"]', True)

        With GPT-OSS final channel:
        >>> extract_answer_from_tags('<|start|>assistant<|channel|>final<|message|>["A", "B", "C"]<|return|>')
        ('["A", "B", "C"]', True)

        Without tags (Qwen3 format):
        >>> extract_answer_from_tags('["A", "B", "C"]')
        ('["A", "B", "C"]', True)
    """
    # Try to find <answer>...</answer> tags first
    pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))

    if matches:
        # Take the last match (in case there are multiple)
        final_answer = matches[-1].group(1).strip()
        return final_answer, True

    # Try to extract from GPT-OSS final channel
    gpt_oss_pattern = r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>'
    gpt_oss_match = re.search(gpt_oss_pattern, text, flags=re.DOTALL)
    if gpt_oss_match:
        final_answer = gpt_oss_match.group(1).strip()
        return final_answer, True

    # Fallback: Use plain text (Qwen3 format)
    # This allows models without <answer> tags in tokenizer to work
    if text.strip():
        return text.strip(), True

    return "", False


def parse_answer_content(content: str, expected_type: str) -> Any:
    """
    Parse extracted answer content based on expected type.

    Args:
        content: Extracted answer string
        expected_type: One of ['list', 'dict', 'str', 'array']

    Returns:
        Parsed answer or None if parsing fails
    """
    if not content:
        return None

    try:
        if expected_type == 'str':
            return content.strip()

        elif expected_type in ['list', 'dict', 'array']:
            # Try ast.literal_eval first (safer)
            try:
                parsed = ast.literal_eval(content)
                return parsed
            except (ValueError, SyntaxError):
                # Fallback to json.loads
                try:
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError:
                    return None

    except Exception as e:
        print(f"[logic] Error parsing answer content: {e}")
        return None

    return None


def compare_lists(predicted: List, ground_truth: List) -> float:
    """
    Compare two lists for exact sequence match.

    Args:
        predicted: Predicted list
        ground_truth: Ground truth list

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Convert both to lists if they are numpy arrays
    if isinstance(predicted, np.ndarray):
        predicted = predicted.tolist()
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    # Normalize strings (strip whitespace, lowercase)
    def normalize_item(item):
        if isinstance(item, str):
            return item.strip().lower()
        return item

    pred_normalized = [normalize_item(x) for x in predicted]
    gt_normalized = [normalize_item(x) for x in ground_truth]

    return 1.0 if pred_normalized == gt_normalized else 0.0


def compare_dicts(predicted: Dict, ground_truth: Dict) -> float:
    """
    Compare two dictionaries (for zebra puzzles).

    Zebra puzzles have structure: {'header': [...], 'rows': [[...], [...]]}
    Returns cell-by-cell accuracy.

    Args:
        predicted: Predicted dictionary
        ground_truth: Ground truth dictionary

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not isinstance(predicted, dict) or not isinstance(ground_truth, dict):
        return 0.0

    # Check if both have required keys
    if 'header' not in predicted or 'rows' not in predicted:
        return 0.0
    if 'header' not in ground_truth or 'rows' not in ground_truth:
        return 0.0

    # Compare headers
    pred_header = predicted['header']
    gt_header = ground_truth['header']

    # Convert to lists if numpy arrays
    if isinstance(pred_header, np.ndarray):
        pred_header = pred_header.tolist()
    if isinstance(gt_header, np.ndarray):
        gt_header = gt_header.tolist()

    if pred_header != gt_header:
        return 0.0  # Headers must match exactly

    # Compare rows cell by cell
    pred_rows = predicted['rows']
    gt_rows = ground_truth['rows']

    # Convert to lists if numpy arrays
    if isinstance(pred_rows, np.ndarray):
        pred_rows = pred_rows.tolist()
    if isinstance(gt_rows, np.ndarray):
        gt_rows = gt_rows.tolist()

    if len(pred_rows) != len(gt_rows):
        return 0.0  # Must have same number of rows

    total_cells = 0
    correct_cells = 0

    for pred_row, gt_row in zip(pred_rows, gt_rows):
        # Convert rows to lists if needed
        if isinstance(pred_row, np.ndarray):
            pred_row = pred_row.tolist()
        if isinstance(gt_row, np.ndarray):
            gt_row = gt_row.tolist()

        if len(pred_row) != len(gt_row):
            continue  # Skip mismatched rows

        for pred_cell, gt_cell in zip(pred_row, gt_row):
            total_cells += 1
            # Normalize string comparison
            pred_str = str(pred_cell).strip().lower()
            gt_str = str(gt_cell).strip().lower()
            if pred_str == gt_str:
                correct_cells += 1

    if total_cells == 0:
        return 0.0

    return correct_cells / total_cells


def compare_strings(predicted: str, ground_truth: str) -> float:
    """
    Compare two strings (for graph problems).

    Args:
        predicted: Predicted string
        ground_truth: Ground truth string

    Returns:
        1.0 if match (case-insensitive), 0.0 otherwise
    """
    pred_normalized = str(predicted).strip().lower()
    gt_normalized = str(ground_truth).strip().lower()

    return 1.0 if pred_normalized == gt_normalized else 0.0


def compare_arrays(predicted: List[List], ground_truth: List[List]) -> float:
    """
    Compare two 2D arrays (for ARC-AGI tasks).

    Returns pixel-by-pixel accuracy with automatic padding if sizes differ.

    Args:
        predicted: Predicted 2D array
        ground_truth: Ground truth 2D array

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    # Convert to lists if numpy arrays
    if isinstance(predicted, np.ndarray):
        predicted = predicted.tolist()
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    if not predicted or not ground_truth:
        return 0.0

    # Get dimensions
    pred_h = len(predicted)
    pred_w = len(predicted[0]) if predicted else 0
    gt_h = len(ground_truth)
    gt_w = len(ground_truth[0]) if ground_truth else 0

    # Use maximum dimensions for comparison
    max_h = max(pred_h, gt_h)
    max_w = max(pred_w, gt_w)

    total_pixels = max_h * max_w
    correct_pixels = 0

    for i in range(max_h):
        for j in range(max_w):
            # Get predicted value (0 if out of bounds)
            pred_val = 0
            if i < pred_h and j < pred_w:
                pred_val = predicted[i][j] if isinstance(predicted[i], (list, np.ndarray)) else 0

            # Get ground truth value (0 if out of bounds)
            gt_val = 0
            if i < gt_h and j < gt_w:
                gt_val = ground_truth[i][j] if isinstance(ground_truth[i], (list, np.ndarray)) else 0

            if pred_val == gt_val:
                correct_pixels += 1

    return correct_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_score(
    model_output: str,
    ground_truth: Union[str, List, Dict],
    data_source: str = "graph_logical",
    timeout_score: float = 0.0,
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for logic domain tasks with cascade rewards.

    Cascade reward system:
    1. reward_think: Check if <think> section is properly formatted (1.0 or 0.0)
    2. reward_fmt: Check if <answer> section exists and can be extracted (1.0 or 0.0)
       - Only computed if reward_think == 1.0 (cascade failure)
    3. score: Verify correctness by comparing with ground truth (0.0 to 1.0)
       - Only computed if reward_fmt == 1.0 (cascade failure)

    Args:
        model_output: The model's response text
        ground_truth: Expected answer (type varies by data_source)
        data_source: Dataset identifier to determine comparison logic
        timeout_score: Score to return on timeout (default 0.0)
        format_type: Format type for parse_think ("auto", "xml", "gpt_oss")
        **kwargs: Additional arguments (ignored)

    Returns:
        Dict with keys:
            - score: Correctness score (0.0 to 1.0, typically binary)
            - reward_think: Thinking format reward (0.0 or 1.0)
            - reward_fmt: Answer format reward (0.0 or 1.0)
    """
    # Initialize rewards
    reward_think = 0.0
    reward_fmt = 0.0
    score = 0.0

    # Step 1: Validate thinking section
    try:
        _, think_success = parse_think(model_output, format_type=format_type)
        reward_think = 1.0 if think_success else 0.0
    except Exception as e:
        print(f"[logic] Error in parse_think: {e}")
        reward_think = 0.0

    # Cascade failure: Only check format if thinking is valid
    if reward_think == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Step 2: Extract answer from <answer> tags
    try:
        # Remove thinking section first
        text_without_think, _ = parse_think(model_output, format_type=format_type)

        # Extract from <answer> tags
        answer_content, answer_success = extract_answer_from_tags(text_without_think)
        reward_fmt = 1.0 if answer_success else 0.0
    except Exception as e:
        print(f"[logic] Error extracting answer: {e}")
        reward_fmt = 0.0

    # Cascade failure: Only compute score if format is valid
    if reward_fmt == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Step 3: Parse and compare answer based on data source
    try:
        # Determine expected type and comparison function
        if "ordering_puzzle" in data_source:
            expected_type = 'list'
            parsed_answer = parse_answer_content(answer_content, expected_type)
            if parsed_answer is not None:
                score = compare_lists(parsed_answer, ground_truth)

        elif "zebra_puzzle" in data_source:
            expected_type = 'dict'
            parsed_answer = parse_answer_content(answer_content, expected_type)
            if parsed_answer is not None:
                score = compare_dicts(parsed_answer, ground_truth)

        elif "graph" in data_source:
            expected_type = 'str'
            parsed_answer = parse_answer_content(answer_content, expected_type)
            if parsed_answer is not None:
                score = compare_strings(parsed_answer, ground_truth)

        elif "arcagi" in data_source or "barc" in data_source:
            expected_type = 'array'
            parsed_answer = parse_answer_content(answer_content, expected_type)
            if parsed_answer is not None:
                score = compare_arrays(parsed_answer, ground_truth)
        else:
            # Unknown data source, try generic string comparison
            print(f"[logic] Warning: Unknown data source '{data_source}', using string comparison")
            parsed_answer = parse_answer_content(answer_content, 'str')
            if parsed_answer is not None:
                score = compare_strings(parsed_answer, str(ground_truth))

    except Exception as e:
        print(f"[logic] Error comparing answer: {e}")
        score = 0.0

    return {
        "score": float(score),
        "reward_think": float(reward_think),
        "reward_fmt": float(reward_fmt),
    }
