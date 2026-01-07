"""
Table reasoning scorer for Guru datasets with boxed answer format.

Handles:
- HiTab: Hierarchical table questions (supports multiple answers with |)
- MultiHierTT: Multi-hierarchical table questions
- FinQA: Financial table questions (numeric answers)

Answer format: \\boxed{answer} or \\boxed{ans1|ans2|ans3}
Qwen3 format: <think>...</think>\\boxed{answer} (NO <answer> tags!)

Cascade reward system:
1. reward_think: Validates <think> section (0.0 or 1.0)
2. reward_fmt: Validates \\boxed{} format (0.0 or 1.0)
3. score: Compares answer with ground truth (0.0 or 1.0)
"""

from typing import Dict, Any, Union, List
from .utils import (
    parse_think,
    parse_answer,
    normalize_text,
    normalize_numeric
)


def parse_multiple_answers(answer_str: str) -> List[str]:
    """
    Parse multiple answers separated by |.

    Args:
        answer_str: Answer string potentially containing multiple answers

    Returns:
        List of individual answers

    Examples:
        >>> parse_multiple_answers("42")
        ['42']
        >>> parse_multiple_answers("A|B|C")
        ['A', 'B', 'C']
    """
    if not isinstance(answer_str, str):
        answer_str = str(answer_str)

    # Split by | and strip whitespace
    answers = [a.strip() for a in answer_str.split('|')]
    return answers


def compare_numeric(pred: str, gt: str, tolerance: float = 1e-3) -> bool:
    """
    Compare two numeric values with tolerance.

    Args:
        pred: Predicted value
        gt: Ground truth value
        tolerance: Comparison tolerance

    Returns:
        True if values match within tolerance
    """
    pred_num = normalize_numeric(pred, precision=5)
    gt_num = normalize_numeric(gt, precision=5)

    if pred_num is None or gt_num is None:
        return False

    # Handle zero case separately
    if abs(gt_num) < 1e-10:
        return abs(pred_num - gt_num) < tolerance

    # Use relative tolerance for non-zero values
    return abs(pred_num - gt_num) / abs(gt_num) < tolerance


def compare_single_answer(pred: str, gt: str, is_numeric: bool = None) -> float:
    """
    Compare single answer (numeric or string).

    Args:
        pred: Predicted answer
        gt: Ground truth answer
        is_numeric: Whether to force numeric comparison (auto-detected if None)

    Returns:
        1.0 if match, 0.0 otherwise
    """
    # Try numeric comparison first if not explicitly set to string
    if is_numeric is None:
        # Auto-detect: try to convert both to numbers
        pred_num = normalize_numeric(pred, precision=5)
        gt_num = normalize_numeric(gt, precision=5)
        is_numeric = (pred_num is not None and gt_num is not None)

    if is_numeric:
        if compare_numeric(pred, gt):
            return 1.0
    else:
        # String comparison (case-insensitive)
        if normalize_text(pred) == normalize_text(gt):
            return 1.0

    return 0.0


def compare_multiple_answers(pred_answers: List[str], gt_answers: List[str]) -> float:
    """
    Compare multiple answers (for HiTab).

    Returns 1.0 only if:
    1. Same number of answers
    2. All answers match (order-independent)

    Args:
        pred_answers: List of predicted answers
        gt_answers: List of ground truth answers

    Returns:
        1.0 if all answers match, 0.0 otherwise
    """
    if len(pred_answers) != len(gt_answers):
        return 0.0

    # Normalize all answers
    pred_normalized = [normalize_text(str(a)) for a in pred_answers]
    gt_normalized = [normalize_text(str(a)) for a in gt_answers]

    # Check if sets are equal (order-independent)
    pred_set = set(pred_normalized)
    gt_set = set(gt_normalized)

    # Also try numeric comparison for each pair
    if pred_set == gt_set:
        return 1.0

    # If string comparison fails, try numeric comparison
    # This handles cases like ["142"] vs ["142.0"]
    if len(pred_answers) == len(gt_answers):
        all_match = True
        for p, g in zip(pred_answers, gt_answers):
            if normalize_text(str(p)) != normalize_text(str(g)):
                # Try numeric comparison
                if not compare_numeric(str(p), str(g)):
                    all_match = False
                    break
        if all_match:
            return 1.0

    return 0.0


def compute_score(
    model_output: str,
    ground_truth: Union[str, float, int],
    data_source: str = "hitab",
    timeout_score: float = 0.0,
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for table reasoning tasks with boxed answer format.

    Cascade reward system:
    1. reward_think: Check if <think> section is properly formatted (0.0 or 1.0)
    2. reward_fmt: Check if \\boxed{} format exists (0.0 or 1.0)
    3. score: Verify correctness by comparing with ground truth (0.0 or 1.0)

    Args:
        model_output: The model's response text (Qwen3: <think>...</think>\\boxed{answer})
        ground_truth: Expected answer (string, float, or int)
        data_source: Dataset identifier ("hitab", "multihier", "finqa")
        timeout_score: Score to return on timeout (default 0.0)
        format_type: Format type for parse_think ("auto", "xml", "gpt_oss")
        **kwargs: Additional arguments (ignored)

    Returns:
        Dict with keys:
            - score: Correctness score (0.0 or 1.0)
            - reward_think: Thinking format reward (0.0 or 1.0)
            - reward_fmt: Answer format reward (0.0 or 1.0)

    Examples:
        >>> compute_score("<think>Reasoning</think>\\nThe answer is \\boxed{42}", "42")
        {"score": 1.0, "reward_think": 1.0, "reward_fmt": 1.0}

        >>> compute_score("<think>Reasoning</think>\\nThe answer is \\boxed{A|B}", "A|B", "hitab")
        {"score": 1.0, "reward_think": 1.0, "reward_fmt": 1.0}
    """
    # Initialize rewards
    reward_think = 0.0
    reward_fmt = 0.0
    score = 0.0

    # Step 1: Validate thinking section
    # Qwen3 format: <think>...</think> followed by plain text (no <answer> tags)
    try:
        pred_without_think, think_success = parse_think(model_output, format_type=format_type)
        reward_think = 1.0 if think_success else 0.0
    except Exception as e:
        print(f"[table_boxed] Error in parse_think: {e}")
        reward_think = 0.0

    # Cascade failure: Only check format if thinking is valid
    if reward_think == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Step 2: Extract answer from \\boxed{} format
    # Qwen3: <think>...</think>\boxed{answer} (NOT <answer>\boxed{}</answer>)
    try:
        pred_answer, pred_format = parse_answer(pred_without_think, format_type=format_type)
        # pred_format: 0 for \boxed{}, -1 if no format matched
        reward_fmt = 1.0 if pred_format >= 0 else 0.0
    except Exception as e:
        print(f"[table_boxed] Error in parse_answer: {e}")
        reward_fmt = 0.0

    # Cascade failure: Only compute score if format is valid
    if reward_fmt == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Step 3: Compare answer with ground truth
    try:
        # Convert ground truth to string
        gt_str = str(ground_truth)

        # Check if multiple answers (contains |)
        if '|' in pred_answer or '|' in gt_str:
            # Multiple answers case (HiTab)
            pred_answers = parse_multiple_answers(pred_answer)
            gt_answers = parse_multiple_answers(gt_str)
            score = compare_multiple_answers(pred_answers, gt_answers)
        else:
            # Single answer case (FinQA, MultiHierTT, or single HiTab)
            # Auto-detect numeric vs string
            score = compare_single_answer(pred_answer, gt_str)

    except Exception as e:
        print(f"[table_boxed] Error comparing answer: {e}")
        score = 0.0

    return {
        "score": float(score),
        "reward_think": float(reward_think),
        "reward_fmt": float(reward_fmt),
    }
