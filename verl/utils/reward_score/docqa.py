"""
Reward scorer for document QA (free text answers).

Expects answer format: "the answer is {text}"
Uses exact match (EM), substring EM, and F1 score for evaluation.
Adapted from Qwen-Doc for use with datatrove.

Key differences from original:
- Simplified logging
- Compatible with datatrove interface
- LLM-as-judge disabled by default
- Think tags are required
"""

import re
import string
from typing import Dict, Optional, Union
from collections import Counter

from .utils import normalize_ground_truth
from .format_handlers import detect_format, get_format_handler


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison.

    Applies:
    - Lowercasing
    - Removing punctuation
    - Removing articles (a, an, the)
    - Normalizing whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple:
    """
    Compute F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer text
        ground_truth: Ground truth answer text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    # Special handling for yes/no/noanswer
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Compute exact match score (after normalization)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def sub_em(prediction: str, ground_truth: str) -> bool:
    """
    Compute substring exact match.

    Returns True if either:
    - ground_truth is substring of prediction, OR
    - prediction is substring of ground_truth
    """
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction)
    return (ground_truth in prediction) or (prediction in ground_truth)


def extract_solution(solution_str: str, format_type: str = "auto") -> Optional[str]:
    """
    Extract the final answer from the model's response string.

    Supports both XML and GPT-OSS formats. Thinking section is REQUIRED.

    Args:
        solution_str: Raw response string from the language model
        format_type: Format type ("auto", "xml", or "gpt_oss")

    Returns:
        Extracted answer text after thinking section, or None if format invalid or missing thinking
    """
    try:
        # Detect format if auto
        format_type_detected = detect_format(solution_str) if format_type == "auto" else format_type
        handler = get_format_handler(format_type_detected)

        # Extract thinking content
        thinking_content, success = handler.extract_thinking(solution_str)

        # Thinking is REQUIRED for docqa
        if thinking_content is None:
            return None

        # Check if format is valid
        if not success:
            return None

        # Remove thinking and return final answer
        text_without_think = handler.remove_thinking(solution_str)
        return text_without_think.strip() if text_without_think else None
    except Exception:
        return None


def parse_model_answer(response: str) -> Optional[str]:
    """
    Parse the final answer from the model's response text.

    Args:
        response: Text extracted from the model's response

    Returns:
        The final answer text, or None if not found
    """
    if not response:
        return None

    # Remove unwanted characters
    response = response.replace('*', '')

    # Check for "the answer is" pattern
    if "the answer is" not in response.lower():
        return None

    # Extract text after "the answer is"
    ans = response.lower().rsplit("the answer is", 1)[-1].strip()

    # Clean up special tokens (from model generation)
    ans = ans.replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '')
    ans = ans.strip().strip('.').strip()

    return ans if ans else None


def compute_score(
    predict_str: str,
    ground_truth: Union[str, Dict],
    data_source: str = "docqa",
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for document QA prediction.

    Uses substring exact match (sub-EM) as the primary metric, which is more
    lenient than strict exact match but stricter than F1 score.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (str or dict with answer)
        data_source: Dataset identifier
        format_type: Format type ("auto", "xml", or "gpt_oss")
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, accurate_score, format_score, em, sub_em, f1, precision, recall
        - score: Same as sub_em (primary metric)
        - accurate_score: Same as sub_em
        - format_score: 1.0 if answer extracted successfully, 0.0 otherwise
        - em: Exact match score (0.0 or 1.0)
        - sub_em: Substring exact match score (0.0 or 1.0)
        - f1: F1 score (0.0-1.0)
        - precision: Precision score (0.0-1.0)
        - recall: Recall score (0.0-1.0)
    """
    # Extract answer from prediction
    answer_text = extract_solution(predict_str, format_type=format_type)

    if not answer_text:
        return {
            "score": 0.0,
            "accurate_score": 0.0,
            "format_score": 0.0,
            "em": 0.0,
            "sub_em": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    # Parse model answer
    pred_answer = parse_model_answer(answer_text)

    if not pred_answer:
        return {
            "score": 0.0,
            "accurate_score": 0.0,
            "format_score": 0.0,
            "em": 0.0,
            "sub_em": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    format_score = 1.0

    # Normalize ground truth
    gt_answer = normalize_ground_truth(ground_truth)

    # Parse ground truth (same format as prediction)
    gt_parsed = parse_model_answer(str(gt_answer))
    if not gt_parsed:
        # If pattern doesn't match, use direct value
        gt_parsed = str(gt_answer)

    # Compute metrics
    em = 1.0 if exact_match_score(pred_answer, gt_parsed) else 0.0
    subem = 1.0 if sub_em(pred_answer, gt_parsed) else 0.0
    f1, precision, recall = f1_score(pred_answer, gt_parsed)

    # Use sub-EM as the primary score (more lenient than EM, stricter than F1)
    score = subem

    return {
        "score": score,
        "accurate_score": score,
        "format_score": format_score,
        "em": em,
        "sub_em": subem,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
