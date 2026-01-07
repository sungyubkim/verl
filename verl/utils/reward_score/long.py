"""
Reward scorer for long-context multiple choice QA.

Expects answer format: "The correct answer is (A)" or "The correct answer is A"
Supports options: A, B, C, D
Adapted from Qwen-Doc for use with datatrove.

Key differences from original:
- Simplified logging
- Compatible with datatrove interface
- Think tags are required
"""

import re
from typing import Dict, Optional, Union

from .utils import normalize_ground_truth
from .format_handlers import detect_format, get_format_handler


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

        # Thinking is REQUIRED for long
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
    Parse model's answer text to extract the multiple choice option.

    Args:
        response: Text extracted from model's response

    Returns:
        Single letter answer (A, B, C, or D), or None if not found
    """
    if not response:
        return None

    response = response.replace('*', '')

    # Try pattern: "The correct answer is (A)"
    match = re.search(r'The correct answer is \(([A-D])\)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try pattern: "The correct answer is A"
    match = re.search(r'The correct answer is ([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def compute_score(
    predict_str: str,
    ground_truth: Union[str, Dict],
    data_source: str = "long",
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for long-context multiple choice prediction.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (str or dict with answer)
        data_source: Dataset identifier
        format_type: Format type ("auto", "xml", or "gpt_oss")
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, accurate_score, format_score
        - score: 1.0 if correct, 0.0 otherwise
        - accurate_score: Same as score
        - format_score: 1.0 if answer extracted successfully, 0.0 otherwise
    """
    # Extract answer from prediction
    answer_text = extract_solution(predict_str, format_type=format_type)

    if not answer_text:
        return {
            "score": 0.0,
            "accurate_score": 0.0,
            "format_score": 0.0,
        }

    # Parse model answer
    pred_answer = parse_model_answer(answer_text)

    if not pred_answer:
        return {
            "score": 0.0,
            "accurate_score": 0.0,
            "format_score": 0.0,
        }

    format_score = 1.0

    # Normalize ground truth
    gt_answer = normalize_ground_truth(ground_truth)

    # Parse ground truth answer
    gt_parsed = parse_model_answer(gt_answer)

    if not gt_parsed:
        # If ground truth doesn't match pattern, try direct comparison
        gt_parsed = gt_answer.strip().upper()

    # Compare answers (case-insensitive)
    accurate_score = 1.0 if pred_answer == gt_parsed else 0.0

    return {
        "score": accurate_score,
        "accurate_score": accurate_score,
        "format_score": format_score,
    }
