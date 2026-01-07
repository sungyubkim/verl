"""
Reward scorer for TFV task: TabFact (Table Fact Verification).

Supports binary classification: entailed or refuted
Adapted from Table-R1 for use with datatrove and Qwen models.

Key differences from original:
- Answer tags are optional (Qwen doesn't use <answer> tags)
- Think tags are optional
- Compatible with both XML and direct JSON formats
"""

import re
import json
from typing import Dict, Optional, Union

from .utils import normalize_ground_truth

# Pattern for strict format checking (with tags)
PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

# JSON answer patterns
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*"(?:entailed|refuted)"\s*\})\s*```')
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*"(?:entailed|refuted)"\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*"answer"\s*:\s*"(?:entailed|refuted)"\s*\})')


def parse_json(answer: str) -> Optional[str]:
    """Parse JSON string to extract answer."""
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        if data["answer"] not in ["entailed", "refuted"]:
            return None
        return data["answer"]
    except json.JSONDecodeError:
        return None


def extract_answer_pattern(predict_str: str) -> Optional[str]:
    """Extract answer from JSON patterns."""
    # Try strict pattern first
    answer_match = ANSWER_PATTERN_1.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)

    # Try looser pattern
    answer_match = ANSWER_PATTERN_2.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)

    return None


def extract_answer(predict_str: str) -> Optional[str]:
    """
    Extract answer from prediction string.

    Supports both formats:
    1. <answer>{"answer": "entailed"}</answer> (with tags)
    2. {"answer": "entailed"} (without tags)
    """
    # Try extracting from <answer> tags first
    answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str)
    if answer_block_match is not None:
        answer = extract_answer_pattern(answer_block_match.group(1))
        if answer is not None:
            return answer

    # Try extracting directly (no tags)
    answer = extract_answer_pattern(predict_str)
    if answer is not None:
        return answer

    return None


def format_check(predict_str: str) -> bool:
    """
    Check if prediction follows strict format.

    For instruct models: requires <think>, <answer> tags and strict JSON format.
    """
    if PATTERN.fullmatch(predict_str):
        for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
            if predict_str.count(tag) != 1:
                return False
        answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str).group(1)
        answer_match = STRICT_ANSWER_PATTERN.search(answer_block_match)
        if answer_match is not None:
            answer = answer_match.group(1).strip()
            final_answer = parse_json(answer)
            if final_answer is None:
                return False
            return True

    return False


def compute_score(
    predict_str: str,
    ground_truth: Union[str, list],
    data_source: str = "tfv",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for TabFact prediction.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer ("entailed" or "refuted", or dict/list with answer)
        data_source: Dataset identifier
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, format_score, accurate_score, bleu_score, rouge_score
    """
    # Normalize ground truth
    ground_truth = normalize_ground_truth(ground_truth)

    # Extract answer from prediction
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    # Check format (optional - 0.0 or 1.0)
    format_score = 1.0 if format_check(predict_str) else 0.0

    # Check accuracy
    accurate_score = 1.0 if answer == ground_truth else 0.0

    return {
        "score": format_score + accurate_score,
        "format_score": format_score,
        "accurate_score": accurate_score,
        "bleu_score": 0.0,
        "rouge_score": 0.0,
    }
