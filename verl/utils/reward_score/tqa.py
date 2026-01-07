"""
Reward scorer for TQA tasks: WTQ (WikiTableQuestions), HiTab.

Supports JSON list format answers: {"answer": ["item1", "item2", ...]}
Adapted from Table-R1 for use with datatrove and Qwen models.

Key differences from original:
- Answer tags are optional (Qwen doesn't use <answer> tags)
- Think tags are optional
- Compatible with both XML and direct JSON formats
"""

import re
import json
from typing import Dict, List, Optional, Union

# Pattern for strict format checking (with tags)
PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

# JSON answer patterns
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*(?:\[[\s\S]*?\])\s*\})\s*```')
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*(?:\[[\s\S]*?\]|"[\s\S]*?"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*"answer"\s*:\s*(?:\[[\s\S]*?\]|"[\s\S]*?"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})')

# Number pattern for normalization
NUMBER_PATTERN = re.compile(r'^[+-]?(?:(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def parse_json(answer: str) -> Optional[List]:
    """Parse JSON string to extract answer list."""
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        if isinstance(data["answer"], list):
            return data["answer"]
        else:
            return [data["answer"]]
    except json.JSONDecodeError:
        return None


def extract_answer_pattern(predict_str: str) -> Optional[List]:
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


def extract_answer(predict_str: str) -> Optional[List]:
    """
    Extract answer from prediction string.

    Supports both formats:
    1. <answer>{"answer": [...]}</answer> (with tags)
    2. {"answer": [...]} (without tags)
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


def normalize_answer(answer: List) -> List:
    """Normalize answer list for comparison."""
    normalized_answer = []
    for x in answer:
        if isinstance(x, int) or isinstance(x, float):
            normalized_answer.append(float(x))
        elif isinstance(x, str):
            if NUMBER_PATTERN.match(x):
                try:
                    normalized_answer.append(float(x.replace(',', '')))
                except ValueError:
                    normalized_answer.append(x)
            else:
                normalized_answer.append(x)
        else:
            return []
    return normalized_answer


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
            for i in final_answer:
                if not isinstance(i, str):
                    return False
            return True

    return False


def compute_score(
    predict_str: str,
    ground_truth: Union[List, str],
    data_source: str = "tqa",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for TQA prediction.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (list or string)
        data_source: Dataset identifier
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, format_score, accurate_score, bleu_score, rouge_score
    """
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

    # Normalize predicted answer
    normalized_answer = normalize_answer(answer)
    if len(normalized_answer) == 0 or len(normalized_answer) > 100:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    # Normalize ground truth
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    normalized_ground_truth = normalize_answer(ground_truth)

    # Check format (optional - 0.0 or 1.0)
    format_score = 1.0 if format_check(predict_str) else 0.0

    # Check accuracy
    accurate_score = 0.0
    if len(normalized_answer) == len(normalized_ground_truth):
        used = [0] * len(normalized_answer)
        for i in range(len(normalized_answer)):
            for j in range(len(normalized_ground_truth)):
                if used[j] == 0:
                    if isinstance(normalized_answer[i], float) and isinstance(normalized_ground_truth[j], float):
                        if abs(normalized_answer[i] - normalized_ground_truth[j]) < 1e-2:
                            used[j] = 1
                            break
                    else:
                        if normalized_answer[i] == normalized_ground_truth[j]:
                            used[j] = 1
                            break
        if sum(used) == len(normalized_answer):
            accurate_score = 1.0

    return {
        "score": format_score + accurate_score,
        "format_score": format_score,
        "accurate_score": accurate_score,
        "bleu_score": 0.0,
        "rouge_score": 0.0,
    }
