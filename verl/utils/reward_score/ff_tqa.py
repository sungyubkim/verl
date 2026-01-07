"""
Reward scorer for FF-TQA task: FeTaQA (Free-form Table Question Answering).

Supports JSON string format answers: {"answer": "free text response"}
Uses BLEU and ROUGE-L metrics for evaluation.
Adapted from Table-R1 for use with datatrove and Qwen models.

Key differences from original:
- Answer tags are optional (Qwen doesn't use <answer> tags)
- Think tags are optional
- Compatible with both XML and direct JSON formats
"""

import re
import json
from typing import Dict, Optional, Union, List
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Initialize scorers
bleu = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Pattern for strict format checking (with tags)
PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

# JSON answer patterns
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*.*?\})\s*```', re.DOTALL)
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*"answer"\s*:\s*.*?\})\s*```', re.DOTALL)
ANSWER_PATTERN_2 = re.compile(r'(\{\s*"answer"\s*:\s*.*?\})', re.DOTALL)


def parse_json(answer: str) -> Optional[Union[str, List]]:
    """Parse JSON string to extract answer."""
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        return data["answer"]
    except json.JSONDecodeError:
        return None


def extract_answer_pattern(predict_str: str) -> Optional[Union[str, List]]:
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


def extract_answer(predict_str: str) -> str:
    """
    Extract answer from prediction string.

    Supports both formats:
    1. <answer>{"answer": "text"}</answer> (with tags)
    2. {"answer": "text"} (without tags)
    3. Fallback to raw text if no JSON pattern found
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

    # Fallback: return the stripped text
    return predict_str.strip()


def normalize_answer(ans: Union[str, List, int, float]) -> str:
    """
    Normalize answer for comparison.

    Args:
        ans: Answer in any format (string, list, number)

    Returns:
        Normalized string
    """
    if isinstance(ans, str):
        return ans.strip()
    elif isinstance(ans, list):
        return " ".join([str(x).strip() for x in ans])
    else:
        return str(ans).strip()


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
    ground_truth: Union[List, str],
    data_source: str = "ff_tqa",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for FF-TQA prediction using BLEU and ROUGE-L.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (list or string)
        data_source: Dataset identifier
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, format_score, accurate_score, bleu_score, rouge_score
        - score: format_score + (bleu + rouge) / 2
        - format_score: 0.0 or 1.0 (optional format check)
        - accurate_score: (bleu + rouge) / 2
        - bleu_score: BLEU score (0.0-1.0)
        - rouge_score: ROUGE-L F1 score (0.0-1.0)
    """
    # Extract answer from prediction
    answer = extract_answer(predict_str)
    if answer is None or answer == "":
        return {
            "score": 0.0,
            "format_score": 0.0,
            "accurate_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
        }

    # Check format (optional - 0.0 or 1.0)
    format_score = 1.0 if format_check(predict_str) else 0.0

    # Normalize answer and ground truth
    answer = normalize_answer(answer)

    # Handle list format ground truth (take first element)
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if len(ground_truth) > 0 else ""
    ground_truth = normalize_answer(ground_truth)

    # Compute BLEU score (0-1 range)
    bleu_score = bleu.sentence_score(answer, [ground_truth]).score / 100

    # Compute ROUGE-L F1 score
    rougel_score = rouge.score(answer, ground_truth)['rougeL'].fmeasure

    # Average of BLEU and ROUGE-L
    accurate_score = (bleu_score + rougel_score) / 2

    return {
        "score": format_score + accurate_score,
        "format_score": format_score,
        "accurate_score": accurate_score,
        "bleu_score": bleu_score,
        "rouge_score": rougel_score,
    }
