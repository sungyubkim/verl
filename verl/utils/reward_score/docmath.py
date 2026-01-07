"""
Reward scorer for document math problems.

Expects answer format: "the answer is {number}"
Supports numeric answers with tolerance-based comparison.
Adapted from Qwen-Doc for use with datatrove.

Key differences from original:
- Simplified logging
- Compatible with datatrove interface
- LLM-as-judge disabled by default
- Think tags are required
"""

import re
import math
from typing import Dict, Optional, Union
from sympy import Rational
import numpy as np

from .utils import normalize_ground_truth
from .format_handlers import detect_format, get_format_handler


def is_number(string: str) -> bool:
    """Check if string represents a number."""
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)


def round_up_to_decimal(number: float, decimals: int) -> float:
    """Round up number to specified decimals."""
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def within_eps(pred: float, gt: float) -> bool:
    """Check if prediction is within epsilon tolerance of ground truth."""
    eps = abs(gt) * 0.0015
    return pred >= gt - eps and pred <= gt + eps


def normalize(prediction: str) -> Union[int, float, bool, str]:
    """
    Normalize answer string to comparable value.

    Handles:
    - Currency symbols (£, €, ¥, $)
    - Units (million, billion, thousand)
    - Percentages (%)
    - Scientific notation
    - Boolean values (true/false, yes/no)
    - Approximation keywords
    """
    # Preprocessing the string
    prediction = prediction.strip().rstrip('.')
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '0'

    # Remove currency and unit words
    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        prediction = prediction.replace(money, '')

    # Replace special tokens
    for symbol in ['=', '≈', '`', '%', '$', '°']:
        if symbol in prediction:
            if symbol == '=' or symbol == '≈':
                prediction = prediction.split(symbol)[-1].strip()
            else:
                prediction = prediction.replace(symbol, '')

    # Detect boolean keywords
    if prediction.lower() in ['true', 'yes', 'false', 'no']:
        return prediction.lower() in ['true', 'yes']

    if 'True' in prediction or 'False' in prediction:
        return 'True' in prediction

    # Detect approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()

    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop units before and after numbers
    # Pattern: "123 units" -> "123"
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        match = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction)
        if match:
            prediction = match.group(1)

    # Pattern: "units 123" -> "123"
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        match = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction)
        if match:
            prediction = match.group(1)

    # Pattern: "123km" -> "123"
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        match = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction)
        if match:
            prediction = match.group(1)

    # Pattern: "km123" -> "123"
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        match = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction)
        if match:
            prediction = match.group(1)

    # Handle scientific notation
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)

    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')

    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')

    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Handle multiple choice options
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        match = re.search(r'\([a-d]\)', prediction)
        if match:
            prediction = '"' + match.group(0) + '"'

    # If empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Convert to number/list/bool
    try:
        prediction = eval(prediction)
    except Exception:
        prediction = 0

    # Type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if prediction and isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif prediction and isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction


def compare_two_numbers(pred: Union[int, float], gt: Union[int, float]) -> bool:
    """
    Compare two numbers with tolerance.

    Handles:
    - Magnitude differences (e.g., 100 vs 1.0 for percentages)
    - Rounding to 3 decimal places
    - Epsilon-based tolerance
    """
    if not isinstance(pred, (int, float)):
        return False

    try:
        v1, v2 = max(abs(gt), abs(pred)), min(abs(gt), abs(pred))

        # Check if magnitude difference is exactly a power of 10
        if (v1 != 0 and v2 != 0) and int(math.log10(v1) - math.log10(v2)) == (math.log10(v1) - math.log10(v2)):
            return True

        # Check for percentage scaling (100x, 1000x, 100000x)
        if v2 <= v1 / 50 and within_eps(pred=v2*100, gt=v1):
            return True
        elif v2 <= v1 / 500 and within_eps(pred=v2*1000, gt=v1):
            return True
        elif v2 <= v1 / 50000 and within_eps(pred=v2*100000, gt=v1):
            return True

        # Check rounding to 3 decimals
        if round_up_to_decimal(v1, 3) == round_up_to_decimal(v2, 3):
            return True

        # Check epsilon tolerance
        return within_eps(pred=pred, gt=gt)
    except (OverflowError, ValueError):
        return False


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

        # Thinking is REQUIRED for docmath
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
        The final answer as a numeric value (string), or None if not found
    """
    if not response:
        return None

    # Remove unwanted characters
    response = response.replace('*', '')

    # Search for pattern: "the answer is {number}"
    match = re.search(r'the answer is (\=?\≈?\`?\%?\$?\°?\£?\€?\¥?-?[0-9\.,]+)', response, re.IGNORECASE)

    if match:
        # Remove commas and trailing dots
        res = match.group(1).replace(',', '').rstrip('.')
        return res

    return None


def compute_score(
    predict_str: str,
    ground_truth: Union[str, Dict],
    data_source: str = "docmath",
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for document math prediction.

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (str or dict with answer)
        data_source: Dataset identifier
        format_type: Format type ("auto", "xml", or "gpt_oss")
        **kwargs: Additional arguments (ignored)

    Returns:
        Dictionary with score, accurate_score, format_score
        - score: 1.0 if correct (within tolerance), 0.0 otherwise
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
    pred_answer_str = parse_model_answer(answer_text)

    if not pred_answer_str:
        return {
            "score": 0.0,
            "accurate_score": 0.0,
            "format_score": 0.0,
        }

    format_score = 1.0

    # Normalize ground truth
    gt_answer_str = normalize_ground_truth(ground_truth)

    # Parse ground truth answer
    gt_parsed_str = parse_model_answer(str(gt_answer_str))
    if not gt_parsed_str:
        # If pattern doesn't match, use direct value
        gt_parsed_str = str(gt_answer_str)

    # Normalize both answers
    try:
        pred_normalized = normalize(pred_answer_str)
        gt_normalized = normalize(gt_parsed_str)

        # Compare based on type
        if isinstance(gt_normalized, bool):
            accurate_score = 1.0 if pred_normalized == gt_normalized else 0.0
        elif isinstance(gt_normalized, (int, float)):
            accurate_score = 1.0 if compare_two_numbers(pred_normalized, gt_normalized) else 0.0
        else:
            # For other types, try direct comparison
            accurate_score = 1.0 if pred_normalized == gt_normalized else 0.0

    except Exception:
        # If normalization/comparison fails, score is 0
        accurate_score = 0.0

    return {
        "score": accurate_score,
        "accurate_score": accurate_score,
        "format_score": format_score,
    }
