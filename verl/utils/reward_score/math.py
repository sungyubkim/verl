"""
Math reward scoring for mathematical reasoning tasks.

Supports datasets like GSM8K, MATH, Numina, etc.
Works with both XML (<think>) and GPT OSS (<|channel|>analysis) formats.
"""

from math_verify import parse, verify

from .utils import parse_think, parse_answer


def compute_score(
    model_output: str,
    ground_truth: str,
    timeout_score: float = 0,
    format_type: str = "auto",
    **kwargs
) -> dict:
    """
    Compute math scoring reward.

    This function evaluates model outputs on three components:
    1. Think: Proper formatting of thinking/reasoning section
    2. Format: Answer in expected format (e.g., \\boxed{})
    3. Correctness: Mathematical equivalence to ground truth

    Args:
        model_output: Model's output text (may include thinking section)
        ground_truth: Expected answer
        timeout_score: Score to return on timeout (not used)
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)
        **kwargs: Additional arguments (ignored)

    Returns:
        dict: Reward scores with keys:
            - score: Correctness score (0 or 1)
            - reward_think: Binary indicator if thinking properly formatted
            - reward_fmt: Binary indicator if answer format matches ground truth

    Examples:
        XML format:
        >>> compute_score("<think>Let me solve...</think>\\n\\\\boxed{42}", "42")
        {'score': 1, 'reward_think': 1.0, 'reward_fmt': 1.0}

        GPT OSS format:
        >>> compute_score(
        ...     "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n"
        ...     "<|start|>assistant<|channel|>final<|message|>\\\\boxed{42}<|return|>",
        ...     "42"
        ... )
        {'score': 1, 'reward_think': 1.0, 'reward_fmt': 1.0}
    """
    reward_think = 0.0
    reward_fmt = 0.0
    reward_correct = 0.0

    # Remove thinking section and check if properly formatted
    # This now supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats
    pred, pass_think_parsed = parse_think(model_output, format_type=format_type)
    if pass_think_parsed:
        reward_think = 1.0

        # Extract answers (supports \\boxed{}, <response>, <|channel|>final, etc.)
        pred_parsed, pred_type = parse_answer(pred, format_type=format_type)
        # Ground truth should always be auto-detected, not forced to match model format
        gt_parsed, gt_type = parse_answer(ground_truth, format_type="auto")

        # Give format reward based on whether prediction matches expected format
        # Format types must match exactly to encourage consistent formatting
        if gt_type >= 0:
            # Ground truth has specific format (boxed, response, final channel, etc.)
            # Prediction must match the same format type
            if pred_type == gt_type:
                reward_fmt = 1.0
        else:
            # Ground truth is plain text (-1)
            # Prediction must also be plain text
            if pred_type == -1:
                reward_fmt = 1.0

        # Only verify mathematical equivalence if format check passed (cascade failure)
        if reward_fmt == 1.0:
            try:
                is_correct = verify(
                    parse(f"\\boxed{{{pred_parsed}}}"),
                    parse(f"\\boxed{{{gt_parsed}}}"),
                    strict=False,
                    float_rounding=2,
                )
                reward_correct = 1.0 if is_correct else 0.0
            except Exception:
                # If verification fails, answer is incorrect
                reward_correct = 0.0

    return {
        "score": reward_correct,
        "reward_think": reward_think,
        "reward_fmt": reward_fmt,
    }
