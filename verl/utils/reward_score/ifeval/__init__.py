"""
IF Eval Reward Scorer for Instruction Following Tasks

Self-contained implementation with local constraint checkers.
Original code from open-instruct (Apache 2.0 licensed).

Compatible with:
- allenai/IF_multi_constraints_upto5
- sungyub/ifbench-verl
- GRPO training scripts

Supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats.
"""

import ast
import json
import logging
import re
from typing import Dict, Any

from .instructions_registry import INSTRUCTION_DICT
from ..format_handlers import remove_thinking as remove_thinking_format_aware


logger = logging.getLogger(__name__)


def remove_thinking_section(text: str, format_type: str = "auto") -> str:
    """
    Remove thinking/reasoning section from model output.

    This function is format-aware and supports:
    - XML format: <think>, <thinking> tags
    - GPT OSS format: <|channel|>analysis blocks
    - Text prefixes: "Think:", "Thinking:", "Reasoning:"

    Args:
        text: Model output text
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        Text with thinking sections removed

    Examples:
        XML format:
        >>> remove_thinking_section("<think>reasoning</think>\\nFinal answer")
        'Final answer'

        GPT OSS format:
        >>> remove_thinking_section("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n<|start|>assistant<|channel|>final<|message|>answer<|return|>")
        '<|start|>assistant<|channel|>final<|message|>answer<|return|>'
    """
    return remove_thinking_format_aware(text, format_type=format_type)


def compute_score(
    model_output: str,
    ground_truth: str,
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute IF Eval score for instruction following.

    Supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats with auto-detection.

    Args:
        model_output: Model's generated response
        ground_truth: Constraint specification as string (JSON format)
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)
        **kwargs: Additional arguments (unused)

    Returns:
        Dict with:
            - score: Average success rate across all constraints (0.0-1.0)
            - reward_fmt: Format reward (always 1.0)
            - reward_think: Thinking reward (always 1.0)

    Example:
        >>> gt = "[{'instruction_id': ['last_word:last_word_answer'], 'kwargs': [{'last_word': 'brief'}]}]"
        >>> output = "This is a test response that ends with brief"
        >>> result = compute_score(output, gt)
        >>> print(result['score'])
        1.0
    """
    # Validate inputs
    if not model_output or not model_output.strip():
        logger.warning("Empty model output received")
        return {"score": 0.0, "reward_fmt": 1.0, "reward_think": 1.0}

    if not ground_truth:
        logger.warning("Empty ground truth received")
        return {"score": 0.0, "reward_fmt": 1.0, "reward_think": 1.0}

    # Parse ground truth
    try:
        # First try ast.literal_eval for Python-style strings
        constraint_dict = ast.literal_eval(ground_truth)
        # Handle nested lists
        if isinstance(constraint_dict, list) and len(constraint_dict) > 0:
            constraint_dict = constraint_dict[0]
        # Handle string-encoded JSON
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
    except Exception as e:
        logger.error(f"Failed to parse ground_truth: {e}")
        logger.error(f"Ground truth value: {ground_truth[:200]}...")
        return {"score": 0.0, "reward_fmt": 1.0, "reward_think": 1.0}

    # Remove thinking section from model output (format-aware)
    answer = remove_thinking_section(model_output, format_type=format_type)

    # Get constraints
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]

    # Check each constraint
    rewards = []

    if len(answer) == 0:
        logger.warning("Empty answer after removing thinking section")
        return {"score": 0.0, "reward_fmt": 1.0, "reward_think": 1.0}

    for instruction_key, args in zip(instruction_keys, args_list):
        # Handle None args
        if args is None:
            args = {}

        # Filter out None values
        args = {k: v for k, v in args.items() if v is not None}

        # Get instruction class from registry
        instruction_cls = INSTRUCTION_DICT[instruction_key]
        instruction_instance = instruction_cls(instruction_key)

        # Build description with args
        instruction_instance.build_description(**args)

        # Check if answer follows the instruction
        if answer.strip() and instruction_instance.check_following(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    # Return average score
    score = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "score": float(score),
        "reward_fmt": 1.0,
        "reward_think": 1.0,
    }


__all__ = ['compute_score', 'INSTRUCTION_DICT', 'remove_thinking_section']
