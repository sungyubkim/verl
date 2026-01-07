"""
ToolRL reward scoring for tool learning tasks.

This module implements the reward function from the ToolRL paper:
"ToolRL: Reward is All Tool Learning Needs" (Qian et al., 2025)

The reward function evaluates model outputs on three components:
1. Format: Correct structure (XML tags or GPT OSS channels)
2. Correctness: Tool name and parameter matching
3. Length (optional): Reasoning length in thinking sections

Supports both:
- XML format: <think>, <tool_call>, <response> tags (Qwen/Llama default)
- GPT OSS format: <|channel|>analysis, to=functions.X, <|channel|>final tokens

Reference: https://github.com/qiancheng0/ToolRL
"""

import re
import json
import os
from collections import Counter

from .format_handlers import detect_format, get_format_handler
from .utils import match_score, get_env_bool


def compute_format_reward(response, answer, max_reward=1.0, min_reward=0.0, format_type="auto"):
    """
    Evaluate format correctness of the response using unified format handlers.

    Supports both XML and GPT OSS formats with auto-detection.
    All format-specific validation logic is delegated to format handlers.

    Args:
        response: Model output string
        answer: Ground truth string
        max_reward: Maximum reward for correct format
        min_reward: Minimum reward for incorrect format
        format_type: Response format ("xml", "gpt_oss", or "auto")

    Returns:
        float: Format reward score
    """
    # Auto-detect format if needed
    if format_type == "auto":
        format_type = detect_format(response)

    # Delegate all format validation to format handlers
    handler = get_format_handler(format_type)
    return handler.compute_format_reward(response, answer, max_reward, min_reward)


def compute_tool_call_reward(gt_tools, pd_tools, max_reward=3.0, min_reward=-3.0):
    """
    Compute reward for tool call correctness.

    Evaluates:
    - Tool name matching (frequency-based)
    - Parameter key matching (frequency-based)
    - Parameter value matching (exact match)

    Supports environment variables:
    - COARSEREWARD=1: Binary matching (all or nothing)
    - INTERMEDIATEREWARD=1: Simplified intermediate scoring
    - REFINEDREWARD=1: Strict exact matching (via match_score)

    Args:
        gt_tools: List of ground truth tool dicts with "name" and "parameters"
        pd_tools: List of predicted tool dicts with "name" and "parameters"
        max_reward: Maximum possible reward
        min_reward: Minimum possible reward

    Returns:
        float: Correctness reward score
    """
    if gt_tools == pd_tools:
        return max_reward

    # COARSEREWARD: Binary matching
    if get_env_bool("COARSEREWARD"):
        return min_reward if gt_tools != pd_tools else max_reward

    # Score tool name matching
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    # INTERMEDIATEREWARD: Simplified scoring
    intermediate_mode = get_env_bool("INTERMEDIATEREWARD")

    # Match each gt_tool to best pd_tool
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if intermediate_mode:
            # Simplified: only count exact tool matches
            local_max_possible += 1.0
        else:
            # Full: count parameters too
            local_max_possible += 1.0 + len(gt_params)

        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            if intermediate_mode:
                # Simplified: only check exact match
                if gt_tool == pd_tool:
                    best_match_score = 1.0
                    best_match_index = i
                    break
            else:
                # Full: score parameters
                pd_params = pd_tool["parameters"]

                # Score parameter keys
                param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Score parameter values
                correctness_score = sum(
                    1.0 for k, v in gt_params.items()
                    if k in pd_params and pd_params[k] == v
                )

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match_index = i

        if best_match_index != -1:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    # Normalize to reward range
    normalized_score = score / local_max_possible if local_max_possible > 0 else 0.0
    return (max_reward - min_reward) * normalized_score + min_reward


def compute_correctness_reward(response, answer, max_reward=3.0, min_reward=-3.0, format_type="auto"):
    """
    Evaluate correctness of tool calls in the response.

    Supports both XML (<tool_call>) and GPT OSS (to=functions.X) formats.

    Args:
        response: Model output string
        answer: Ground truth string
        max_reward: Maximum reward for correct tool calls
        min_reward: Minimum reward for incorrect tool calls
        format_type: Response format ("xml", "gpt_oss", or "auto")

    Returns:
        float: Correctness reward score
    """
    # Auto-detect format if needed
    if format_type == "auto":
        format_type = detect_format(response)

    handler = get_format_handler(format_type)

    # Check if tool call is expected
    gt_tools = handler.extract_tool_calls(answer)
    if not gt_tools:
        return 0.0

    # Extract predicted tools
    try:
        pd_tools = handler.extract_tool_calls(response)

        if not pd_tools:
            return min_reward

        return compute_tool_call_reward(gt_tools, pd_tools, max_reward, min_reward)
    except Exception:
        return min_reward


def compute_length_reward(response, max_reward=1.0, min_reward=0.0, max_words=512, format_type="auto"):
    """
    Reward longer reasoning in thinking sections.

    Supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats.

    Args:
        response: Model output string
        max_reward: Maximum reward for optimal length
        min_reward: Minimum reward for short/missing reasoning
        max_words: Target word count for maximum reward
        format_type: Response format ("xml", "gpt_oss", or "auto")

    Returns:
        float: Length reward score
    """
    # Auto-detect format if needed
    if format_type == "auto":
        format_type = detect_format(response)

    handler = get_format_handler(format_type)

    # Extract thinking content
    think_content, success = handler.extract_thinking(response)

    if not think_content:
        return min_reward

    word_count = len(think_content.split())

    # Linear scaling up to max_words
    reward_ratio = min(word_count / max_words, 1.0)

    return reward_ratio * (max_reward - min_reward) + min_reward


def extract_assistant_response(solution_str, model_type="auto", format_type="auto"):
    """
    Extract assistant response from chat template.

    Supports:
    - Llama format: <|start_header_id|>assistant<|end_header_id|>...<|eot_id|>
    - Qwen format: <|im_start|>assistant...<|im_end|>
    - GPT OSS format: <|start|>assistant...<|end|>/<|return|>/<|call|>
    - Raw format: Direct response without chat template

    Args:
        solution_str: Full model output including chat template
        model_type: Model type ("llama", "qwen", or "auto" for detection)
        format_type: Response format ("xml", "gpt_oss", or "auto")

    Returns:
        str: Extracted assistant response
    """
    # Auto-detect format if needed
    if format_type == "auto":
        format_type = detect_format(solution_str)

    # Use format handler for extraction
    handler = get_format_handler(format_type)
    return handler.extract_assistant_response(solution_str, model_type)


def compute_score(
    model_output: str,
    ground_truth: str,
    step: int = 0,
    model_type: str = "auto",
    enable_length_reward: bool = False,
    format_type: str = "auto",
    **kwargs
) -> dict:
    """
    Compute ToolRL reward score for tool learning tasks.

    This function evaluates model outputs on three components:
    1. Format: Structure validation (XML tags or GPT OSS channels)
    2. Correctness: Tool name and parameter matching
    3. Length (optional): Reasoning length in thinking sections

    Supports both XML and GPT OSS formats with automatic detection.

    Environment variables (VERL compatibility):
    - WITHLENGTH=1: Auto-enable length reward
    - CORRECTMAX1=1: Set correctness max reward to 1 (default: 3)
    - SCHEDULEREWARD=1: Apply step-based reward scaling
    - REFINEDREWARD=1: Strict exact matching (no partial credit)
    - COARSEREWARD=1: Binary match/no-match scoring
    - INTERMEDIATEREWARD=1: Simplified intermediate scoring

    Args:
        model_output: Full model output string (may include chat template)
        ground_truth: Expected output from reward_model.ground_truth
        step: Training step number (for dynamic reward scaling)
        model_type: Model type for response extraction ("llama", "qwen", "auto")
        enable_length_reward: Whether to include length reward component
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)
        **kwargs: Additional arguments (ignored)

    Returns:
        dict: Reward scores with keys:
            - score: Total reward (sum of all components)
            - reward_fmt: Format reward (0 to 1)
            - reward_correct: Correctness reward (-3 to 3, or -1 to 1 if CORRECTMAX1=1)
            - reward_length: Length reward (0 to 1, if enabled)
            - reward_think: Binary indicator if thinking section present

    Examples:
        XML format:
        >>> compute_score("<think>reasoning</think>\\n<tool_call>\\n{...}\\n</tool_call>", "...")
        {'score': 4.0, 'reward_fmt': 1.0, 'reward_correct': 3.0, ...}

        GPT OSS format:
        >>> compute_score("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>...", "...")
        {'score': 4.0, 'reward_fmt': 1.0, 'reward_correct': 3.0, ...}
    """
    # Check environment variables
    if get_env_bool("WITHLENGTH"):
        enable_length_reward = True

    # Set reward ranges based on environment
    if get_env_bool("CORRECTMAX1"):
        tool_max = 1.0
        tool_min = -1.0
    else:
        tool_max = 3.0
        tool_min = -3.0

    format_max = 1.0
    format_min = 0.0
    length_max = 1.0
    length_min = 0.0

    # Apply SCHEDULEREWARD if enabled
    if get_env_bool("SCHEDULEREWARD"):
        # Reward scheduling based on step
        # Format: gradually reduce range as training progresses
        format_max = 2.0 - (2.0 - 1.0) * min(step / 150.0, 1.0)
        format_min = -2.0 + (2.0 - 0.0) * min(step / 150.0, 1.0)
        if format_max < 1.0:
            format_max = 1.0
        if format_min > -1.0:
            format_min = -1.0

        # Correctness: gradually increase range as training progresses
        scheduled_tool_max = (tool_max - 2.0) * min(step / 150.0, 1.0) + 2.0
        scheduled_tool_min = (tool_min + 2.0) * min(step / 150.0, 1.0) - 2.0
        if scheduled_tool_max > tool_max:
            scheduled_tool_max = tool_max
        if scheduled_tool_min < tool_min:
            scheduled_tool_min = tool_min
        tool_max = scheduled_tool_max
        tool_min = scheduled_tool_min

    # Extract assistant response (format-aware)
    response = extract_assistant_response(model_output, model_type, format_type)

    # Auto-detect format for scoring if needed
    if format_type == "auto":
        format_type = detect_format(response)

    # Compute component scores (all format-aware)
    format_score = compute_format_reward(response, ground_truth, max_reward=format_max, min_reward=format_min, format_type=format_type)
    correctness_score = compute_correctness_reward(response, ground_truth, max_reward=tool_max, min_reward=tool_min, format_type=format_type)

    # Optional length reward
    length_score = 0.0
    if enable_length_reward:
        # SCHEDULELENGTH: Dynamic max words threshold
        if get_env_bool("SCHEDULELENGTH"):
            max_words = int((640 - 384) * min(step / 105.0, 1.0) + 384)
        else:
            max_words = 512

        length_score = compute_length_reward(response, max_reward=length_max, min_reward=length_min, max_words=max_words, format_type=format_type)

    # Binary indicator for thinking section presence (format-aware)
    handler = get_format_handler(format_type)
    think_content, _ = handler.extract_thinking(response)
    think_indicator = 1.0 if think_content else 0.0

    # Total score
    total_score = format_score + correctness_score + length_score

    return {
        "score": total_score,
        "reward_fmt": format_score,
        "reward_correct": correctness_score,
        "reward_length": length_score,
        "reward_think": think_indicator,
    }
