"""
XML-like format handler for standard response format.

Handles responses with XML-like tags:
- <think>...</think> for reasoning
- <tool_call>...</tool_call> for tool invocations
- <response>...</response> for final answers
- <answer>...</answer> for answers (used in some scorers)
"""

import re
import json
from typing import Optional, Tuple, List, Dict, Any

from .base import BaseFormatHandler


class XMLFormatHandler(BaseFormatHandler):
    """Handler for XML-like response format (Qwen/Llama default)."""

    @property
    def format_name(self) -> str:
        return "xml"

    def detect(self, text: str) -> bool:
        """Detect XML format by looking for <think>, <tool_call>, or <response> tags."""
        xml_tags = ["<think>", "<tool_call>", "<response>", "<answer>"]
        return any(tag in text for tag in xml_tags)

    def extract_thinking(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract content from <think> or <thinking> tags.

        Returns:
            (thinking_content, success) where success indicates proper formatting
        """
        # Try <think> tag first
        tag_count_start = text.count("<think>")
        tag_count_end = text.count("</think>")

        if tag_count_start > 0:
            if tag_count_start == tag_count_end == 1:
                # Extract thinking content
                match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                if match:
                    return match.group(1).strip(), True
            return None, False

        # Try <thinking> tag as alternative
        tag_count_start = text.count("<thinking>")
        tag_count_end = text.count("</thinking>")

        if tag_count_start > 0:
            if tag_count_start == tag_count_end == 1:
                match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
                if match:
                    return match.group(1).strip(), True
            return None, False

        # No thinking tags found - this is OK
        return None, True

    def remove_thinking(self, text: str) -> str:
        """
        Remove <think> or <thinking> sections from text.

        Also removes text-based thinking prefixes like "Think:", "Thinking:", etc.
        """
        # Remove <think>...</think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove <thinking>...</thinking> tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove text-based thinking prefixes
        thinking_prefixes = [
            r'^\s*Think:.*?(?=\n\n|\n[A-Z]|\n$)',
            r'^\s*Thinking:.*?(?=\n\n|\n[A-Z]|\n$)',
            r'^\s*Reasoning:.*?(?=\n\n|\n[A-Z]|\n$)',
            r'^\s*Thought:.*?(?=\n\n|\n[A-Z]|\n$)',
        ]

        for pattern in thinking_prefixes:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)

        return text.strip()

    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract content from <response> or <answer> tags.
        """
        # Try <response> tag first
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try <answer> tag
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from <tool_call>...</tool_call> tags.

        Expected format:
        <tool_call>
        {"name": "tool_name", "parameters": {...}}
        {"name": "tool_name2", "parameters": {...}}
        </tool_call>
        """
        tools = []

        # Extract content between tags
        match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not match:
            return tools

        tool_content = match.group(1).strip()

        # Parse each line as JSON
        for line in tool_content.split('\n'):
            line = line.strip()
            if not line:
                continue

            try:
                tool_dict = json.loads(line)
                if isinstance(tool_dict, dict) and 'name' in tool_dict:
                    tools.append(tool_dict)
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue

        return tools

    def check_format(self, text: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Check if text follows the expected XML format structure.

        Args:
            text: Text to validate
            expected_components: List of expected components like ['think', 'tool_call', 'response']

        Returns:
            True if format is valid
        """
        if expected_components is None:
            # Default: expect at least <think> tags
            return text.count("<think>") == text.count("</think>") == 1

        # Check for specific components
        has_think = "<think>" in text and "</think>" in text
        has_tool_call = "<tool_call>" in text and "</tool_call>" in text
        has_response = "<response>" in text and "</response>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        component_map = {
            'think': has_think or "<thinking>" in text,
            'thinking': has_think or "<thinking>" in text,
            'tool_call': has_tool_call,
            'response': has_response,
            'answer': has_answer,
        }

        # All expected components must be present
        return all(component_map.get(comp, False) for comp in expected_components)

    def extract_assistant_response(self, text: str, model_type: str = "auto") -> str:
        """
        Extract assistant response from chat template wrapper.

        Supports:
        - Llama: <|start_header_id|>assistant<|end_header_id|>...<|eot_id|>
        - Qwen: <|im_start|>assistant...<|im_end|>
        - Raw: No wrapper
        """
        # Auto-detect model type
        if model_type == "auto":
            if "<|start_header_id|>assistant<|end_header_id|>" in text:
                model_type = "llama"
            elif "<|im_start|>assistant" in text:
                model_type = "qwen"
            else:
                # Assume raw format
                return text.strip()

        # Extract based on model type
        if model_type == "llama":
            parts = text.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                return parts[-1].split("<|eot_id|>")[0].strip()
        elif model_type == "qwen":
            parts = text.split("<|im_start|>assistant")
            if len(parts) > 1:
                return parts[-1].split("<|im_end|>")[0].strip()

        return text.strip()

    def compute_format_reward(
        self,
        response: str,
        ground_truth: str,
        max_reward: float = 1.0,
        min_reward: float = 0.0
    ) -> float:
        """
        Compute format reward for ToolRL XML responses.

        Validates response structure based on ground truth pattern:
        - <think> + <response> (no tool call)
        - <think> + <tool_call> (no response)
        - <think> + <tool_call> + <response>
        - <think> only (with optional trailing content for Qwen3 compatibility)

        Args:
            response: Model's response text
            ground_truth: Expected response text (used to determine expected pattern)
            max_reward: Reward for correct format
            min_reward: Reward for incorrect format

        Returns:
            Format reward score

        Examples:
            Qwen3 format (with trailing content):
            >>> handler.compute_format_reward(
            ...     "<think>reasoning</think>\\nFinal answer: 42",
            ...     "<think>...</think>\\nFinal answer: X"
            ... )
            1.0

            Traditional format with <response> tags:
            >>> handler.compute_format_reward(
            ...     "<think>reasoning</think>\\n<response>answer</response>",
            ...     "<think>...</think>\\n<response>...</response>"
            ... )
            1.0
        """
        reward = min_reward

        # Check what components ground truth has
        has_gt_response = "<response>" in ground_truth
        has_gt_tool_call = "<tool_call>" in ground_truth
        has_gt_answer = "<answer>" in ground_truth

        if has_gt_answer:
            # Pattern: <think>...</think>\n<answer>...</answer> (CodeV traditional)
            # OR: <think>...</think>\n[plain text with code] (CodeV Qwen3)
            if response.count("<think>") == 1 and response.count("</think>") == 1:
                # Check if response has <answer> tags (traditional format)
                if response.count("<answer>") == 1 and response.count("</answer>") == 1:
                    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>"
                    if re.search(pattern, response, re.DOTALL):
                        reward = max_reward
                # Or allow plain text after </think> (Qwen3 format)
                elif (response.find("<think>") < response.find("</think>") and
                      response.strip().startswith("<think>")):
                    reward = max_reward

        elif has_gt_response and not has_gt_tool_call:
            # Pattern: <think>...</think>\n<response>...</response>
            if (response.count("<think>") == 1 and response.count("</think>") == 1 and
                response.count("<response>") == 1 and response.count("</response>") == 1):
                # Remove $ anchor to allow trailing content after </response>
                pattern = r"^<think>.*?</think>\n<response>.*?</response>"
                if re.search(pattern, response, re.DOTALL):
                    reward = max_reward

        elif not has_gt_response and has_gt_tool_call:
            # Pattern: <think>...</think>\n<tool_call>\n...\n</tool_call>
            if (response.count("<think>") == 1 and response.count("</think>") == 1 and
                response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1):
                # Remove $ anchor to allow trailing content after </tool_call>
                pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>"
                if re.search(pattern, response, re.DOTALL):
                    reward = max_reward

        elif has_gt_response and has_gt_tool_call:
            # Pattern: <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>
            if (response.count("<think>") == 1 and response.count("</think>") == 1 and
                response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and
                response.count("<response>") == 1 and response.count("</response>") == 1):
                # Remove $ anchor to allow trailing content after </response>
                pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>"
                if re.search(pattern, response, re.DOTALL):
                    reward = max_reward

        else:
            # Pattern: <think>...</think> with optional trailing content (Qwen3 support)
            # Qwen3 doesn't use <response> tags, just outputs plain text after </think>
            if response.count("<think>") == 1 and response.count("</think>") == 1:
                # Verify proper order and positioning
                if (response.find("<think>") < response.find("</think>") and
                    response.strip().startswith("<think>")):
                    reward = max_reward

        return reward
