"""
GPT OSS 120B format handler.

Handles responses in GPT OSS format with special tokens:
- <|start|>assistant<|channel|>analysis<|message|>...<|end|> for reasoning
- <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{params}<|call|> for tool calls
- <|start|>assistant<|channel|>final<|message|>...<|return|> for final answers

Reference: https://huggingface.co/openai/gpt-oss-120b
"""

import re
import json
from typing import Optional, Tuple, List, Dict, Any

from .base import BaseFormatHandler


class GPTOSSFormatHandler(BaseFormatHandler):
    """Handler for GPT OSS 120B response format."""

    @property
    def format_name(self) -> str:
        return "gpt_oss"

    def detect(self, text: str) -> bool:
        """Detect GPT OSS format by looking for <|start|>assistant and <|channel|> tokens."""
        return '<|start|>assistant' in text and '<|channel|>' in text

    def extract_thinking(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract content from analysis channel.

        Format: <|start|>assistant<|channel|>analysis<|message|>...<|end|>

        Returns:
            (thinking_content, success) where success indicates proper formatting
        """
        pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip(), True

        # Check if analysis channel is present but malformed
        if '<|channel|>analysis' in text:
            return None, False

        # No analysis channel - this is OK
        return None, True

    def remove_thinking(self, text: str) -> str:
        """
        Remove analysis channel sections from text.
        """
        pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
        return text.strip()

    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract content from final channel.

        Format: <|start|>assistant<|channel|>final<|message|>...<|return|>
        """
        pattern = r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        return None

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from GPT OSS format.

        Format: <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{params}<|call|>

        Returns:
            List of tool call dicts with 'name' and 'parameters' keys
        """
        tools = []

        # Pattern to match tool calls
        pattern = r'<\|start\|>assistant to=functions\.(\w+)<\|channel\|>commentary json<\|message\|>(.*?)<\|call\|>'

        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            tool_name = match.group(1)
            params_json = match.group(2).strip()

            try:
                parameters = json.loads(params_json)
                tools.append({
                    "name": tool_name,
                    "parameters": parameters
                })
            except json.JSONDecodeError as e:
                # Add with raw string if parsing fails
                tools.append({
                    "name": tool_name,
                    "parameters": {"_raw": params_json, "_error": str(e)}
                })

        return tools

    def check_format(self, text: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Check if text follows the expected GPT OSS format structure.

        Args:
            text: Text to validate
            expected_components: List of expected components like ['thinking', 'tool_call', 'final']

        Returns:
            True if format is valid
        """
        # Check for basic GPT OSS tokens
        has_start = '<|start|>' in text
        has_channel = '<|channel|>' in text
        has_message = '<|message|>' in text

        if not (has_start and has_channel and has_message):
            return False

        if expected_components is None:
            # Default: just check for valid GPT OSS tokens
            return True

        # Check for specific components
        has_analysis = '<|channel|>analysis' in text and '<|end|>' in text
        has_tool_call = 'to=functions.' in text and '<|call|>' in text
        has_final = '<|channel|>final' in text and '<|return|>' in text

        component_map = {
            'thinking': has_analysis,
            'analysis': has_analysis,
            'tool_call': has_tool_call,
            'final': has_final,
            'response': has_final,
        }

        # All expected components must be present
        return all(component_map.get(comp, False) for comp in expected_components)

    def extract_assistant_response(self, text: str, model_type: str = "auto") -> str:
        """
        Extract assistant response from GPT OSS format.

        GPT OSS uses its own wrapper format, so we extract all assistant messages.

        Format: Multiple <|start|>assistant...<|end|> or <|call|> or <|return|> blocks
        """
        # Find all assistant messages
        pattern = r'<\|start\|>assistant.*?(?:<\|end\|>|<\|return\|>|<\|call\|>)'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Join all assistant messages
            return '\n'.join(matches)

        # Fallback: try to extract everything after last <|start|>assistant
        if '<|start|>assistant' in text:
            return '<|start|>assistant' + text.split('<|start|>assistant')[-1]

        # No GPT OSS format found, return as-is
        return text.strip()

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains tool call in GPT OSS format."""
        return 'to=functions.' in text and '<|call|>' in text

    def has_final_response(self, text: str) -> bool:
        """Check if text contains final response in GPT OSS format."""
        return '<|channel|>final' in text and '<|return|>' in text

    def has_analysis(self, text: str) -> bool:
        """Check if text contains analysis channel in GPT OSS format."""
        return '<|channel|>analysis' in text

    def compute_format_reward(
        self,
        response: str,
        ground_truth: str,
        max_reward: float = 1.0,
        min_reward: float = 0.0
    ) -> float:
        """
        Compute format reward for GPT OSS responses.

        Checks if response follows the same pattern as ground truth:
        - Pattern 1: Analysis + Tool call
        - Pattern 2: Analysis + Final response
        - Pattern 3: Analysis + Tool call + Final response
        - Pattern 4: Analysis only

        Args:
            response: Model output
            ground_truth: Expected output
            max_reward: Maximum reward for correct format
            min_reward: Minimum reward for incorrect format

        Returns:
            Format reward score
        """
        # Check ground truth pattern
        has_tool_call_in_ans = self.has_tool_call(ground_truth)
        has_final_in_ans = self.has_final_response(ground_truth)
        has_analysis_in_ans = self.has_analysis(ground_truth)

        # Check response pattern
        has_tool_call_in_resp = self.has_tool_call(response)
        has_final_in_resp = self.has_final_response(response)
        has_analysis_in_resp = self.has_analysis(response)

        # Pattern 1: Analysis + Tool call (most common)
        if has_analysis_in_ans and has_tool_call_in_ans and not has_final_in_ans:
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant to=functions\.\w+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>'
            if re.search(pattern, response, re.DOTALL):
                if has_analysis_in_resp and has_tool_call_in_resp:
                    return max_reward

        # Pattern 2: Analysis + Final response (no tool call)
        elif has_analysis_in_ans and has_final_in_ans and not has_tool_call_in_ans:
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant<\|channel\|>final<\|message\|>.*?<\|return\|>'
            if re.search(pattern, response, re.DOTALL):
                if has_analysis_in_resp and has_final_in_resp:
                    return max_reward

        # Pattern 3: Analysis + Tool call + Final response (rare)
        elif has_analysis_in_ans and has_tool_call_in_ans and has_final_in_ans:
            pattern = r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>.*?<\|start\|>assistant to=functions\.\w+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>.*?<\|start\|>assistant<\|channel\|>final<\|message\|>.*?<\|return\|>'
            if re.search(pattern, response, re.DOTALL):
                if has_analysis_in_resp and has_tool_call_in_resp and has_final_in_resp:
                    return max_reward

        # Pattern 4: Analysis only (very rare)
        elif has_analysis_in_ans and not has_tool_call_in_ans and not has_final_in_ans:
            pattern = r'^<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>$'
            if re.search(pattern, response, re.DOTALL):
                return max_reward

        return min_reward
