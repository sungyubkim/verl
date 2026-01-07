"""
Base classes for response format handlers.

Format handlers provide a unified interface for parsing different response formats
(XML-like tags, GPT OSS tokens, etc.) used in reward scoring.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any


class BaseFormatHandler(ABC):
    """
    Abstract base class for response format handlers.

    Each format handler implements parsing logic for a specific response format,
    allowing scorers to be format-agnostic.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this format (e.g., 'xml', 'gpt_oss')."""
        pass

    @abstractmethod
    def detect(self, text: str) -> bool:
        """
        Check if text matches this format.

        Args:
            text: Text to check

        Returns:
            True if text appears to be in this format
        """
        pass

    @abstractmethod
    def extract_thinking(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract thinking/reasoning content from text.

        Args:
            text: Full response text

        Returns:
            Tuple of (thinking_content, success)
            - thinking_content: The extracted thinking text, or None if not found
            - success: True if thinking section was properly formatted
        """
        pass

    @abstractmethod
    def remove_thinking(self, text: str) -> str:
        """
        Remove thinking/reasoning sections from text.

        Args:
            text: Full response text

        Returns:
            Text with thinking sections removed
        """
        pass

    @abstractmethod
    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract final response/answer from text.

        Args:
            text: Full response text

        Returns:
            Final response content, or None if not found
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text.

        Args:
            text: Full response text

        Returns:
            List of tool call dicts with 'name' and 'parameters' keys
        """
        pass

    @abstractmethod
    def check_format(self, text: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Check if text follows the expected format structure.

        Args:
            text: Text to validate
            expected_components: Optional list of expected components
                                (e.g., ['thinking', 'tool_call', 'response'])

        Returns:
            True if format is valid
        """
        pass

    @abstractmethod
    def extract_assistant_response(self, text: str, model_type: str = "auto") -> str:
        """
        Extract assistant response from chat template wrapper.

        Args:
            text: Full text including possible chat template markers
            model_type: Model type hint (e.g., 'llama', 'qwen', 'auto')

        Returns:
            Extracted assistant response
        """
        pass

    def compute_format_reward(
        self,
        response: str,
        ground_truth: str,
        max_reward: float = 1.0,
        min_reward: float = 0.0
    ) -> float:
        """
        Compute format reward for tool learning tasks.

        Default implementation returns max_reward if basic format is valid,
        min_reward otherwise. Subclasses can override for format-specific scoring logic.

        Args:
            response: Model's response text
            ground_truth: Expected response text (used to determine expected pattern)
            max_reward: Reward for correct format
            min_reward: Reward for incorrect format

        Returns:
            Format reward score

        Note:
            This default implementation uses check_format() for basic validation.
            Subclasses like XMLFormatHandler and GPTOSSFormatHandler override this
            to implement format-specific pattern matching logic.
        """
        return max_reward if self.check_format(response) else min_reward
