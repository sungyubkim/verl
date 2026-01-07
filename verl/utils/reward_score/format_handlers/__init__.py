"""
Format handlers for different response formats.

This module provides auto-detection and unified API for parsing different
response formats used in reward scoring:
- XML format: <think>, <tool_call>, <response> tags (default for Qwen/Llama)
- GPT OSS format: <|channel|>analysis, to=functions.X, <|channel|>final tokens

Usage:
    from datatrove.utils.reward_score.format_handlers import detect_format, get_format_handler

    # Auto-detect format
    format_type = detect_format(text)
    handler = get_format_handler(format_type)

    # Extract thinking
    thinking, success = handler.extract_thinking(text)

    # Extract tool calls
    tools = handler.extract_tool_calls(text)
"""

from typing import Optional, Tuple, List, Dict, Any

from .base import BaseFormatHandler
from .xml_format import XMLFormatHandler
from .gpt_oss_format import GPTOSSFormatHandler


# Registry of available format handlers
_HANDLERS = {
    "xml": XMLFormatHandler(),
    "gpt_oss": GPTOSSFormatHandler(),
}


def detect_format(text: str, preference: Optional[str] = None) -> str:
    """
    Auto-detect response format from text.

    Detection priority:
    1. If preference is specified and detected, use it
    2. GPT OSS format (more specific tokens)
    3. XML format (default fallback)

    Args:
        text: Text to analyze
        preference: Optional format preference (e.g., "gpt_oss", "xml")

    Returns:
        Format name ("gpt_oss", "xml")
    """
    # If preference is given, check if it matches
    if preference and preference in _HANDLERS:
        if _HANDLERS[preference].detect(text):
            return preference

    # Check GPT OSS first (more specific)
    if _HANDLERS["gpt_oss"].detect(text):
        return "gpt_oss"

    # Default to XML format
    return "xml"


def get_format_handler(format_type: str = "auto", text: Optional[str] = None) -> BaseFormatHandler:
    """
    Get a format handler instance.

    Args:
        format_type: Format type ("xml", "gpt_oss", or "auto")
        text: Optional text for auto-detection (required if format_type="auto")

    Returns:
        Format handler instance

    Raises:
        ValueError: If format_type="auto" but text is not provided
        ValueError: If format_type is unknown
    """
    if format_type == "auto":
        if text is None:
            raise ValueError("text parameter is required when format_type='auto'")
        format_type = detect_format(text)

    if format_type not in _HANDLERS:
        raise ValueError(f"Unknown format type: {format_type}. Available: {list(_HANDLERS.keys())}")

    return _HANDLERS[format_type]


# Convenience functions that auto-detect format

def extract_thinking(text: str, format_type: str = "auto") -> Tuple[Optional[str], bool]:
    """
    Extract thinking/reasoning content with auto-detection.

    Args:
        text: Full response text
        format_type: Format type or "auto" for auto-detection

    Returns:
        Tuple of (thinking_content, success)
    """
    handler = get_format_handler(format_type, text)
    return handler.extract_thinking(text)


def remove_thinking(text: str, format_type: str = "auto") -> str:
    """
    Remove thinking sections with auto-detection.

    Args:
        text: Full response text
        format_type: Format type or "auto" for auto-detection

    Returns:
        Text with thinking removed
    """
    handler = get_format_handler(format_type, text)
    return handler.remove_thinking(text)


def extract_final_response(text: str, format_type: str = "auto") -> Optional[str]:
    """
    Extract final response with auto-detection.

    Args:
        text: Full response text
        format_type: Format type or "auto" for auto-detection

    Returns:
        Final response content or None
    """
    handler = get_format_handler(format_type, text)
    return handler.extract_final_response(text)


def extract_tool_calls(text: str, format_type: str = "auto") -> List[Dict[str, Any]]:
    """
    Extract tool calls with auto-detection.

    Args:
        text: Full response text
        format_type: Format type or "auto" for auto-detection

    Returns:
        List of tool call dicts
    """
    handler = get_format_handler(format_type, text)
    return handler.extract_tool_calls(text)


def check_format(
    text: str,
    expected_components: Optional[List[str]] = None,
    format_type: str = "auto"
) -> bool:
    """
    Check if text follows expected format with auto-detection.

    Args:
        text: Text to validate
        expected_components: Optional list of expected components
        format_type: Format type or "auto" for auto-detection

    Returns:
        True if format is valid
    """
    handler = get_format_handler(format_type, text)
    return handler.check_format(text, expected_components)


def extract_assistant_response(
    text: str,
    format_type: str = "auto",
    model_type: str = "auto"
) -> str:
    """
    Extract assistant response from chat template with auto-detection.

    Args:
        text: Full text including chat template
        format_type: Format type or "auto" for auto-detection
        model_type: Model type hint for chat template extraction

    Returns:
        Extracted assistant response
    """
    handler = get_format_handler(format_type, text)
    return handler.extract_assistant_response(text, model_type)


__all__ = [
    # Classes
    "BaseFormatHandler",
    "XMLFormatHandler",
    "GPTOSSFormatHandler",
    # Functions
    "detect_format",
    "get_format_handler",
    "extract_thinking",
    "remove_thinking",
    "extract_final_response",
    "extract_tool_calls",
    "check_format",
    "extract_assistant_response",
]
