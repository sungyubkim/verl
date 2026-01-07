"""Mapping from RLVR-IFeval function names to IFEval instruction IDs.

This module provides mappings to convert RLVR-IFeval dataset format
to the IFEval format used by datatrove. All RLVR functions map to
existing IFEval instruction classes, so no new implementations are needed.
"""

from typing import Any, Dict, Optional

# RLVR function name → IFEval instruction ID mapping
# All 25 RLVR functions map to existing IFEval instructions
RLVR_TO_IFEVAL_MAP = {
    # Keywords constraints
    "verify_keywords": "keywords:existence",
    "verify_keyword_frequency": "keywords:frequency",
    "validate_forbidden_words": "keywords:forbidden_words",
    "verify_letter_frequency": "keywords:letter_frequency",
    # Language constraints
    "validate_response_language": "language:response_language",
    # Length constraints
    "verify_paragraph_count": "length_constraints:number_paragraphs",
    "validate_word_constraint": "length_constraints:number_words",
    "verify_sentence_constraint": "length_constraints:number_sentences",
    "validate_paragraphs": "length_constraints:nth_paragraph_first_word",
    # Detectable content
    "verify_postscript": "detectable_content:postscript",
    "validate_placeholders": "detectable_content:number_placeholders",
    # Detectable format
    "verify_bullet_points": "detectable_format:number_bullet_lists",
    "validate_title": "detectable_format:title",
    "validate_choice": "detectable_format:constrained_response",
    "validate_highlighted_sections": "detectable_format:number_highlighted_sections",
    "validate_sections": "detectable_format:multiple_sections",
    "validate_json_format": "detectable_format:json_format",
    # Combination
    "validate_repeat_prompt": "combination:repeat_prompt",
    "validate_two_responses": "combination:two_responses",
    # Case changes
    "validate_uppercase": "change_case:english_capital",
    "validate_lowercase": "change_case:english_lowercase",
    "validate_frequency_capital_words": "change_case:capital_word_frequency",
    # Start/end
    "validate_end": "startend:end_checker",
    "validate_quotation": "startend:quotation",
    # Punctuation
    "validate_no_commas": "punctuation:no_comma",
}

# RLVR parameter names → IFEval parameter names mapping
# Only functions with parameters are listed here
PARAM_NAME_MAP = {
    "verify_paragraph_count": {"N": "num_paragraphs"},
    "validate_word_constraint": {"N": "num_words", "quantifier": "relation"},
    "verify_sentence_constraint": {"N": "num_sentences", "quantifier": "relation"},
    "validate_paragraphs": {"N": "num_paragraphs", "i": "nth_paragraph", "first_word": "first_word"},
    "verify_postscript": {"postscript_marker": "postscript_marker"},
    "validate_placeholders": {"N": "num_placeholders"},
    "verify_bullet_points": {"N": "num_bullets"},
    "verify_keywords": {"keyword_list": "keywords"},
    "verify_keyword_frequency": {"word": "keyword", "N": "frequency"},
    "validate_forbidden_words": {"forbidden_words": "forbidden_words"},
    "verify_letter_frequency": {"letter": "letter", "N": "let_frequency"},
    "validate_response_language": {"language": "language"},
    "validate_highlighted_sections": {"N": "num_highlights"},
    # Note: IFEval has typo "section_spliter" (one 't'), not "section_splitter"
    "validate_sections": {"N": "num_sections", "section_splitter": "section_spliter"},
    "validate_repeat_prompt": {"original_prompt": "prompt_to_repeat"},
    "validate_frequency_capital_words": {"N": "capital_frequency", "quantifier": "capital_relation"},
    "validate_end": {"end_phrase": "end_phrase"},
    # Note: validate_choice has no parameter mapping because IFEval's ConstrainedResponseChecker
    # uses hardcoded options and doesn't support custom options from RLVR
    # Functions without parameters (return None):
    # - validate_title
    # - validate_json_format
    # - validate_two_responses
    # - validate_uppercase
    # - validate_lowercase
    # - validate_quotation
    # - validate_no_commas
    # - validate_choice (uses hardcoded options in IFEval)
}


def map_param_names(func_name: str, rlvr_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map RLVR parameter names to IFEval parameter names.

    Args:
        func_name: RLVR function name
        rlvr_params: Dictionary of RLVR parameters (may contain null values)

    Returns:
        Dictionary of IFEval parameters with null values filtered out,
        or None if no parameters are needed for this function.

    Examples:
        >>> map_param_names("verify_paragraph_count", {"N": 5})
        {"num_paragraphs": 5}

        >>> map_param_names("validate_word_constraint", {"N": 100, "quantifier": "at least"})
        {"num_words": 100, "relation": "at least"}

        >>> map_param_names("validate_lowercase", {"N": None})
        None
    """
    # If function has no parameter mapping, return None
    if func_name not in PARAM_NAME_MAP:
        return None

    param_map = PARAM_NAME_MAP[func_name]
    ifeval_params = {}

    # Map parameter names and filter out None values
    for rlvr_key, ifeval_key in param_map.items():
        if rlvr_key in rlvr_params and rlvr_params[rlvr_key] is not None:
            value = rlvr_params[rlvr_key]

            # Normalize quantifier/relation values
            if ifeval_key in ("relation", "capital_relation"):
                quantifier = value
                # For "at most N" we need to convert to "less than N+1"
                if quantifier == "at most":
                    value = "less than"
                    # Adjust the numeric value
                    num_key = "N"
                    if num_key in rlvr_params and rlvr_params[num_key] is not None:
                        # Find the corresponding IFEval numeric key
                        for rk, ik in param_map.items():
                            if rk == num_key:
                                ifeval_params[ik] = rlvr_params[num_key] + 1
                                break
                elif quantifier == "around":
                    # "around X" is approximated as "at least X" (lenient)
                    value = "at least"
                elif quantifier == "at least":
                    value = "at least"
                # else: keep original value

            ifeval_params[ifeval_key] = value

    # Return None if no parameters after filtering
    return ifeval_params if ifeval_params else None


def get_ifeval_instruction_id(func_name: str) -> str:
    """Get IFEval instruction ID for a given RLVR function name.

    Args:
        func_name: RLVR function name

    Returns:
        IFEval instruction ID

    Raises:
        KeyError: If function name is not recognized

    Example:
        >>> get_ifeval_instruction_id("validate_lowercase")
        "change_case:english_lowercase"
    """
    if func_name not in RLVR_TO_IFEVAL_MAP:
        raise KeyError(f"Unknown RLVR function: {func_name}")
    return RLVR_TO_IFEVAL_MAP[func_name]
