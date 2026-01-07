from typing import TypedDict


class NormalizedScore(TypedDict, total=False):
    """Standard schema for all scorer outputs.

    All scorers must return a dict conforming to this schema to ensure
    consistent Parquet writes and avoid schema mismatch errors.

    Attributes:
        score: The primary score value (required, always present)
        error: Error message if scoring failed, empty string "" on success (required, always present)
               Empty string instead of None for Parquet compatibility
        reward_think: Reward for thinking/reasoning content. 0.0 if not applicable/not computed.
        reward_fmt: Reward for format correctness. 0.0 if not applicable/not computed.
        acc: Accuracy score for pass@k computation. Binary (0/1) derived from score or correctness.
        reward_length: Reward for response length. 0.0 if not applicable/not computed.

    Note: All reward_* fields use 0.0 instead of None for Parquet compatibility.
          PyArrow requires consistent types - using 0.0 ensures the fields are always float type.
    """
    score: float
    error: str
    reward_think: float
    reward_fmt: float
    acc: float
    reward_length: float


def normalize_score(raw_result, error=None) -> NormalizedScore:
    """Normalize any scorer output to standard schema.

    Ensures all required fields are present with consistent types to prevent
    Parquet schema mismatch errors when writing documents.

    Args:
        raw_result: Raw scorer output (dict, tuple, or scalar)
        error: Optional error message to include in normalized result

    Returns:
        NormalizedScore with all fields properly typed and present
    """
    import json

    # Handle different input types
    if isinstance(raw_result, dict):
        base = raw_result.copy()
    elif isinstance(raw_result, tuple):
        # Sandbox fusion returns (score, metadata_list)
        score_value = float(raw_result[0])
        metadata_list = raw_result[1] if len(raw_result) > 1 else []

        base = {"score": score_value}

        # Extract error messages from metadata_list
        if isinstance(metadata_list, list) and metadata_list:
            errors = []
            for item in metadata_list:
                if isinstance(item, dict):
                    if "error" in item and item["error"]:
                        errors.append(str(item["error"]))

            # Combine all errors into error field
            if errors:
                base["error"] = "; ".join(errors)
    elif isinstance(raw_result, (int, float, bool)):
        base = {"score": float(raw_result)}
    else:
        base = {"score": 0.0}
        error = error or f"Invalid scorer result type: {type(raw_result)}"

    # Define standard fields that will be extracted
    standard_fields = {"score", "error", "reward_think", "reward_fmt", "acc", "reward_length"}

    # Check if there are extra fields (dataset-specific debugging info)
    extra_fields = {}
    if isinstance(raw_result, dict):
        for key, value in raw_result.items():
            if key not in standard_fields:
                # Serialize extra fields for preservation
                extra_fields[key] = value

    # Ensure all required fields are present with proper types
    # For Parquet compatibility:
    # - Use empty string "" instead of None for string fields (error)
    # - Use 0.0 instead of None for float fields (reward_*)
    # This prevents PyArrow schema inference issues where None -> null type conflicts with target type
    error_value = error or base.get("error")

    # If there are extra fields to preserve, serialize them and add to error field
    if extra_fields:
        try:
            extra_json = json.dumps(extra_fields, sort_keys=True)
            debug_info = f"[Debug] {extra_json}"

            # Append to existing error message or use as standalone debug info
            if error_value:
                error_value = f"{error_value} {debug_info}"
            else:
                error_value = debug_info
        except Exception as e:
            # If serialization fails, just note that extra fields existed
            if error_value:
                error_value = f"{error_value} [Debug] Failed to serialize extra fields: {e}"
            else:
                error_value = f"[Debug] Failed to serialize extra fields: {e}"

    normalized: NormalizedScore = {
        "score": float(base.get("score", 0.0)),
        "error": error_value if error_value is not None else "",  # "" instead of None
        "reward_think": float(base.get("reward_think") or 0.0),  # 0.0 instead of None
        "reward_fmt": float(base.get("reward_fmt") or 0.0),      # 0.0 instead of None
        "acc": float(base.get("acc") or 0.0),  # 0.0 instead of None
        "reward_length": float(base.get("reward_length") or 0.0),    # 0.0 instead of None
    }

    return normalized


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        sandbox_fusion_url (str, optional): URL for sandbox fusion server (required for code/verilog execution).
        concurrent_semaphore: Semaphore for limiting concurrent requests.
        memory_limit_mb (int, optional): Memory limit for code execution in MB.

    Returns:
        NormalizedScore: A dictionary with standardized fields (score, error, reward_*).
                        All fields are guaranteed to be present with consistent types.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source in [
        "openai/gsm8k",
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "HuggingFaceH4/MATH-500",
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
        "math_dapo",
        "math",
        "math_dapo_reasoning",
        # Additional math datasets
        "Big-Math-RL-Verified",
        "DAPO-Math-17K",
        "DeepScaleR-Preview",
        "MathX-5M",
        "OpenR1-Math-220k",
        "orz-math-72k",
        "train-math-deepscaler",
        "train-math-numinamath1.5_amc_aime",
        "train-math-numinamath1.5_aops_forum",
        "train-math-numinamath1.5_cn_contest",
        "train-math-numinamath1.5_olympiads",
        "train-math-numinamath1.5_olympiads_ref",
        "train-math-still3",
    ]:
        from . import math

        res = math.compute_score(solution_str, ground_truth, **kwargs)
    elif data_source in [
        "rlla",
        "toolrl",
        "tool_learning",
        "toolace",
        "hammer",
        "xlam",
        "sungyub/toolrl-verl",
        "rlla_gpt",  # GPT-OSS format ToolRL
    ]:
        from . import toolrl

        res = toolrl.compute_score(
            solution_str,
            ground_truth,
            step=kwargs.get("step", 0),
            model_type=kwargs.get("model_type", "auto"),
            enable_length_reward=kwargs.get("enable_length_reward", False),
            format_type=kwargs.get("format_type", "auto"),
            **{k: v for k, v in kwargs.items() if k not in ["step", "model_type", "enable_length_reward", "format_type"]}
        )
    elif data_source in [
        "codecontests",
        "apps",
        "codeforces",
        "taco",
        # Additional code execution datasets
        "code-contests-plus",
        # KodCode datasets (sungyub/kodcode-v1-verl) - 12 variants
        "kodcode-algorithm",
        "kodcode-apps",
        "kodcode-code_contests",
        "kodcode-codeforces",
        "kodcode-data_structure",
        "kodcode-docs",
        "kodcode-evol",
        "kodcode-filter",
        "kodcode-leetcode",
        "kodcode-package",
        "kodcode-prefill",
        "kodcode-taco",
        # AceCode datasets (sungyub/acecode-87k-verl)
        "bigcode_python_fns",  # AceCode: BigCode Python functions
        "evol",  # AceCode: Evol-Instruct style
        "oss",  # AceCode: Open source software
        "rstar-coder",
        "train-code-leetcode-Easy",
        "train-code-leetcode-Medium",
        "train-code-leetcode-Hard",
        "test-code-leetcode-Easy",
        "test-code-leetcode-Medium",
        "test-code-leetcode-Hard",
        "train-code-taco-easy",
        "train-code-taco-medium",
        "train-code-taco-hard",
        "train-code-taco-medium_hard",
        "train-code-taco-very_hard",
        "train-code-taco-unknown_difficulty",
    ]:
        # Code execution scoring requires external sandbox service
        if sandbox_fusion_url is None:
            raise ValueError(
                f"Code execution scoring for {data_source} requires a sandbox_fusion_url. "
                "Please set SANDBOX_FUSION_URL in your configuration to point to a running "
                "sandbox fusion server (e.g., 'http://your-sandbox-server.com:5000'). "
                "See VERL documentation for sandbox setup: "
                "https://github.com/volcengine/verl/tree/main/verl/utils/reward_score/sandbox_fusion"
            )

        from . import sandbox_fusion

        # Pass the URL directly, ground_truth likely contains test cases here
        res = sandbox_fusion.compute_score(
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            solution_str,
            ground_truth,
            continuous=True,
        )
    elif data_source in ["allenai/IF_multi_constraints_upto5", "ifeval", "sungyub/ifbench-verl", "sungyub/ifeval-rlvr-verl"]:
        # Instruction Following evaluation
        from . import ifeval

        res = ifeval.compute_score(
            solution_str,
            ground_truth,
            **kwargs
        )
    elif data_source in ["codev", "sungyub/codev-r1-verl"]:
        # CodeV Verilog code generation with equivalence checking
        if sandbox_fusion_url is None:
            raise ValueError(
                f"CodeV scoring for {data_source} requires a sandbox_fusion_url for Verilog simulation. "
                "Please set SANDBOX_FUSION_URL in your configuration to point to a running "
                "sandbox fusion server (e.g., 'http://localhost:8080/run_code'). "
                "See documentation: https://github.com/bytedance/SandboxFusion"
            )

        from . import codev

        res = codev.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            **kwargs
        )
    elif data_source in ['hitab', 'multihier', 'finqa']:
        # Table reasoning with boxed answer format (Guru datasets)
        from . import table_boxed

        res = table_boxed.compute_score(
            model_output=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['WTQ', 'HiTab']:
        # Table QA: WikiTableQuestions, HiTab (JSON list answers)
        from . import tqa

        res = tqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['TabFact']:
        # Table Fact Verification (binary: entailed/refuted)
        from . import tfv

        res = tfv.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['FeTaQA']:
        # Free-form Table QA with BLEU/ROUGE scoring
        from . import ff_tqa

        res = ff_tqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "long_toc_choices" in data_source:
        # Long-context multiple choice QA (A-D)
        from . import long

        res = long.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "docmath" in data_source:
        # Document math problems with numeric answers
        from . import docmath

        res = docmath.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "multihoprag" in data_source or "musique" in data_source:
        # Document QA with free text answers (EM/F1 scoring)
        from . import docqa

        res = docqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif (
        data_source in [
            "ordering_puzzle",
            "zebra_puzzle",
            "graph_logical",
            "arcagi1",
            "arcagi2",
            "barc",
        ]
        or "puzzle" in data_source
        or "arcagi" in data_source
        or "barc" in data_source
    ):
        # Logic domain scoring: ordering puzzles, zebra puzzles, graph problems, ARC-AGI
        from . import logic

        res = logic.compute_score(
            model_output=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )

    # Normalize the scorer result to ensure consistent schema
    return normalize_score(res)


def compute_score_safe(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
) -> NormalizedScore:
    """Safe wrapper around compute_score that catches exceptions.

    This function ensures that even if scoring fails, a normalized result
    with an error message is returned instead of raising an exception.

    Args:
        Same as compute_score()

    Returns:
        NormalizedScore: Always returns a normalized score dict, even on error
    """
    try:
        return compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
            **kwargs,
        )
    except Exception as e:
        # Return normalized error result
        return normalize_score(
            raw_result={"score": 0.0},
            error=f"{type(e).__name__}: {str(e)}"
        )


__all__ = ["compute_score", "compute_score_safe", "normalize_score", "NormalizedScore"]


# Backward compatibility alias
default_compute_score = compute_score
