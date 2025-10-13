# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from pathlib import Path

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def extract_linguistic_context(
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor,
    data: DataProto,
    tokenizer=None,
    top_k: int = 5,
    context_window: int = 5,
    max_prompt_chars: int = 200,
    max_response_chars: int = 200,
) -> list[dict]:
    """
    Extract linguistic context for tokens with high divergence between rollout and actor.

    Args:
        rollout_log_probs: Log probabilities from rollout (batch_size, response_length)
        actor_log_probs: Log probabilities from actor (batch_size, response_length)
        response_mask: Mask for valid tokens (batch_size, response_length)
        responses: Response token IDs (batch_size, response_length)
        data: DataProto containing additional batch information
        tokenizer: Tokenizer for decoding (optional)
        top_k: Number of top divergent tokens to extract
        context_window: Number of tokens before/after to include as context
        max_prompt_chars: Maximum characters to log for prompt
        max_response_chars: Maximum characters to log for response

    Returns:
        List of dictionaries containing linguistic context for each high-divergence token
    """
    # Compute probability differences
    rollout_probs = torch.exp(rollout_log_probs)
    actor_probs = torch.exp(actor_log_probs)
    prob_diff = torch.abs(rollout_probs - actor_probs)

    # Mask out invalid positions
    prob_diff_masked = prob_diff * response_mask
    batch_size, seq_len = prob_diff_masked.shape

    # Find top-k divergent tokens
    prob_diff_flat = prob_diff_masked.view(-1)
    topk_values, topk_indices = torch.topk(prob_diff_flat, min(top_k, prob_diff_flat.numel()))

    divergence_logs = []

    for rank, (div_value, flat_idx) in enumerate(zip(topk_values, topk_indices)):
        # Convert flat index to (batch_idx, seq_idx)
        batch_idx = flat_idx.item() // seq_len
        seq_idx = flat_idx.item() % seq_len

        # Extract token information
        token_id = responses[batch_idx, seq_idx].item()
        rollout_prob = rollout_probs[batch_idx, seq_idx].item()
        actor_prob = actor_probs[batch_idx, seq_idx].item()
        rollout_logprob = rollout_log_probs[batch_idx, seq_idx].item()
        actor_logprob = actor_log_probs[batch_idx, seq_idx].item()

        log_entry = {
            "rank": rank,
            "batch_idx": batch_idx,
            "position": seq_idx,
            "position_from_end": seq_len - seq_idx - 1,
            "sequence_length": seq_len,
            "token_id": token_id,
            "prob_divergence": div_value.item(),
            "rollout_prob": rollout_prob,
            "actor_prob": actor_prob,
            "rollout_logprob": rollout_logprob,
            "actor_logprob": actor_logprob,
            "logprob_diff": abs(rollout_logprob - actor_logprob),
        }

        # Decode token and context if tokenizer available
        if tokenizer is not None:
            try:
                # Decode the divergent token
                token_text = tokenizer.decode([token_id])
                log_entry["token_text"] = repr(token_text)

                # Extract context window (±context_window tokens)
                context_start = max(0, seq_idx - context_window)
                context_end = min(seq_len, seq_idx + context_window + 1)
                context_ids = responses[batch_idx, context_start:context_end].tolist()
                context_text = tokenizer.decode(context_ids)
                log_entry["context_text"] = repr(context_text)
                log_entry["context_start_pos"] = context_start
                log_entry["context_end_pos"] = context_end

                # Decode full response (only valid tokens)
                response_ids = responses[batch_idx].tolist()
                response_mask_sample = response_mask[batch_idx].tolist()
                valid_response_ids = [
                    tid for tid, mask in zip(response_ids, response_mask_sample) if mask > 0
                ]
                full_response = tokenizer.decode(valid_response_ids)
                # Truncate if too long
                if len(full_response) > max_response_chars:
                    full_response = full_response[:max_response_chars] + "..."
                log_entry["full_response"] = repr(full_response)
                log_entry["response_num_tokens"] = len(valid_response_ids)

                # Extract prompt if available
                if "input_ids" in data.batch:
                    input_ids = data.batch["input_ids"][batch_idx]
                    # Prompt is everything before the response
                    prompt_length = len(input_ids) - len(responses[batch_idx])
                    if prompt_length > 0:
                        prompt_ids = input_ids[:prompt_length].tolist()
                        prompt_text = tokenizer.decode(prompt_ids)
                        # Truncate if too long
                        if len(prompt_text) > max_prompt_chars:
                            prompt_text = prompt_text[:max_prompt_chars] + "..."
                        log_entry["prompt_text"] = repr(prompt_text)
                        log_entry["prompt_num_tokens"] = prompt_length

            except Exception as e:
                logger.warning(f"Failed to decode token {token_id}: {e}")
                log_entry["token_text"] = f"<decode_error:{token_id}>"
                log_entry["decode_error"] = str(e)
        else:
            log_entry["token_text"] = f"<no_tokenizer:{token_id}>"

        divergence_logs.append(log_entry)

    return divergence_logs


def write_divergence_logs_to_jsonl(divergence_logs: list[dict], output_path: str, iteration: int = None):
    """
    Write divergence logs to a JSONL file.

    Args:
        divergence_logs: List of dictionaries containing divergence information
        output_path: Path to JSONL file (will be created if doesn't exist)
        iteration: Optional training iteration number to include in logs
    """
    if not divergence_logs:
        return

    # Create directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Append to JSONL file
    with open(output_file, "a") as f:
        for log_entry in divergence_logs:
            # Add iteration info if provided
            if iteration is not None:
                log_entry["iteration"] = iteration

            # Write as single-line JSON
            f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Wrote {len(divergence_logs)} divergence logs to {output_path}")


def log_divergence_human_readable(divergence_logs: list[dict]):
    """
    Log divergence information in human-readable format using Python logging.

    Args:
        divergence_logs: List of dictionaries containing divergence information
    """
    if not divergence_logs:
        return

    logger.info("=" * 80)
    logger.info(f"High Divergence Tokens (Top {len(divergence_logs)}):")
    logger.info("=" * 80)

    for log_entry in divergence_logs:
        logger.info(f"\n[Rank {log_entry['rank']}] Divergence: {log_entry['prob_divergence']:.4f}")
        logger.info(f"  Token: {log_entry.get('token_text', 'N/A')} (ID: {log_entry['token_id']})")
        logger.info(
            f"  Position: {log_entry['position']}/{log_entry['sequence_length']} "
            f"(from end: {log_entry['position_from_end']})"
        )
        logger.info(f"  Batch Index: {log_entry['batch_idx']}")
        logger.info(f"  Rollout: prob={log_entry['rollout_prob']:.6f}, logprob={log_entry['rollout_logprob']:.4f}")
        logger.info(f"  Actor:   prob={log_entry['actor_prob']:.6f}, logprob={log_entry['actor_logprob']:.4f}")
        logger.info(f"  LogProb Diff: {log_entry['logprob_diff']:.4f}")

        if "context_text" in log_entry:
            logger.info(f"  Context [{log_entry['context_start_pos']}:{log_entry['context_end_pos']}]: "
                       f"{log_entry['context_text']}")

        if "prompt_text" in log_entry:
            logger.info(f"  Prompt ({log_entry.get('prompt_num_tokens', '?')} tokens): "
                       f"{log_entry['prompt_text']}")

        if "full_response" in log_entry:
            logger.info(f"  Response ({log_entry.get('response_num_tokens', '?')} tokens): "
                       f"{log_entry['full_response']}")

        if "decode_error" in log_entry:
            logger.warning(f"  Decode Error: {log_entry['decode_error']}")

    logger.info("=" * 80)


def calculate_debug_metrics(
    data: DataProto,
    tokenizer=None,
    log_divergence: bool = True,
    divergence_threshold: float = 0.8,
    divergence_top_k: int = 5,
    divergence_jsonl_path: str = None,
    iteration: int = None,
) -> dict:
    """
    Calculate rollout vs actor logprobs diff and optionally log high-divergence tokens.

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
        tokenizer: optional tokenizer for decoding tokens in divergence logs
        log_divergence: whether to log detailed divergence information (default: True)
        divergence_threshold: minimum max divergence to trigger logging (default: 0.8)
        divergence_top_k: number of top divergent tokens to log (default: 5)
        divergence_jsonl_path: path to JSONL file for structured logs (optional)
        iteration: training iteration number for logging (optional)

    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor
            "training/high_divergence_tokens": (optional) list of detailed token information
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)

    max_divergence = torch.max(rollout_probs_diff).detach().item()

    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": max_divergence,
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }

    # Log detailed divergence information if requested and threshold exceeded
    if log_divergence and max_divergence >= divergence_threshold:
        divergence_logs = extract_linguistic_context(
            rollout_log_probs=rollout_old_log_probs,
            actor_log_probs=actor_old_log_probs,
            response_mask=response_mask,
            responses=responses,
            data=data,
            tokenizer=tokenizer,
            top_k=divergence_top_k,
            context_window=5,
            max_prompt_chars=200,
            max_response_chars=200,
        )

        if divergence_logs:
            # Log to Python logging (human-readable)
            log_divergence_human_readable(divergence_logs)

            # Write to JSONL file if path provided
            if divergence_jsonl_path is not None:
                write_divergence_logs_to_jsonl(divergence_logs, divergence_jsonl_path, iteration)

            # Include in metrics for external logging systems
            metrics["training/high_divergence_tokens"] = divergence_logs

    return metrics
