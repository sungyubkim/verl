# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
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

"""
Test for batch composition independence.

This test verifies that the same sample produces the same output regardless of
which other samples are in the same micro-batch.

Background:
- When micro_batch_size=1, log_prob computation gives consistent results
- When micro_batch_size=2 with shuffle=True, log_prob differs even for the same sample
- This test isolates the issue to find where batch composition affects results

Run with:
    # 1 GPU test
    torchrun --nproc_per_node=1 tests/models/test_batch_composition.py

    # TP=2 test
    torchrun --nproc_per_node=2 tests/models/test_batch_composition.py --tp 2
"""

import argparse
import os
import shutil
import tempfile

os.environ["NCCL_DEBUG"] = "WARN"

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, Qwen3Config


def create_test_model(tmp_dir: str) -> str:
    """Create a small test model for testing."""
    config = Qwen3Config(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    path = os.path.join(tmp_dir, "test_model")
    model.save_pretrained(path)
    config.save_pretrained(path)
    return path


def create_sample(
    seqlen: int,
    vocab_size: int,
    device: torch.device,
    seed: int = 42,
    response_ratio: float = 0.5,
    left_padding_ratio: float = 0.2,
):
    """Create a single sample with specified sequence length.

    Args:
        seqlen: Total sequence length (including padding)
        vocab_size: Vocabulary size
        device: Device to create tensors on
        seed: Random seed for reproducibility
        response_ratio: Ratio of sequence that is response
        left_padding_ratio: Ratio of sequence that is left padding
    """
    torch.manual_seed(seed)

    # Calculate lengths
    left_pad_len = int(seqlen * left_padding_ratio)
    valid_len = seqlen - left_pad_len
    response_length = int(seqlen * response_ratio)

    # Create input_ids with left padding (pad token = 0)
    input_ids = torch.zeros(seqlen, dtype=torch.long, device=device)
    input_ids[left_pad_len:] = torch.randint(1, vocab_size, (valid_len,), device=device)

    # Create attention mask (1 for valid, 0 for padding)
    attention_mask = torch.zeros(seqlen, dtype=torch.bool, device=device)
    attention_mask[left_pad_len:] = True

    # Create position ids
    position_ids = torch.zeros(seqlen, dtype=torch.long, device=device)
    position_ids[left_pad_len:] = torch.arange(valid_len, device=device)

    # Create responses (last response_length tokens)
    responses = input_ids[-response_length:].clone()

    # Create label and label_mask
    label = position_ids.clone()
    label[-response_length - 1:-1] = responses

    label_mask = attention_mask.clone()
    label_mask[:-response_length - 1] = False
    label_mask[-1] = False

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "label": label,
        "label_mask": label_mask,
        "responses": responses,
        "response_length": response_length,
    }


def batch_samples(*samples):
    """Stack multiple samples into a batch."""
    batch = {}
    for key in samples[0].keys():
        if key == "response_length":
            batch[key] = samples[0][key]
        else:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
    return batch


def pad_to_same_length(sample, target_seqlen, device):
    """Pad a sample to target sequence length (add more left padding)."""
    current_seqlen = sample["input_ids"].shape[0]
    if current_seqlen >= target_seqlen:
        return sample

    pad_len = target_seqlen - current_seqlen

    padded = {}
    for key, val in sample.items():
        if key == "response_length":
            padded[key] = val
        elif key in ["input_ids", "position_ids", "label"]:
            padded[key] = torch.cat([torch.zeros(pad_len, dtype=val.dtype, device=device), val])
        elif key in ["attention_mask", "label_mask"]:
            padded[key] = torch.cat([torch.zeros(pad_len, dtype=val.dtype, device=device), val])
        elif key == "responses":
            padded[key] = val  # responses don't need padding

    return padded


def broadcast_model_path(model_path: str, rank: int) -> str:
    """Broadcast model path from rank 0 to all other ranks."""
    if rank == 0:
        path_bytes = model_path.encode("utf-8")
        path_tensor = torch.tensor(
            list(path_bytes) + [0] * (512 - len(path_bytes)), dtype=torch.int64, device="cuda"
        )
    else:
        path_tensor = torch.zeros(512, dtype=torch.int64, device="cuda")

    dist.broadcast(path_tensor, src=0)

    path_list = path_tensor.tolist()
    model_path = bytes([b for b in path_list if b != 0]).decode("utf-8")
    return model_path


def test_batch_composition(tp_size: int = 1, vanilla_mbridge: bool = True):
    """Test that batch composition doesn't affect individual sample results.

    This test:
    1. Creates two samples A and B with different sequence lengths
    2. Runs forward pass with batch_size=1: [A] and [B] separately
    3. Runs forward pass with batch_size=2: [A, B] together
    4. Compares results - they should be identical for the same sample
    """
    import megatron.core.parallel_state as mpu

    from verl.models.mcore import get_mcore_forward_fn
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine import EngineRegistry
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create test model
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_batch_comp_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
        print(f"[Rank {rank}] Created test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=True,
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    checkpoint_config = CheckpointConfig()

    # Build engine
    engine = EngineRegistry.new(
        model_type="language_model",
        backend="megatron",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    engine.initialize()

    vocab_size = model_config.hf_config.vocab_size
    device = torch.device("cuda")

    # Create two samples with DIFFERENT sequence lengths
    # This is key - different lengths mean different padding patterns
    sample_A = create_sample(seqlen=64, vocab_size=vocab_size, device=device, seed=100)
    sample_B = create_sample(seqlen=64, vocab_size=vocab_size, device=device, seed=200, left_padding_ratio=0.4)

    # For batching, pad to same length
    max_seqlen = 64
    sample_A_padded = pad_to_same_length(sample_A, max_seqlen, device)
    sample_B_padded = pad_to_same_length(sample_B, max_seqlen, device)

    # Define logits processor
    def logits_processor(logits, label, label_mask):
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    # Get BSHD forward function
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)
    model = engine.module[0]

    # =========================================
    # Case 1: Sample A alone (batch_size=1)
    # =========================================
    batch_A = batch_samples(sample_A_padded)
    with torch.no_grad():
        output_A1 = bshd_forward(
            model=model,
            input_ids=batch_A["input_ids"],
            attention_mask=batch_A["attention_mask"],
            position_ids=batch_A["position_ids"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args={"label": batch_A["label"], "label_mask": batch_A["label_mask"]},
        )

    # =========================================
    # Case 2: Sample B alone (batch_size=1)
    # =========================================
    batch_B = batch_samples(sample_B_padded)
    with torch.no_grad():
        output_B1 = bshd_forward(
            model=model,
            input_ids=batch_B["input_ids"],
            attention_mask=batch_B["attention_mask"],
            position_ids=batch_B["position_ids"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args={"label": batch_B["label"], "label_mask": batch_B["label_mask"]},
        )

    # =========================================
    # Case 3: [A, B] together (batch_size=2)
    # =========================================
    batch_AB = batch_samples(sample_A_padded, sample_B_padded)
    with torch.no_grad():
        output_AB = bshd_forward(
            model=model,
            input_ids=batch_AB["input_ids"],
            attention_mask=batch_AB["attention_mask"],
            position_ids=batch_AB["position_ids"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args={"label": batch_AB["label"], "label_mask": batch_AB["label_mask"]},
        )

    # =========================================
    # Compare results
    # =========================================
    if mpu.is_pipeline_last_stage():
        log_prob_A1 = output_A1["log_probs"][0]  # [seqlen]
        log_prob_B1 = output_B1["log_probs"][0]  # [seqlen]
        log_prob_A2 = output_AB["log_probs"][0]  # First sample from batch
        log_prob_B2 = output_AB["log_probs"][1]  # Second sample from batch

        # Compare Sample A
        diff_A = (log_prob_A1 - log_prob_A2).abs()
        diff_A_max = diff_A.max().item()
        diff_A_mean = diff_A.mean().item()

        # Compare Sample B
        diff_B = (log_prob_B1 - log_prob_B2).abs()
        diff_B_max = diff_B.max().item()
        diff_B_mean = diff_B.mean().item()

        print(f"\n{'='*60}")
        print(f"[Rank {rank}] Batch Composition Test Results")
        print(f"{'='*60}")
        print(f"Sample A (batch=1 vs batch=2):")
        print(f"  - diff max: {diff_A_max:.6f}")
        print(f"  - diff mean: {diff_A_mean:.6f}")
        print(f"  - log_prob_A1 range: [{log_prob_A1.min():.4f}, {log_prob_A1.max():.4f}]")
        print(f"  - log_prob_A2 range: [{log_prob_A2.min():.4f}, {log_prob_A2.max():.4f}]")

        print(f"\nSample B (batch=1 vs batch=2):")
        print(f"  - diff max: {diff_B_max:.6f}")
        print(f"  - diff mean: {diff_B_mean:.6f}")
        print(f"  - log_prob_B1 range: [{log_prob_B1.min():.4f}, {log_prob_B1.max():.4f}]")
        print(f"  - log_prob_B2 range: [{log_prob_B2.min():.4f}, {log_prob_B2.max():.4f}]")
        print(f"{'='*60}")

        # Check for significant differences
        tolerance = 1e-5
        if diff_A_max > tolerance:
            print(f"\n[WARNING] Sample A differs significantly!")
            print(f"  Max diff location: {diff_A.argmax().item()}")
            max_idx = diff_A.argmax().item()
            print(f"  At max diff: A1={log_prob_A1[max_idx]:.6f}, A2={log_prob_A2[max_idx]:.6f}")

            # Check if it's in valid (non-padding) region
            valid_mask_A = batch_A["label_mask"][0]
            print(f"  Is max diff in valid region: {valid_mask_A[max_idx].item()}")

        if diff_B_max > tolerance:
            print(f"\n[WARNING] Sample B differs significantly!")
            print(f"  Max diff location: {diff_B.argmax().item()}")
            max_idx = diff_B.argmax().item()
            print(f"  At max diff: B1={log_prob_B1[max_idx]:.6f}, B2={log_prob_B2[max_idx]:.6f}")

        # Final verdict
        if diff_A_max < tolerance and diff_B_max < tolerance:
            print(f"\n✓ TEST PASSED: Batch composition does not affect results")
        else:
            print(f"\n✗ TEST FAILED: Batch composition affects results!")
            print(f"  This indicates samples interfere with each other in forward pass")

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] Test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--bridge", type=str, default="megatron", choices=["mbridge", "megatron"])
    args = parser.parse_args()

    vanilla_mbridge = args.bridge == "mbridge"
    test_batch_composition(tp_size=args.tp, vanilla_mbridge=vanilla_mbridge)
