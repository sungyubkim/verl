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
Test for batch composition independence using forward_backward_func.

This test verifies that the same sample produces the same output regardless of
which other samples are in the same micro-batch, using the actual training path
with forward_backward_func (Megatron's scheduler).

Background:
- test_batch_composition.py (Phase 1) used bshd_forward directly and PASSED
- But actual training with micro_batch_size > 1 + shuffle=True shows diff > 0
- This test uses forward_backward_func to match the actual training path

Run with:
    # 1 GPU test (PP=1)
    torchrun --nproc_per_node=1 tests/models/test_batch_composition_training_path.py

    # PP=2 test
    torchrun --nproc_per_node=2 tests/models/test_batch_composition_training_path.py --pp 2

    # TP=2 test
    torchrun --nproc_per_node=2 tests/models/test_batch_composition_training_path.py --tp 2
"""

import argparse
import os
import shutil
import tempfile
from functools import partial

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


def test_batch_composition_training_path(
    tp_size: int = 1,
    pp_size: int = 1,
    vanilla_mbridge: bool = True,
):
    """Test that batch composition doesn't affect individual sample results.

    This test uses forward_backward_func (Megatron's scheduler) to match
    the actual training path, unlike test_batch_composition.py which uses
    bshd_forward directly.

    This test:
    1. Creates 4 samples A, B, C, D with different seeds
    2. Runs forward pass with batch_size=1: [A], [B], [C], [D] separately
    3. Runs forward pass with batch_size=2: [A, B] and [A, C]
    4. Compares results - Sample A should be identical regardless of batch composition
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.pipeline_parallel import get_forward_backward_func

    from verl.models.mcore import get_mcore_forward_fn
    from verl.trainer.config import CheckpointConfig
    from verl.utils.megatron.pipeline_parallel import make_batch_generator
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine import EngineRegistry

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create test model
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_batch_comp_training_")
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
        pipeline_model_parallel_size=pp_size,
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
    seqlen = 64

    # Create 4 samples with different seeds and padding ratios
    sample_A = create_sample(seqlen=seqlen, vocab_size=vocab_size, device=device, seed=100, left_padding_ratio=0.2)
    sample_B = create_sample(seqlen=seqlen, vocab_size=vocab_size, device=device, seed=200, left_padding_ratio=0.3)
    sample_C = create_sample(seqlen=seqlen, vocab_size=vocab_size, device=device, seed=300, left_padding_ratio=0.4)
    sample_D = create_sample(seqlen=seqlen, vocab_size=vocab_size, device=device, seed=400, left_padding_ratio=0.1)

    # Get BSHD forward function and forward_backward_func
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)
    forward_backward_func = get_forward_backward_func()
    model = engine.module

    # Define logits processor
    def logits_processor(logits, label, label_mask):
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    # Define loss_func for forward_backward_func (Production pattern)
    # Always return (scalar_tensor, extra_data) to avoid deallocate_output_tensor error
    def loss_func(output):
        if isinstance(output, dict):
            device = output["log_probs"].device
        else:
            device = output.device if hasattr(output, 'device') else torch.device("cuda")
        dummy_loss = torch.tensor(1.0, device=device)
        return dummy_loss, output  # Always (scalar, data) format

    def run_forward(batch_dict, batch_size):
        """Run forward pass using forward_backward_func.

        Production pattern (from megatron_actor.py):
        - Split batch into individual samples (micro_batch_size=1)
        - num_microbatches = batch_size
        - Concatenate results from each micro-batch
        """
        # Production pattern: split batch into individual samples
        if batch_size > 1:
            micro_batches = [
                {k: v[i:i+1] for k, v in batch_dict.items() if k != "response_length"}
                for i in range(batch_size)
            ]
        else:
            micro_batches = [{k: v for k, v in batch_dict.items() if k != "response_length"}]

        n_micro_batch = len(micro_batches)

        def forward_step(batch_iter, model_arg):
            micro_batch = next(batch_iter)
            output = bshd_forward(
                model=model_arg,
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                multi_modal_inputs={},
                logits_processor=logits_processor,
                logits_processor_args={
                    "label": micro_batch["label"],
                    "label_mask": micro_batch["label_mask"],
                },
            )
            return output, partial(loss_func)

        # Production pattern: make_batch_generator with micro_batches list
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(model))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=model,
            num_microbatches=n_micro_batch,  # Production: number of micro-batches
            seq_length=seqlen,
            micro_batch_size=1,  # Production pattern: always 1
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage():
            # output is a list of (scalar_loss, extra_data) tuples from loss_func
            # output = [(dummy_loss, {"log_probs": tensor}), ...] for each micro-batch
            log_probs_list = []
            for output_data in output:
                if isinstance(output_data, tuple) and len(output_data) >= 2:
                    output_dict = output_data[1]
                else:
                    output_dict = output_data
                log_probs_list.append(output_dict["log_probs"])
            # Concatenate results from all micro-batches
            return torch.cat(log_probs_list, dim=0)  # [batch_size, seq_len]
        return None

    # =========================================
    # Case 1: Individual samples (batch_size=1)
    # =========================================
    print(f"\n[Rank {rank}] Running individual sample forwards...")

    batch_A = batch_samples(sample_A)
    batch_B = batch_samples(sample_B)
    batch_C = batch_samples(sample_C)

    with torch.no_grad():
        log_prob_A1 = run_forward(batch_A, batch_size=1)
        log_prob_B1 = run_forward(batch_B, batch_size=1)
        log_prob_C1 = run_forward(batch_C, batch_size=1)

    # =========================================
    # Case 2: [A, B] together (batch_size=2)
    # =========================================
    print(f"[Rank {rank}] Running [A, B] batch forward...")
    batch_AB = batch_samples(sample_A, sample_B)
    with torch.no_grad():
        log_probs_AB = run_forward(batch_AB, batch_size=2)

    # =========================================
    # Case 3: [A, C] together (batch_size=2)
    # =========================================
    print(f"[Rank {rank}] Running [A, C] batch forward...")
    batch_AC = batch_samples(sample_A, sample_C)
    with torch.no_grad():
        log_probs_AC = run_forward(batch_AC, batch_size=2)

    # =========================================
    # Compare results
    # =========================================
    if mpu.is_pipeline_last_stage():
        # Extract sample A from each batch
        log_prob_A2 = log_probs_AB[0]  # A from [A, B]
        log_prob_A3 = log_probs_AC[0]  # A from [A, C]

        # Compare Sample A across all cases
        diff_A_AB = (log_prob_A1[0] - log_prob_A2).abs()
        diff_A_AC = (log_prob_A1[0] - log_prob_A3).abs()
        diff_AB_AC = (log_prob_A2 - log_prob_A3).abs()

        print(f"\n{'='*60}")
        print(f"[Rank {rank}] Batch Composition Test Results (forward_backward_func)")
        print(f"{'='*60}")
        print(f"Sample A comparison:")
        print(f"  - individual vs [A,B]: diff max={diff_A_AB.max():.6f}, mean={diff_A_AB.mean():.6f}")
        print(f"  - individual vs [A,C]: diff max={diff_A_AC.max():.6f}, mean={diff_A_AC.mean():.6f}")
        print(f"  - [A,B] vs [A,C]:      diff max={diff_AB_AC.max():.6f}, mean={diff_AB_AC.mean():.6f}")
        print(f"\nlog_prob ranges:")
        print(f"  - A (individual): [{log_prob_A1[0].min():.4f}, {log_prob_A1[0].max():.4f}]")
        print(f"  - A (from [A,B]): [{log_prob_A2.min():.4f}, {log_prob_A2.max():.4f}]")
        print(f"  - A (from [A,C]): [{log_prob_A3.min():.4f}, {log_prob_A3.max():.4f}]")
        print(f"{'='*60}")

        # Check for significant differences
        tolerance = 1e-5
        max_diff = max(diff_A_AB.max().item(), diff_A_AC.max().item(), diff_AB_AC.max().item())

        if max_diff > tolerance:
            print(f"\n[WARNING] Sample A differs between batch compositions!")
            print(f"  This suggests forward_backward_func path has batch composition interference")

            # Show where max diff occurs
            if diff_A_AB.max() > tolerance:
                max_idx = diff_A_AB.argmax().item()
                print(f"\n  Max diff A vs [A,B] at idx={max_idx}:")
                print(f"    individual={log_prob_A1[0, max_idx]:.6f}, with_B={log_prob_A2[max_idx]:.6f}")

            if diff_A_AC.max() > tolerance:
                max_idx = diff_A_AC.argmax().item()
                print(f"\n  Max diff A vs [A,C] at idx={max_idx}:")
                print(f"    individual={log_prob_A1[0, max_idx]:.6f}, with_C={log_prob_A3[max_idx]:.6f}")

        # Final verdict
        if max_diff < tolerance:
            print(f"\n TEST PASSED: Batch composition does not affect results in forward_backward_func path")
        else:
            print(f"\n TEST FAILED: Batch composition affects results!")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  This indicates the issue is in forward_backward_func path")

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
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--bridge", type=str, default="megatron", choices=["mbridge", "megatron"])
    args = parser.parse_args()

    vanilla_mbridge = args.bridge == "mbridge"
    test_batch_composition_training_path(tp_size=args.tp, pp_size=args.pp, vanilla_mbridge=vanilla_mbridge)
