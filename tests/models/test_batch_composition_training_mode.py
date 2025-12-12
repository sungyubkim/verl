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
Test for batch composition independence in training mode (forward_only=False).

This test verifies that the same sample produces the same log_prob regardless of
batch composition, simulating the actual production scenario:
- compute_log_prob: forward_only=True with batch composition X
- update_policy: forward_only=False with batch composition Y (shuffled)

Background:
- Phase 1 (bshd_forward direct): PASSED
- Phase 2 (forward_backward_func + forward_only=True): PASSED
- Phase 3 (this test): forward_only=False + multi micro-batch + shuffle simulation

Run with:
    # 1 GPU test - all tests
    torchrun --nproc_per_node=1 tests/models/test_batch_composition_training_mode.py

    # gradient influence only
    torchrun --nproc_per_node=1 tests/models/test_batch_composition_training_mode.py --test gradient

    # multi micro-batch test
    torchrun --nproc_per_node=1 tests/models/test_batch_composition_training_mode.py --test multi_mb

    # shuffle simulation (production scenario)
    torchrun --nproc_per_node=1 tests/models/test_batch_composition_training_mode.py --test shuffle
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
    """Create a single sample with specified sequence length."""
    torch.manual_seed(seed)

    left_pad_len = int(seqlen * left_padding_ratio)
    valid_len = seqlen - left_pad_len
    response_length = int(seqlen * response_ratio)

    input_ids = torch.zeros(seqlen, dtype=torch.long, device=device)
    input_ids[left_pad_len:] = torch.randint(1, vocab_size, (valid_len,), device=device)

    attention_mask = torch.zeros(seqlen, dtype=torch.bool, device=device)
    attention_mask[left_pad_len:] = True

    position_ids = torch.zeros(seqlen, dtype=torch.long, device=device)
    position_ids[left_pad_len:] = torch.arange(valid_len, device=device)

    responses = input_ids[-response_length:].clone()

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


def test_training_mode_batch_composition(
    tp_size: int = 1,
    pp_size: int = 1,
    vanilla_mbridge: bool = False,
    test_mode: str = "all",
):
    """Test batch composition independence in training mode.

    This test simulates the production scenario:
    - compute_log_prob: forward_only=True, batch composition [A,B], [C,D]
    - update_policy: forward_only=False, batch composition [A,C], [B,D] (shuffled)

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
        test_mode: "all", "gradient", "multi_mb", or "shuffle"
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

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create test model
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_training_mode_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
        print(f"[Rank {rank}] Created test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine - NOTE: forward_only=False for training mode
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=False,  # Training mode!
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=1,
    )
    optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    checkpoint_config = CheckpointConfig()

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

    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)
    forward_backward_func = get_forward_backward_func()
    model = engine.module

    def logits_processor(logits, label, label_mask):
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    # Production pattern: always return (scalar_tensor, extra_data)
    # This works with both forward_only=True and forward_only=False
    # Note: collect_non_loss_data=True + forward_only=False causes
    # "expected tensor, found dict" error in deallocate_output_tensor
    def loss_func(output):
        if isinstance(output, dict):
            device = output["log_probs"].device
        else:
            device = output.device if hasattr(output, 'device') else torch.device("cuda")
        dummy_loss = torch.tensor(1.0, device=device)
        return dummy_loss, output  # Always (scalar, dict)

    def run_forward(batch_dict, batch_size, forward_only):
        """Run forward pass using forward_backward_func.

        Production pattern (works with forward_only=True and False):
        - loss_func always returns (scalar, dict)
        - No collect_non_loss_data (causes error with forward_only=False)
        """
        # Production pattern: gradient cleanup before forward_only=False
        # Without this, multiple forward_backward_func calls with forward_only=False
        # will cause hang due to accumulated gradients
        if not forward_only:
            engine.optimizer_zero_grad()

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

        batch_for_gen = {k: v for k, v in batch_dict.items() if k != "response_length"}
        batch_generator = make_batch_generator([batch_for_gen], vpp_size=len(model))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=model,
            num_microbatches=1,
            seq_length=seqlen,
            micro_batch_size=batch_size,
            forward_only=forward_only,
            # No collect_non_loss_data - causes error with forward_only=False
        )

        # Synchronize after backward pass to ensure all async operations complete
        # Without this, subsequent forward passes may hang waiting for incomplete backward ops
        if not forward_only:
            torch.cuda.synchronize()
            # Use DP group barrier instead of world group all-reduce
            # World group collectives can conflict with Megatron's PP/TP groups
            dp_group = mpu.get_data_parallel_group()
            dist.barrier(group=dp_group)

        if mpu.is_pipeline_last_stage():
            # output is a list of (scalar_loss, extra_data) tuples from loss_func
            assert len(output) == 1, f"Expected 1 microbatch output, got {len(output)}"
            output_data = output[0]
            if isinstance(output_data, tuple) and len(output_data) >= 2:
                output_dict = output_data[1]  # Extract dict from second element
            else:
                output_dict = output_data
            return output_dict["log_probs"]
        return None

    results = {}

    # =========================================
    # Test 1: Gradient influence test
    # Compare forward_only=True vs forward_only=False with SAME batch composition
    # =========================================
    if test_mode in ["all", "gradient"]:
        print(f"\n[Rank {rank}] === Test 1: Gradient Influence ===")

        batch_AB = batch_samples(sample_A, sample_B)

        # forward_only=True (no gradients)
        with torch.no_grad():
            log_probs_AB_no_grad = run_forward(batch_AB, batch_size=2, forward_only=True)

        # forward_only=False (with gradients) - need to reset model state
        log_probs_AB_with_grad = run_forward(batch_AB, batch_size=2, forward_only=False)

        if mpu.is_pipeline_last_stage():
            diff = (log_probs_AB_no_grad - log_probs_AB_with_grad).abs()
            results["gradient_diff_max"] = diff.max().item()
            results["gradient_diff_mean"] = diff.mean().item()

            print(f"[Rank {rank}] Same batch [A,B]: forward_only=True vs False")
            print(f"  diff max: {diff.max():.6f}, mean: {diff.mean():.6f}")

            if diff.max() > 1e-5:
                print(f"  [WARNING] Gradient mode affects log_prob values!")
            else:
                print(f"  [OK] Gradient mode does not affect log_prob values")

    # =========================================
    # Test 2: Batch composition with forward_only=True (control)
    # =========================================
    if test_mode in ["all", "multi_mb"]:
        print(f"\n[Rank {rank}] === Test 2: Batch Composition (forward_only=True) ===")

        batch_AB = batch_samples(sample_A, sample_B)
        batch_AC = batch_samples(sample_A, sample_C)

        with torch.no_grad():
            log_probs_AB = run_forward(batch_AB, batch_size=2, forward_only=True)
            log_probs_AC = run_forward(batch_AC, batch_size=2, forward_only=True)

        if mpu.is_pipeline_last_stage():
            # Compare sample A in different batches
            diff_A = (log_probs_AB[0] - log_probs_AC[0]).abs()
            results["batch_comp_true_diff_max"] = diff_A.max().item()

            print(f"[Rank {rank}] Sample A: [A,B] vs [A,C] (forward_only=True)")
            print(f"  diff max: {diff_A.max():.6f}, mean: {diff_A.mean():.6f}")

            if diff_A.max() > 1e-5:
                print(f"  [WARNING] Batch composition affects log_prob (forward_only=True)!")
            else:
                print(f"  [OK] Batch composition does not affect log_prob (forward_only=True)")

    # =========================================
    # Test 3: Batch composition with forward_only=False
    # =========================================
    if test_mode in ["all", "multi_mb"]:
        print(f"\n[Rank {rank}] === Test 3: Batch Composition (forward_only=False) ===")

        batch_AB = batch_samples(sample_A, sample_B)
        batch_AC = batch_samples(sample_A, sample_C)

        # Need fresh forward passes
        log_probs_AB = run_forward(batch_AB, batch_size=2, forward_only=False)
        log_probs_AC = run_forward(batch_AC, batch_size=2, forward_only=False)

        if mpu.is_pipeline_last_stage():
            diff_A = (log_probs_AB[0] - log_probs_AC[0]).abs()
            results["batch_comp_false_diff_max"] = diff_A.max().item()

            print(f"[Rank {rank}] Sample A: [A,B] vs [A,C] (forward_only=False)")
            print(f"  diff max: {diff_A.max():.6f}, mean: {diff_A.mean():.6f}")

            if diff_A.max() > 1e-5:
                print(f"  [WARNING] Batch composition affects log_prob (forward_only=False)!")
            else:
                print(f"  [OK] Batch composition does not affect log_prob (forward_only=False)")

    # =========================================
    # Test 4: Production scenario simulation
    # compute_log_prob: forward_only=True, [A,B]
    # update_policy: forward_only=False, [A,C] (shuffled)
    # =========================================
    if test_mode in ["all", "shuffle"]:
        print(f"\n[Rank {rank}] === Test 4: Production Scenario Simulation ===")

        batch_AB = batch_samples(sample_A, sample_B)
        batch_AC = batch_samples(sample_A, sample_C)

        # Simulate compute_log_prob (forward_only=True)
        with torch.no_grad():
            old_log_probs_AB = run_forward(batch_AB, batch_size=2, forward_only=True)

        # Simulate update_policy with shuffled batch (forward_only=False)
        new_log_probs_AC = run_forward(batch_AC, batch_size=2, forward_only=False)

        if mpu.is_pipeline_last_stage():
            # Compare sample A: old (from [A,B]) vs new (from [A,C])
            diff_A = (old_log_probs_AB[0] - new_log_probs_AC[0]).abs()
            results["shuffle_sim_diff_max"] = diff_A.max().item()
            results["shuffle_sim_diff_mean"] = diff_A.mean().item()

            print(f"[Rank {rank}] Production scenario: compute_log_prob [A,B] vs update_policy [A,C]")
            print(f"  Sample A diff max: {diff_A.max():.6f}, mean: {diff_A.mean():.6f}")
            print(f"  old_log_probs[A] range: [{old_log_probs_AB[0].min():.4f}, {old_log_probs_AB[0].max():.4f}]")
            print(f"  new_log_probs[A] range: [{new_log_probs_AC[0].min():.4f}, {new_log_probs_AC[0].max():.4f}]")

            if diff_A.max() > 1e-5:
                print(f"\n  [ISSUE REPRODUCED] Production scenario shows diff > 0!")
                max_idx = diff_A.argmax().item()
                print(f"  Max diff at idx={max_idx}")
                print(f"  Values: old={old_log_probs_AB[0, max_idx]:.6f}, new={new_log_probs_AC[0, max_idx]:.6f}")
            else:
                print(f"\n  [OK] Production scenario shows no significant diff")

    # =========================================
    # Summary
    # =========================================
    if mpu.is_pipeline_last_stage():
        print(f"\n{'='*60}")
        print(f"[Rank {rank}] Summary of Results")
        print(f"{'='*60}")
        for key, value in results.items():
            status = "FAIL" if value > 1e-5 else "PASS"
            print(f"  {key}: {value:.6f} [{status}]")
        print(f"{'='*60}")

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
    parser.add_argument("--test", type=str, default="all", choices=["all", "gradient", "multi_mb", "shuffle"],
                        help="Test mode: all, gradient, multi_mb, or shuffle")
    args = parser.parse_args()

    vanilla_mbridge = args.bridge == "mbridge"
    test_training_mode_batch_composition(
        tp_size=args.tp,
        pp_size=args.pp,
        vanilla_mbridge=vanilla_mbridge,
        test_mode=args.test,
    )
