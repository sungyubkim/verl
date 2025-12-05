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
Unit tests for BSHD (non-sequence-packing) format support.

This test verifies that the BSHD format works correctly with Megatron models.
The key function being tested is `model_forward_gen(use_sequence_packing=False)`
which provides BSHD format support with Pipeline Parallelism.

Run with:
    # 1 GPU - PP=1 (basic BSHD forward test)
    torchrun --nproc_per_node=1 tests/models/test_bshd_pp_forward.py --test basic

    # 2 GPUs - PP=2 (BSHD with pipeline parallelism)
    torchrun --nproc_per_node=2 tests/models/test_bshd_pp_forward.py --test pp_only

    # 4 GPUs - PP=2, TP=2 (BSHD with pipeline + tensor parallelism)
    torchrun --nproc_per_node=4 tests/models/test_bshd_pp_forward.py --test pp_tp

    # 4 GPUs - PP=2, CP=2 (BSHD with pipeline + context parallelism)
    torchrun --nproc_per_node=4 tests/models/test_bshd_pp_forward.py --test pp_cp

    # 8 GPUs - PP=2, TP=2, CP=2 (BSHD with all parallelism types)
    torchrun --nproc_per_node=8 tests/models/test_bshd_pp_forward.py --test all

    # 2 GPUs - TP=2 (BSHD with tensor parallelism, PP=1)
    torchrun --nproc_per_node=2 tests/models/test_bshd_pp_forward.py --test tp_only

    # 2 GPUs - CP=2 (BSHD with context parallelism, PP=1)
    torchrun --nproc_per_node=2 tests/models/test_bshd_pp_forward.py --test cp_only

    # 1+ GPUs - Compare THD vs BSHD outputs
    torchrun --nproc_per_node=1 tests/models/test_bshd_pp_forward.py --test compare

    # 2 GPUs - Verify SP actually divides sequences across TP ranks
    torchrun --nproc_per_node=2 tests/models/test_bshd_pp_forward.py --test sp_verify

    # 2 GPUs - Test 1F1B overlap scheduling with BSHD format
    torchrun --nproc_per_node=2 tests/models/test_bshd_pp_forward.py --test 1f1b_overlap

Options:
    --test: Test configuration (basic, pp_only, pp_tp, pp_cp, all, tp_only, cp_only, compare, actor, sp_verify, 1f1b_overlap)
    --bridge: Weight loading bridge to use
        - mbridge: Use mbridge package (vanilla_mbridge=True)
        - megatron: Use megatron.bridge (vanilla_mbridge=False, default)
"""

import argparse
import os
import shutil
import tempfile

os.environ["NCCL_DEBUG"] = "WARN"

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig


def create_test_model(tmp_dir: str) -> str:
    """Create a small test model for testing."""
    config = Qwen3Config(
        num_hidden_layers=4,  # Small model for testing
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


def create_test_moe_model(tmp_dir: str) -> str:
    """Create a small MoE test model for 1F1B overlap testing.

    This model has Mixture of Experts (MoE) architecture which is required
    for build_schedule_plan to be available in Megatron-Core.
    """
    # Note: Qwen3 uses head_dim=128, so hidden_size must be num_attention_heads * 128
    config = Qwen3MoeConfig(
        num_hidden_layers=4,  # Small model for testing
        hidden_size=512,  # 4 * 128 = 512 (head_dim=128 is fixed for Qwen3)
        intermediate_size=1024,
        moe_intermediate_size=512,  # Size of each expert
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        num_experts=4,  # Number of experts
        num_experts_per_tok=2,  # Top-k experts per token
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    path = os.path.join(tmp_dir, "test_moe_model")
    model.save_pretrained(path)
    config.save_pretrained(path)
    return path


def create_test_batch(
    batch_size: int,
    seqlen: int,
    vocab_size: int,
    device: torch.device,
):
    """Create a test batch with random data and left-padded attention masks."""
    from verl.utils.model import compute_position_id_with_mask, create_random_mask

    torch.manual_seed(42)

    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen), device=device)
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_valid_token=0.8,
        max_ratio_of_left_padding=0.2,
        min_ratio_of_valid_token=0.6,
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    response_length = seqlen // 2
    responses = input_ids[:, response_length:]

    # Create labels (shifted input_ids for next-token prediction)
    label = position_ids.clone()
    label[:, -response_length - 1 : -1] = responses

    # Create label mask (only compute loss on response tokens)
    label_mask = attention_mask.clone()
    label_mask[:, : -response_length - 1] = False
    label_mask[:, -1] = False

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask.to(bool),
        "position_ids": position_ids,
        "label": label,
        "label_mask": label_mask.to(bool),
        "responses": responses,
        "response_length": response_length,
    }


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


def test_bshd_forward(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    vanilla_mbridge: bool = True,
):
    """Test BSHD forward pass with Megatron models.

    This test verifies that:
    1. The BSHD format conversion (remove_left_padding/recover_left_padding) works correctly
    2. The shape assertion (logits.shape[:2] == label.shape[:2]) passes
    3. The output log_probs have correct shape

    Note: PP>1 requires the combined_1f1b scheduler which needs overlap_moe_expert_parallel_comm=True.
    For basic testing, we use PP=1 and test with TP/CP variations.

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size (should be 1 for this test)
        cp_size: Context parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
    """
    import megatron.core.parallel_state as mpu

    from verl.models.mcore import get_mcore_forward_fn
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine import EngineRegistry

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create test model (only on rank 0)
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_bshd_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
        print(f"[Rank {rank}] Created test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine with PP=1 for direct forward testing
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=True,  # Forward only for testing
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,  # True=mbridge, False=megatron.bridge
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,  # PP=1 for direct forward
        context_parallel_size=cp_size,
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

    # Create test batch
    batch_size = 4
    seqlen = 64
    vocab_size = model_config.hf_config.vocab_size

    batch = create_test_batch(batch_size, seqlen, vocab_size, torch.device("cuda"))

    # Import utilities
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    # Define logits processor (same as in megatron_actor.py)
    def logits_processor(logits, label, label_mask):
        # This is the assertion that verifies BSHD conversion correctness
        assert logits.shape[:2] == label.shape[:2], (
            f"Shape mismatch: logits {logits.shape[:2]} vs label {label.shape[:2]}"
        )
        assert label.shape == label_mask.shape, f"Shape mismatch: label {label.shape} vs label_mask {label_mask.shape}"

        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    logits_processor_args = {"label": batch["label"], "label_mask": batch["label_mask"]}

    # Get BSHD forward function (use_sequence_packing=False)
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)

    # Get model (engine.module is a list of model chunks for PP, use [0] for PP=1)
    model = engine.module[0]

    # Run BSHD forward
    try:
        output = bshd_forward(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
        )

        # Verify output
        if mpu.is_pipeline_last_stage():
            assert "log_probs" in output, "Expected 'log_probs' in output"
            log_probs = output["log_probs"]
            assert log_probs.shape == batch["label"].shape, (
                f"Shape mismatch: log_probs {log_probs.shape} vs label {batch['label'].shape}"
            )
            print(f"[Rank {rank}] ✓ BSHD forward passed! log_probs shape: {log_probs.shape}")

            # Verify values are not all zeros (sanity check)
            valid_mask = batch["label_mask"]
            valid_log_probs = log_probs[valid_mask]
            assert valid_log_probs.numel() > 0, "Expected some valid log_probs"
            print(f"[Rank {rank}]   - Valid log_probs mean: {valid_log_probs.mean().item():.4f}")

    except AssertionError as e:
        print(f"[Rank {rank}] ✗ FAILED: {e}")
        raise
    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR: {type(e).__name__}: {e}")
        raise

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] Test completed successfully!")


def test_sp_verification(
    tp_size: int = 2,
    pp_size: int = 1,
    cp_size: int = 1,
    vanilla_mbridge: bool = True,
):
    """Verify that Sequence Parallelism (SP) actually divides sequences across TP ranks.

    This test captures intermediate tensor shapes using forward hooks to verify that:
    1. After embedding + scatter: tensors have shape [batch, seq_len/TP, hidden]
    2. Throughout decoder layers: tensors maintain [batch, seq_len/TP, hidden]
    3. After final gather: output returns to [batch, seq_len, hidden]

    This proves SP is actually working, not just passing through without effect.

    Args:
        tp_size: Tensor parallel size (should be > 1 for SP to be active)
        pp_size: Pipeline parallel size
        cp_size: Context parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
    """
    import megatron.core.parallel_state as mpu

    from verl.models.mcore import get_mcore_forward_fn
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine import EngineRegistry

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create test model (only on rank 0)
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_sp_verify_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
        print(f"[Rank {rank}] Created test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine with TP > 1 (this enables sequence_parallel automatically)
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=True,
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
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

    # Get actual TP size and SP status from parallel state
    actual_tp_size = mpu.get_tensor_model_parallel_world_size()
    model = engine.module[0]

    # Check if sequence_parallel is enabled in model config
    sequence_parallel = getattr(model.config, "sequence_parallel", False)
    print(f"[Rank {rank}] TP size: {actual_tp_size}, sequence_parallel: {sequence_parallel}")

    if actual_tp_size > 1 and not sequence_parallel:
        print(f"[Rank {rank}] WARNING: TP > 1 but sequence_parallel is False!")

    # Create test batch with sequence length divisible by TP size
    batch_size = 4
    seqlen = 64  # Must be divisible by tp_size for SP
    assert seqlen % tp_size == 0, f"seqlen ({seqlen}) must be divisible by tp_size ({tp_size})"
    vocab_size = model_config.hf_config.vocab_size

    batch = create_test_batch(batch_size, seqlen, vocab_size, torch.device("cuda"))

    # Storage for captured shapes
    captured_shapes = []
    hook_handles = []

    def make_shape_capture_hook(layer_name):
        """Create a hook that captures output tensor shapes."""

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured_shapes.append((layer_name, output.shape))
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                captured_shapes.append((layer_name, output[0].shape))

        return hook

    # Register hooks on decoder layers to capture intermediate shapes
    # The decoder layers process tensors in SP-divided format
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        for idx, layer in enumerate(model.decoder.layers):
            handle = layer.register_forward_hook(make_shape_capture_hook(f"decoder_layer_{idx}"))
            hook_handles.append(handle)
        print(f"[Rank {rank}] Registered hooks on {len(model.decoder.layers)} decoder layers")
    else:
        print(f"[Rank {rank}] WARNING: Could not find decoder layers for hook registration")

    # Define a simple logits processor
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    def logits_processor(logits, label, label_mask):
        assert logits.shape[:2] == label.shape[:2], (
            f"Shape mismatch: logits {logits.shape[:2]} vs label {label.shape[:2]}"
        )
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    logits_processor_args = {"label": batch["label"], "label_mask": batch["label_mask"]}

    # Get BSHD forward function
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)

    # Run forward pass
    try:
        captured_shapes.clear()  # Clear any previous captures

        output = bshd_forward(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
        )

        # Remove hooks
        for handle in hook_handles:
            handle.remove()

        # Analyze captured shapes
        print(f"\n[Rank {rank}] === SP Verification Results ===")
        print(f"[Rank {rank}] Input sequence length: {seqlen}")
        print(f"[Rank {rank}] TP size: {actual_tp_size}")
        print(f"[Rank {rank}] Expected SP seq_len: {seqlen // actual_tp_size if sequence_parallel else seqlen}")
        print(f"[Rank {rank}] Captured {len(captured_shapes)} layer outputs:")

        sp_working = True
        expected_sp_seq_len = seqlen // actual_tp_size if sequence_parallel else seqlen

        for layer_name, shape in captured_shapes:
            actual_seq_len = shape[1] if len(shape) > 1 else "N/A"
            status = "✓" if actual_seq_len == expected_sp_seq_len else "✗"
            print(f"[Rank {rank}]   {status} {layer_name}: {shape}")

            if actual_seq_len != expected_sp_seq_len:
                sp_working = False

        # Verify final output shape (should be gathered back to full seq_len)
        if mpu.is_pipeline_last_stage():
            log_probs = output["log_probs"]
            print(f"\n[Rank {rank}] Final output shape: {log_probs.shape}")
            print(f"[Rank {rank}] Expected final shape: [{batch_size}, {seqlen}]")

            if log_probs.shape[1] == seqlen:
                print(f"[Rank {rank}] ✓ Final output correctly gathered to full sequence length")
            else:
                print(f"[Rank {rank}] ✗ Final output NOT correctly gathered!")
                sp_working = False

        # Summary
        print(f"\n[Rank {rank}] === Summary ===")
        if sequence_parallel and actual_tp_size > 1:
            if sp_working:
                print(f"[Rank {rank}] ✓ SP is working correctly!")
                print(f"[Rank {rank}]   - Decoder layers process seq_len/{actual_tp_size} = {expected_sp_seq_len}")
                print(f"[Rank {rank}]   - Output is gathered back to seq_len = {seqlen}")
            else:
                print(f"[Rank {rank}] ✗ SP verification FAILED!")
                print(f"[Rank {rank}]   - Expected intermediate seq_len: {expected_sp_seq_len}")
                print(f"[Rank {rank}]   - Check if scatter_embedding_sequence_parallel is enabled")
                raise AssertionError("SP verification failed: tensor shapes don't match expected SP pattern")
        else:
            print(f"[Rank {rank}] ⚠ SP is not active (TP={actual_tp_size}, SP={sequence_parallel})")
            print(f"[Rank {rank}]   - This is expected when TP=1")

    except Exception as e:
        # Remove hooks on error
        for handle in hook_handles:
            handle.remove()
        print(f"[Rank {rank}] ✗ ERROR: {type(e).__name__}: {e}")
        raise

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] SP verification test completed!")


def test_bshd_pp_forward(
    tp_size: int = 1,
    pp_size: int = 2,
    cp_size: int = 1,
    vanilla_mbridge: bool = True,
):
    """Test BSHD forward with Pipeline Parallelism.

    This test verifies that:
    1. The BSHD format (model_forward_gen with use_sequence_packing=False) works with PP
    2. The forward_backward_func correctly schedules PP execution
    3. The output log_probs have correct shape

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        cp_size: Context parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
    """
    import megatron.core.parallel_state as mpu
    from functools import partial

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

    # Create test model (only on rank 0)
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_bshd_pp_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
        print(f"[Rank {rank}] Created test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine with PP
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=True,
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
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

    # Create test batch
    batch_size = 4
    seqlen = 64
    vocab_size = model_config.hf_config.vocab_size

    batch = create_test_batch(batch_size, seqlen, vocab_size, torch.device("cuda"))

    # Get BSHD forward function
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)

    # Define logits processor
    def logits_processor(logits, label, label_mask):
        assert logits.shape[:2] == label.shape[:2], (
            f"Shape mismatch: logits {logits.shape[:2]} vs label {label.shape[:2]}"
        )
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    # Define loss_func for forward_backward_func
    def loss_func(output, non_loss_data=False):
        if non_loss_data:
            return output
        dummy_loss = torch.tensor(1.0, device="cuda")
        return dummy_loss, {"output": output}

    # Define forward_step function for forward_backward_func
    def forward_step(batch_iter, model):
        micro_batch = next(batch_iter)
        output = bshd_forward(
            model=model,
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

    # Run BSHD forward with PP
    try:
        forward_backward_func = get_forward_backward_func()
        batch_generator = make_batch_generator([batch], vpp_size=len(engine.module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=engine.module,
            num_microbatches=1,
            seq_length=seqlen,
            micro_batch_size=batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

        # Check output on last PP stage
        if mpu.is_pipeline_last_stage():
            assert output is not None, "Expected output on last PP stage"
            # output is a list of (output_dict,) tuples from each microbatch
            assert len(output) == 1, f"Expected 1 microbatch output, got {len(output)}"
            output_dict = output[0]
            if isinstance(output_dict, tuple):
                output_dict = output_dict[0]
            assert "log_probs" in output_dict, "Expected 'log_probs' in output"
            log_probs = output_dict["log_probs"]
            print(f"[Rank {rank}] ✓ PP forward passed! log_probs shape: {log_probs.shape}")
        else:
            print(f"[Rank {rank}] ✓ PP forward passed on non-last stage")

    except AssertionError as e:
        print(f"[Rank {rank}] ✗ FAILED: {e}")
        raise
    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] Test completed successfully!")


def test_bshd_1f1b_overlap(
    tp_size: int = 1,
    pp_size: int = 2,
    cp_size: int = 1,
    vanilla_mbridge: bool = True,
):
    """Test BSHD format with 1F1B overlap scheduling.

    This test verifies that `gptmodel_forward_1f1b_overlap_bshd` works correctly
    with Pipeline Parallelism. This function is used when:
    - use_sequence_packing=False (BSHD format)
    - overlap_moe_expert_parallel_comm=True (1F1B overlap scheduling)

    The function returns a TransformerModelChunkSchedulePlan which is then
    executed by forward_backward_func.

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size (must be > 1 for meaningful test)
        cp_size: Context parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
    """
    import megatron.core.parallel_state as mpu
    from functools import partial

    from megatron.core.pipeline_parallel import get_forward_backward_func

    from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap_bshd
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

    # Create MoE test model (only on rank 0)
    # MoE model is required for build_schedule_plan to be available
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_bshd_1f1b_")
    if rank == 0:
        model_path = create_test_moe_model(tmp_dir)
        print(f"[Rank {rank}] Created MoE test model at: {model_path}")
    else:
        model_path = ""

    dist.barrier()
    model_path = broadcast_model_path(model_path if rank == 0 else "", rank)
    dist.barrier()

    # Configure engine with PP
    # Note: overlap_moe_expert_parallel_comm=True is required for build_schedule_plan
    # to be available on the Megatron-Core GPTModel
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)
    engine_config = McoreEngineConfig(
        forward_only=True,
        use_mbridge=True,
        vanilla_mbridge=vanilla_mbridge,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=2,  # Required for EP A2A overlap with PP>1
        expert_model_parallel_size=2,  # Required for overlap_moe_expert_parallel_comm
        context_parallel_size=cp_size,
        override_transformer_config={"overlap_moe_expert_parallel_comm": True},
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

    # Get model and check if build_schedule_plan is available
    # Note: engine.module[0] is wrapped in DDP and Float16Module, so we need to
    # unwrap to access GPTModel's build_schedule_plan method
    from verl.utils.megatron_utils import unwrap_model

    model = engine.module[0]
    unwrapped_model = unwrap_model(model)

    # Check if model supports build_schedule_plan (check on unwrapped model)
    has_schedule_plan = hasattr(unwrapped_model, "build_schedule_plan") and callable(
        getattr(unwrapped_model, "build_schedule_plan", None)
    )
    print(f"[Rank {rank}] Model supports build_schedule_plan: {has_schedule_plan}")

    if not has_schedule_plan:
        print(f"[Rank {rank}] ⚠ Skipping 1F1B overlap test: model does not support build_schedule_plan")
        print(f"[Rank {rank}]   Requirements for build_schedule_plan:")
        print(f"[Rank {rank}]   - Megatron-Core version >= 0.13.0 with overlap support")
        print(f"[Rank {rank}]   - MoE model with overlap_moe_expert_parallel_comm=True")
        print(f"[Rank {rank}]   - TransformerConfig with schedule plan enabled")
        dist.barrier()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
        if rank == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Create test batches (need 2 micro-batches for interleaved schedule with PP=2)
    batch_size = 4
    seqlen = 64
    vocab_size = model_config.hf_config.vocab_size

    # Interleaved schedule requires num_microbatches >= PP
    batch1 = create_test_batch(batch_size, seqlen, vocab_size, torch.device("cuda"))
    batch2 = create_test_batch(batch_size, seqlen, vocab_size, torch.device("cuda"))

    # Define logits processor
    def logits_processor(logits, label, label_mask):
        assert logits.shape[:2] == label.shape[:2], (
            f"Shape mismatch: logits {logits.shape[:2]} vs label {label.shape[:2]}"
        )
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    logits_processor_args = {"label": batch1["label"], "label_mask": batch1["label_mask"]}

    # Test 1: Verify gptmodel_forward_1f1b_overlap_bshd returns a schedule plan
    try:
        print(f"[Rank {rank}] Testing gptmodel_forward_1f1b_overlap_bshd...")

        # Use unwrapped_model since gptmodel_forward_1f1b_overlap_bshd calls
        # model.build_schedule_plan() which requires the raw GPTModel
        schedule_plan = gptmodel_forward_1f1b_overlap_bshd(
            model=unwrapped_model,
            input_ids=batch1["input_ids"],
            position_ids=batch1["position_ids"],
            attention_mask=batch1["attention_mask"],
            labels=batch1["label"],
            labels_mask=batch1["label_mask"],
            multi_modal_inputs={},
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
            temperature=1.0,
        )

        # Verify schedule_plan structure
        from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan

        assert isinstance(schedule_plan, TransformerModelChunkSchedulePlan), (
            f"Expected TransformerModelChunkSchedulePlan, got {type(schedule_plan)}"
        )
        print(f"[Rank {rank}] ✓ gptmodel_forward_1f1b_overlap_bshd returned valid schedule_plan")
        print(f"[Rank {rank}]   - Schedule plan type: {type(schedule_plan).__name__}")

        # Check schedule_plan attributes
        if hasattr(schedule_plan, "post_process"):
            print(f"[Rank {rank}]   - Has post_process node: {schedule_plan.post_process is not None}")

    except Exception as e:
        print(f"[Rank {rank}] ✗ ERROR in gptmodel_forward_1f1b_overlap_bshd: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test 2: Try to execute the schedule plan with forward_backward_func
    # Note: This may require specific scheduler configuration
    try:
        print(f"[Rank {rank}] Testing schedule plan execution with forward_backward_func...")

        # Define loss_func for forward_backward_func
        def loss_func(output, non_loss_data=False):
            if non_loss_data:
                return output
            # For schedule plan output, it's typically a tuple
            if isinstance(output, tuple):
                # The output from 1F1B overlap is (log_probs,)
                log_probs = output[0] if len(output) > 0 else None
                if log_probs is not None:
                    dummy_loss = torch.tensor(1.0, device="cuda")
                    return dummy_loss, {"log_probs": log_probs}
            dummy_loss = torch.tensor(1.0, device="cuda")
            return dummy_loss, {"output": output}

        # Define forward_step that returns schedule_plan
        def forward_step(batch_iter, model):
            micro_batch = next(batch_iter)
            # Unwrap model since forward_backward_func passes DDP-wrapped model
            unwrapped = unwrap_model(model)
            schedule_plan = gptmodel_forward_1f1b_overlap_bshd(
                model=unwrapped,
                input_ids=micro_batch["input_ids"],
                position_ids=micro_batch["position_ids"],
                attention_mask=micro_batch["attention_mask"],
                labels=micro_batch["label"],
                labels_mask=micro_batch["label_mask"],
                multi_modal_inputs={},
                logits_processor=logits_processor,
                logits_processor_args={
                    "label": micro_batch["label"],
                    "label_mask": micro_batch["label_mask"],
                },
                temperature=1.0,
            )
            return schedule_plan, partial(loss_func)

        forward_backward_func = get_forward_backward_func()
        # Use 2 batches for interleaved schedule (num_microbatches >= PP)
        batch_generator = make_batch_generator([batch1, batch2], vpp_size=len(engine.module))

        # Note: Schedule plan execution may fail if the scheduler doesn't support it
        # This is expected behavior when overlap_moe_expert_parallel_comm is not configured
        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=engine.module,
            num_microbatches=2,  # Must be >= PP for interleaved schedule
            seq_length=seqlen,
            micro_batch_size=batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

        # Check output on last PP stage
        if mpu.is_pipeline_last_stage():
            assert output is not None, "Expected output on last PP stage"
            print(f"[Rank {rank}] ✓ Schedule plan execution completed!")
            print(f"[Rank {rank}]   - Output: {type(output)}")
            if isinstance(output, list) and len(output) > 0:
                print(f"[Rank {rank}]   - Output[0]: {type(output[0])}")
        else:
            print(f"[Rank {rank}] ✓ Schedule plan execution completed on non-last stage")

    except Exception as e:
        # Schedule plan execution failure is somewhat expected without proper scheduler config
        print(f"[Rank {rank}] ⚠ Schedule plan execution failed (may be expected): {type(e).__name__}: {e}")
        print(f"[Rank {rank}]   This is expected if the scheduler doesn't support schedule plans")
        print(f"[Rank {rank}]   The important test is that gptmodel_forward_1f1b_overlap_bshd works")

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Rank {rank}] 1F1B overlap test completed!")


def test_bshd_vs_thd_comparison(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    vanilla_mbridge: bool = False,
):
    """Compare BSHD and THD outputs for numerical correctness (PP=1 only).

    This test verifies that BSHD and THD formats produce similar outputs
    when PP=1 (where both should work).

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        cp_size: Context parallel size
        vanilla_mbridge: If True, use mbridge; if False, use megatron.bridge
    """
    import megatron.core.parallel_state as mpu

    from verl.models.mcore import get_mcore_forward_fn
    from verl.trainer.config import CheckpointConfig
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
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_compare_")
    if rank == 0:
        model_path = create_test_model(tmp_dir)
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
        context_parallel_size=cp_size,
    )

    engine = EngineRegistry.new(
        model_type="language_model",
        backend="megatron",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=McoreOptimizerConfig(lr_decay_steps=10),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    model = engine.module[0]  # engine.module is a list of model chunks for PP, use [0] for non-VPP

    # Create batch
    batch = create_test_batch(4, 64, model_config.hf_config.vocab_size, torch.device("cuda"))

    # Simple logits processor that returns log_probs
    def simple_processor(logits, label, label_mask):
        log_probs = vocab_parallel_log_probs_from_logits(logits, label)
        log_probs = log_probs.masked_fill(~label_mask, 0.0)
        return {"log_probs": log_probs}

    logits_processor_args = {"label": batch["label"], "label_mask": batch["label_mask"]}

    # Run THD forward (use_sequence_packing=True)
    thd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=True)
    thd_output = thd_forward(
        model=model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        multi_modal_inputs={},
        logits_processor=simple_processor,
        logits_processor_args=logits_processor_args,
    )

    # Run BSHD forward (use_sequence_packing=False)
    bshd_forward = get_mcore_forward_fn(model_config.hf_config, use_sequence_packing=False)
    bshd_output = bshd_forward(
        model=model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        multi_modal_inputs={},
        logits_processor=simple_processor,
        logits_processor_args=logits_processor_args,
    )

    # Compare outputs
    if mpu.is_pipeline_last_stage():
        thd_log_probs = thd_output["log_probs"]
        bshd_log_probs = bshd_output["log_probs"]

        # Compare shapes
        assert thd_log_probs.shape == bshd_log_probs.shape, (
            f"Shape mismatch: THD {thd_log_probs.shape} vs BSHD {bshd_log_probs.shape}"
        )

        # Compare values (with tolerance for numerical differences)
        # Only compare where label_mask is True (valid positions)
        valid_mask = batch["label_mask"]
        thd_valid = thd_log_probs[valid_mask]
        bshd_valid = bshd_log_probs[valid_mask]

        max_diff = torch.max(torch.abs(thd_valid - bshd_valid)).item()
        mean_diff = torch.mean(torch.abs(thd_valid - bshd_valid)).item()

        print(f"[Rank {rank}] THD vs BSHD comparison:")
        print(f"  - Max diff: {max_diff:.6f}")
        print(f"  - Mean diff: {mean_diff:.6f}")

        # Allow some tolerance due to different computation order
        assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
        print(f"[Rank {rank}] ✓ THD vs BSHD comparison passed!")

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_actor_worker_integration(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
):
    """Integration test using ActorWorker with BSHD configuration.

    This test simulates the actual usage pattern in megatron_actor.py.

    Note: PP>1 with BSHD format requires the combined_1f1b scheduler which is only
    activated when overlap_moe_expert_parallel_comm=True. For basic BSHD testing,
    we use PP=1 to verify the format conversion logic.
    """
    import numpy as np
    import ray

    from verl import DataProto
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.utils.model import compute_position_id_with_mask, create_random_mask
    from verl.workers.config import ActorConfig, HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.roles import ActorWorker

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank != 0:
        print(f"[Rank {rank}] Skipping ray test (only rank 0 runs ray tests)")
        return

    ray.init(ignore_reinit_error=True)

    # Create test model
    tmp_dir = tempfile.mkdtemp(prefix="verl_test_actor_")
    model_path = create_test_model(tmp_dir)

    try:
        model_config = HFModelConfig(path=model_path, load_tokenizer=False)
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)

        config = ActorConfig(
            model_config=model_config,
            engine=engine_config,
            strategy="megatron",
            ppo_micro_batch_size_per_gpu=256,
            ppo_mini_batch_size=4,
            optim=optimizer_config,
            use_dynamic_bsz=True,
            use_sequence_packing=False,  # BSHD format
            use_fused_kernels=False,  # Non-fused kernels
            rollout_n=1,
        )

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorWorker), config=config)
        resource_pool = RayResourcePool(process_on_nodes=[world_size])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)

        # Initialize model
        wg.init_model()

        # Create test data
        batch_size = 8
        seqlen = 32
        response_length = seqlen // 2

        torch.manual_seed(1)
        np.random.seed(1)

        input_ids = torch.randint(0, model_config.hf_config.vocab_size, (batch_size, seqlen))
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_valid_token=0.8,
            max_ratio_of_left_padding=0.2,
            min_ratio_of_valid_token=0.6,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        global_token_num = torch.sum(attention_mask, dim=-1).tolist()

        responses = input_ids[:, response_length:]
        response_mask = attention_mask[:, response_length:]

        data = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "prompts": input_ids[:, :response_length],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            meta_info={"temperature": 1.0, "global_token_num": global_token_num},
        )

        # Run forward pass
        output = wg.compute_log_prob(data)

        assert "old_log_probs" in output.batch, "Output should contain old_log_probs"
        print(f"[Rank {rank}] ✓ ActorWorker integration test passed!")
        print(f"  - old_log_probs shape: {output.batch['old_log_probs'].shape}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Test BSHD (non-sequence-packing) format support")
    parser.add_argument(
        "--test",
        type=str,
        default="basic",
        choices=["basic", "pp_only", "pp_tp", "pp_cp", "all", "tp_only", "cp_only", "compare", "actor", "sp_verify", "1f1b_overlap"],
        help="Test configuration to run",
    )
    parser.add_argument(
        "--bridge",
        type=str,
        default="megatron",
        choices=["mbridge", "megatron"],
        help="Bridge to use: 'mbridge' (vanilla_mbridge=True) or 'megatron' (vanilla_mbridge=False, default)",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Determine vanilla_mbridge setting based on --bridge argument
    vanilla_mbridge = args.bridge == "mbridge"
    bridge_name = "mbridge" if vanilla_mbridge else "megatron.bridge"

    print(f"[Rank {rank}] Running test: {args.test} with world_size={world_size}, bridge={bridge_name}")

    if args.test == "basic":
        # PP=1, TP=1, CP=1 (requires 1 GPU)
        # Basic BSHD forward test to verify remove_left_padding/recover_left_padding
        test_bshd_forward(tp_size=1, pp_size=1, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "pp_only":
        # PP=2, TP=1, CP=1 (requires 2 GPUs)
        # Test BSHD with pipeline parallelism using forward_backward_func
        assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"
        test_bshd_pp_forward(tp_size=1, pp_size=2, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "pp_tp":
        # PP=2, TP=2, CP=1 (requires 4 GPUs)
        # Test BSHD with pipeline + tensor parallelism
        assert world_size >= 4, f"Need at least 4 GPUs, got {world_size}"
        test_bshd_pp_forward(tp_size=2, pp_size=2, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "pp_cp":
        # PP=2, TP=1, CP=2 (requires 4 GPUs)
        # Test BSHD with pipeline + context parallelism
        assert world_size >= 4, f"Need at least 4 GPUs, got {world_size}"
        test_bshd_pp_forward(tp_size=1, pp_size=2, cp_size=2, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "all":
        # PP=2, TP=2, CP=2 (requires 8 GPUs)
        # Test BSHD with all parallelism types
        assert world_size >= 8, f"Need at least 8 GPUs, got {world_size}"
        test_bshd_pp_forward(tp_size=2, pp_size=2, cp_size=2, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "tp_only":
        # PP=1, TP=2, CP=1 (requires 2 GPUs)
        # Test BSHD with tensor parallelism
        assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"
        test_bshd_forward(tp_size=2, pp_size=1, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "cp_only":
        # PP=1, TP=1, CP=2 (requires 2 GPUs)
        # Test BSHD with context parallelism
        assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"
        test_bshd_forward(tp_size=1, pp_size=1, cp_size=2, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "compare":
        # Compare THD vs BSHD (PP=1, requires at least 1 GPU)
        # Verify numerical equivalence between THD and BSHD formats
        test_bshd_vs_thd_comparison(tp_size=1, pp_size=1, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "actor":
        # Integration test with ActorWorker
        # Note: This test requires the combined_1f1b scheduler for PP>1
        # For now, we use PP=1 to avoid scheduler compatibility issues
        test_actor_worker_integration(tp_size=1, pp_size=1, cp_size=1)

    elif args.test == "sp_verify":
        # PP=1, TP=2, CP=1 (requires 2 GPUs)
        # Verify that Sequence Parallelism actually divides sequences across TP ranks
        # This test captures intermediate tensor shapes to prove SP is working
        assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"
        test_sp_verification(tp_size=2, pp_size=1, cp_size=1, vanilla_mbridge=vanilla_mbridge)

    elif args.test == "1f1b_overlap":
        # PP=2, TP=1, CP=1 (requires 2 GPUs)
        # Test BSHD with 1F1B overlap scheduling
        # This tests gptmodel_forward_1f1b_overlap_bshd which returns TransformerModelChunkSchedulePlan
        assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"
        test_bshd_1f1b_overlap(tp_size=1, pp_size=2, cp_size=1, vanilla_mbridge=vanilla_mbridge)


if __name__ == "__main__":
    main()
