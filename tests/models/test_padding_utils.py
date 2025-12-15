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
Unit tests for preprocess_bshd and postprocess_bshd utilities.

These tests verify that the padding conversion functions work correctly
in isolation, without requiring a full distributed environment.

Note: Tests import via backward compatibility aliases (remove_left_padding/recover_left_padding)
      which map to preprocess_bshd/postprocess_bshd.

Run with:
    python -m pytest tests/models/test_padding_utils.py -v

    # Or with torchrun for distributed tests
    torchrun --nproc_per_node=2 tests/models/test_padding_utils.py
"""

import argparse
import sys
from unittest import mock

import torch


def setup_mock_mpu(cp_size: int = 1, cp_rank: int = 0, tp_size: int = 1):
    """Setup mock megatron parallel state for unit tests."""
    mock_mpu = mock.MagicMock()
    mock_mpu.get_context_parallel_world_size.return_value = cp_size
    mock_mpu.get_context_parallel_rank.return_value = cp_rank
    mock_mpu.get_tensor_model_parallel_world_size.return_value = tp_size
    return mock_mpu


def create_left_padded_sample(
    batch_size: int,
    seq_len: int,
    padding_ratios: list[float] = None,
    device: torch.device = None,
    seed: int = 42,
):
    """Create left-padded input tensors for testing.

    Args:
        batch_size: Number of samples in batch
        seq_len: Total sequence length
        padding_ratios: List of padding ratios for each sample (default: varies per sample)
        device: Device to create tensors on
        seed: Random seed for reproducibility

    Returns:
        input_ids, attention_mask, position_ids
    """
    torch.manual_seed(seed)

    if device is None:
        device = torch.device("cpu")

    if padding_ratios is None:
        # Default: varying padding ratios
        padding_ratios = [(i + 1) / (batch_size + 2) for i in range(batch_size)]

    input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for i, ratio in enumerate(padding_ratios):
        pad_len = int(seq_len * ratio)
        # Set padding tokens to 0
        input_ids[i, :pad_len] = 0
        # Set attention mask (1 for valid, 0 for padding)
        attention_mask[i, pad_len:] = True
        # Set position ids for valid tokens
        valid_len = seq_len - pad_len
        position_ids[i, pad_len:] = torch.arange(valid_len, device=device)

    return input_ids, attention_mask, position_ids


class TestRemoveLeftPadding:
    """Tests for remove_left_padding function."""

    def test_output_shape_basic(self):
        """Test that output shape is correct for basic case (CP=1)."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 4, 64
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # Output batch size should be preserved
            assert new_ids.shape[0] == batch_size
            # Attention mask should be 4D [B, 1, 1, S]
            assert new_mask.ndim == 4
            assert new_mask.shape[0] == batch_size
            assert new_mask.shape[1] == 1
            assert new_mask.shape[2] == 1

    def test_4d_attention_mask_format(self):
        """Test that output attention_mask is 4D [B, 1, 1, S] format."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 32
            input_ids, attention_mask, position_ids = create_left_padded_sample(
                batch_size, seq_len, padding_ratios=[0.25, 0.25]
            )

            _, new_mask, _ = remove_left_padding(input_ids, attention_mask, position_ids)

            # Should be 4D for TransformerEngine FusedAttention
            assert new_mask.ndim == 4, f"Expected 4D, got {new_mask.ndim}D"
            assert new_mask.shape[1] == 1, f"Expected shape[1]=1, got {new_mask.shape[1]}"
            assert new_mask.shape[2] == 1, f"Expected shape[2]=1, got {new_mask.shape[2]}"

    def test_right_padding_conversion(self):
        """Test that left padding is correctly converted to right padding."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 32
            input_ids, attention_mask, position_ids = create_left_padded_sample(
                batch_size, seq_len, padding_ratios=[0.25, 0.5]
            )

            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # Convert 4D mask to 2D for easier checking
            new_mask_2d = new_mask.squeeze(1).squeeze(1)

            # For each sample, valid tokens should be at the beginning (right-padding)
            for i in range(batch_size):
                valid_len = attention_mask[i].sum().item()
                # First valid_len tokens should be marked as valid
                assert new_mask_2d[i, :valid_len].all(), f"Sample {i}: expected valid tokens at beginning"
                # Original valid tokens should be preserved
                original_valid = input_ids[i, attention_mask[i]]
                assert torch.equal(new_ids[i, :valid_len], original_valid), f"Sample {i}: token mismatch"

    def test_variable_padding_lengths(self):
        """Test handling of different padding lengths per sample."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 4, 64
            padding_ratios = [0.1, 0.3, 0.5, 0.7]

            input_ids, attention_mask, position_ids = create_left_padded_sample(
                batch_size, seq_len, padding_ratios=padding_ratios
            )

            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            new_mask_2d = new_mask.squeeze(1).squeeze(1)

            for i, ratio in enumerate(padding_ratios):
                valid_len = int(seq_len * (1 - ratio))
                # First valid_len tokens should be non-zero (original tokens were non-zero)
                assert (new_ids[i, :valid_len] != 0).all(), f"Sample {i}: valid tokens should be non-zero"

    def test_no_padding_case(self):
        """Test when there is no padding at all."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 32
            # No padding
            input_ids, attention_mask, position_ids = create_left_padded_sample(
                batch_size, seq_len, padding_ratios=[0.0, 0.0]
            )

            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # All tokens should be valid
            new_mask_2d = new_mask.squeeze(1).squeeze(1)
            # Output may have alignment padding, but original tokens should be preserved
            for i in range(batch_size):
                assert torch.equal(new_ids[i, :seq_len], input_ids[i]), f"Sample {i}: tokens should be unchanged"

    def test_all_padding_case(self):
        """Test when entire sequence is padding (edge case)."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 32

            # All padding (valid_len = 0 per sample)
            input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

            # Should not crash
            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # All output should be zeros (no valid tokens)
            assert (new_ids == 0).all(), "All padding input should result in all-zero output"

    def test_fixed_seq_len_parameter(self):
        """Test that fixed_seq_len parameter constrains output length."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 64
            fixed_len = 48

            input_ids, attention_mask, position_ids = create_left_padded_sample(
                batch_size, seq_len, padding_ratios=[0.25, 0.25]
            )

            new_ids, _, _ = remove_left_padding(input_ids, attention_mask, position_ids, fixed_seq_len=fixed_len)

            # Output should be at least fixed_len (may have alignment padding)
            assert new_ids.shape[1] >= fixed_len, f"Expected seq_len >= {fixed_len}, got {new_ids.shape[1]}"

    def test_pre_process_false(self):
        """Test behavior when pre_process=False."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import remove_left_padding

            batch_size, seq_len = 2, 32
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            new_ids, new_mask, new_pos = remove_left_padding(
                input_ids, attention_mask, position_ids, pre_process=False
            )

            # When pre_process=False, input_ids should be returned unchanged
            assert torch.equal(new_ids, input_ids), "pre_process=False should return original input_ids"


class TestRecoverLeftPadding:
    """Tests for recover_left_padding function."""

    def test_roundtrip_consistency(self):
        """Test that remove -> recover gives back original values at valid positions."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding, remove_left_padding

            batch_size, seq_len = 4, 64
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            # Forward: remove left padding
            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # Create a result tensor with same shape as new_ids (simulating model output)
            # Use float for typical model output
            dummy_result = new_ids.unsqueeze(-1).float()  # [batch, new_seq, 1]

            # Backward: recover left padding
            recovered = recover_left_padding(dummy_result, new_mask, attention_mask, seq_len)

            # Check shape
            assert recovered.shape == (batch_size, seq_len, 1), f"Expected {(batch_size, seq_len, 1)}, got {recovered.shape}"

            # Check values at valid positions
            recovered_squeezed = recovered.squeeze(-1).long()
            for i in range(batch_size):
                valid_mask = attention_mask[i]
                original_valid = input_ids[i, valid_mask]
                recovered_valid = recovered_squeezed[i, valid_mask]
                assert torch.equal(original_valid, recovered_valid), f"Sample {i}: roundtrip mismatch"

    def test_2d_result_tensor(self):
        """Test recovery with 2D result tensor (e.g., log_probs)."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding, remove_left_padding

            batch_size, seq_len = 2, 32
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # 2D result (like log_probs before unsqueeze)
            dummy_result = torch.randn(batch_size, new_ids.shape[1])

            # Need to add last dim for recover_left_padding
            recovered = recover_left_padding(dummy_result.unsqueeze(-1), new_mask, attention_mask, seq_len)

            assert recovered.shape == (batch_size, seq_len, 1)

    def test_post_process_false(self):
        """Test that post_process=False returns input unchanged."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding

            batch_size, seq_len = 2, 32
            result = torch.randn(batch_size, seq_len, 10)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            recovered = recover_left_padding(result, attention_mask, attention_mask, seq_len, post_process=False)

            assert torch.equal(recovered, result), "post_process=False should return input unchanged"

    def test_4d_attention_mask_input(self):
        """Test that 4D attention mask is correctly handled."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding, remove_left_padding

            batch_size, seq_len = 2, 32
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            new_ids, new_mask, _ = remove_left_padding(input_ids, attention_mask, position_ids)

            # new_mask is 4D [B, 1, 1, S]
            assert new_mask.ndim == 4

            dummy_result = torch.randn(batch_size, new_ids.shape[1], 1)

            # Should handle 4D mask without error
            recovered = recover_left_padding(dummy_result, new_mask, attention_mask, seq_len)

            assert recovered.shape == (batch_size, seq_len, 1)


class TestPaddingUtilsIntegration:
    """Integration tests for padding utilities."""

    def test_logits_processor_args_transformation(self):
        """Test that label and label_mask can be correctly transformed."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding, remove_left_padding

            batch_size, seq_len = 4, 64
            vocab_size = 1000
            response_len = 16  # Last 16 tokens are response

            # Create inputs
            input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len)

            # Create label and label_mask (typical for RL training)
            label = input_ids.clone()
            label_mask = attention_mask.clone()
            label_mask[:, :-response_len] = False  # Only response tokens are valid for loss

            # Transform input_ids
            new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

            # Transform label (as done in model_forward.py)
            label_transformed, _, _ = remove_left_padding(
                label.unsqueeze(-1),  # Add last dim
                attention_mask,
                position_ids,
            )
            label_transformed = label_transformed.squeeze(-1)

            # Transform label_mask
            label_mask_transformed, _, _ = remove_left_padding(
                label_mask.unsqueeze(-1).float(),  # Convert to float and add dim
                attention_mask,
                position_ids,
            )
            label_mask_transformed = label_mask_transformed.squeeze(-1).bool()

            # Verify shapes match
            assert label_transformed.shape == new_ids.shape, "label shape should match new_ids"
            assert label_mask_transformed.shape == new_ids.shape, "label_mask shape should match new_ids"

            # Create dummy log_probs (simulating logits_processor output)
            log_probs = torch.randn_like(label_transformed.float()) * -5  # Negative values

            # Recover to original format
            log_probs_recovered = recover_left_padding(
                log_probs.unsqueeze(-1), new_mask, attention_mask, seq_len
            ).squeeze(-1)

            # Verify final shape
            assert log_probs_recovered.shape == (batch_size, seq_len)

    def test_batch_independence(self):
        """Test that samples in a batch don't affect each other's transformation."""
        with mock.patch("verl.models.mcore.util.mpu", setup_mock_mpu(cp_size=1)):
            from verl.models.mcore.util import recover_left_padding, remove_left_padding

            seq_len = 64

            # Sample A alone
            ids_A, mask_A, pos_A = create_left_padded_sample(1, seq_len, padding_ratios=[0.2], seed=100)
            new_A, new_mask_A, _ = remove_left_padding(ids_A, mask_A, pos_A)

            # Sample B alone
            ids_B, mask_B, pos_B = create_left_padded_sample(1, seq_len, padding_ratios=[0.5], seed=200)
            new_B, new_mask_B, _ = remove_left_padding(ids_B, mask_B, pos_B)

            # [A, B] together
            ids_AB = torch.cat([ids_A, ids_B], dim=0)
            mask_AB = torch.cat([mask_A, mask_B], dim=0)
            pos_AB = torch.cat([pos_A, pos_B], dim=0)
            new_AB, new_mask_AB, _ = remove_left_padding(ids_AB, mask_AB, pos_AB)

            # Sample A's valid tokens should be identical in both cases
            valid_len_A = mask_A[0].sum().item()
            assert torch.equal(
                new_A[0, :valid_len_A], new_AB[0, :valid_len_A]
            ), "Sample A should be same alone or in batch"

            # Sample B's valid tokens should be identical
            valid_len_B = mask_B[0].sum().item()
            assert torch.equal(
                new_B[0, :valid_len_B], new_AB[1, :valid_len_B]
            ), "Sample B should be same alone or in batch"


def run_pytest():
    """Run tests using pytest."""
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))


def run_distributed_tests(tp_size: int = 1, cp_size: int = 1):
    """Run tests in distributed environment with real megatron parallel state."""
    import os

    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Initialize megatron parallel state
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
    )

    print(f"[Rank {rank}] Running distributed tests with TP={tp_size}, CP={cp_size}")

    # Run actual tests with real distributed environment
    from verl.models.mcore.util import recover_left_padding, remove_left_padding

    device = torch.device("cuda")
    batch_size, seq_len = 4, 128

    input_ids, attention_mask, position_ids = create_left_padded_sample(batch_size, seq_len, device=device)

    # Test remove_left_padding
    new_ids, new_mask, new_pos = remove_left_padding(input_ids, attention_mask, position_ids)

    print(f"[Rank {rank}] Input shape: {input_ids.shape}, Output shape: {new_ids.shape}")
    print(f"[Rank {rank}] Attention mask shape: {new_mask.shape}")

    # Test roundtrip
    dummy_result = new_ids.unsqueeze(-1).float()
    recovered = recover_left_padding(dummy_result, new_mask, attention_mask, seq_len)

    print(f"[Rank {rank}] Recovered shape: {recovered.shape}")

    # Verify roundtrip
    recovered_squeezed = recovered.squeeze(-1).long()
    all_match = True
    for i in range(batch_size):
        valid_mask = attention_mask[i]
        if not torch.equal(input_ids[i, valid_mask], recovered_squeezed[i, valid_mask]):
            all_match = False
            print(f"[Rank {rank}] Sample {i}: MISMATCH")

    if all_match:
        print(f"[Rank {rank}] ✓ All roundtrip tests passed!")
    else:
        print(f"[Rank {rank}] ✗ Roundtrip test failed!")

    # Cleanup
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--distributed", action="store_true", help="Run distributed tests")
    args = parser.parse_args()

    if args.distributed:
        run_distributed_tests(tp_size=args.tp, cp_size=args.cp)
    else:
        run_pytest()
