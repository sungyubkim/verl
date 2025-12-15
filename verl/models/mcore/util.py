# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import math

import torch
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

from verl.utils.model import CausalLMOutputForPPO


def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True, use_fp8_padding=False
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if use_fp8_padding:
        # if fp8 is enabled, ensure the sequence is padded to multiples of 16 for better performance
        original_align_size = align_size
        align_size = math.lcm(16, align_size)

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    if use_fp8_padding:
        # make sure all the sequences are padded to multiples of 128 for TE compatibility
        align_size_last = original_align_size * 128
        pad_size_last = (align_size_last - cu_seqlens_padded[-1] % align_size_last) % align_size_last
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx : start_idx + s]
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


def preprocess_thd_no_padding(
    input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess THD (Token-Head-Dimension) packed sequences for nested tensor inputs.
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    seqlens_in_batch = input_ids.offsets().diff()

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        if need_roll:
            saved_roll_dict = {}
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

            if need_roll:
                # Handle roll for cp_size > 1 case
                saved_roll_dict[start_idx + half_seqlen - 1] = d[(cp_rank + 1) * half_seqlen]
                if remain_len > 0:
                    if remain_end == d.shape[0]:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[0]
                    else:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[remain_end]

        if need_roll:
            input_ids_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
            if len(saved_roll_dict) > 0:
                for k, v in saved_roll_dict.items():
                    input_ids_rmpad[k] = v

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_thd_no_padding(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    input_ids: torch.Tensor,
    batch_size: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    # The reason why we use input_ids.offsets() instead of packed_seq_params.cu_seqlens_q.diff()
    # is that the latter one is the padded length, while the former one is the original length.
    cu_seqlens = input_ids.offsets()
    seq_lens_cpu: list[int] = cu_seqlens.diff().tolist()

    output_new = []

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]

    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new.append(output[0][start_idx : start_idx + s])
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new.append(tmp[:s_len])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor


def preprocess_bshd(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
    fixed_seq_len: int = None,
):
    """
    Preprocess BSHD (Batch-Sequence-Head-Dimension) format sequences.

    When Context Parallelism (CP) is enabled, this function also splits each sequence
    into chunks distributed across CP ranks. Each GPU gets 2 chunks following the
    load-balancing pattern for causal masking:
    - GPU0: chunk[0] + chunk[-1] (first and last)
    - GPU1: chunk[1] + chunk[-2] (second and second-last)
    - ...

    Args:
        input_ids: Input token IDs tensor
        attention_mask: 2D attention mask tensor
        position_ids: Position IDs tensor
        sequence_parallel: Whether sequence parallelism is enabled
        pre_process: Whether this rank handles input preprocessing
        fixed_seq_len: If provided, use this as the sequence length instead of
            computing from attention_mask. Required for PP>1 to ensure consistent
            tensor shapes across micro-batches for P2P communication buffers.

    Returns:
        new_input_ids: Right-padded input_ids (split for CP if cp_size > 1)
        new_attention_mask: 4D attention mask [batch_size, 1, 1, seq_len] for
            TransformerEngine FusedAttention compatibility with BSHD format
        new_position_ids: Right-padded position_ids (split for CP if cp_size > 1)
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)

    # Use fixed_seq_len if provided (for PP>1 to ensure consistent P2P buffer shapes)
    # Otherwise, compute dynamically from attention_mask
    if fixed_seq_len is not None:
        seq_len = fixed_seq_len
    else:
        seq_len = seq_lens.max().item()

    # Calculate alignment based on CP, SP, and cuDNN requirements
    if cp_size > 1:
        # CP requires alignment to cp_size * 2 * 64 for proper chunk splitting
        sp_world_size = mpu.get_tensor_model_parallel_world_size() if sequence_parallel else 1
        alignment = cp_size * 2 * max(64, sp_world_size)
    elif sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        alignment = sp_world_size
    else:
        # Pad to multiple of 64 for cuDNN FusedAttention kernel alignment
        # Reference: https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/operations/Attention.html
        alignment = 64

    pad_size = (alignment - seq_len % alignment) % alignment
    seq_len_padded = seq_len + pad_size

    if cp_size > 1:
        # Each GPU gets seq_len_padded / cp_size tokens after splitting
        seq_len_per_gpu = seq_len_padded // cp_size
        shape[1] = seq_len_per_gpu
    else:
        shape[1] = seq_len_padded

    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)

    # Create 2D mask first, then convert to 4D for FusedAttention
    new_attention_mask_2d = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, shape[1])
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, shape[1]))

    for i in range(batch_size):
        valid_len = seq_lens[i].item()

        if cp_size <= 1:
            # Standard path: simple right-padding conversion
            if pre_process:
                new_input_ids[i, :valid_len] = input_ids[i, attention_mask[i]]
            new_attention_mask_2d[i, :valid_len] = attention_mask[i, attention_mask[i]]
            new_position_ids[i, :valid_len] = position_ids[i, attention_mask[i]]
        else:
            # CP chunk splitting path
            # Split each sequence into cp_size * 2 chunks, each GPU gets 2 chunks
            chunk_size = seq_len_padded // (cp_size * 2)
            half_seq = seq_len_per_gpu // 2

            # Extract original data (valid tokens only)
            orig_data = input_ids[i, attention_mask[i]]  # [valid_len]
            orig_pos = position_ids[i, attention_mask[i]]  # [valid_len]

            # First chunk (from the front): chunk[cp_rank]
            first_start = chunk_size * cp_rank
            first_end = min(first_start + chunk_size, valid_len)
            first_len = max(0, first_end - first_start)

            if first_len > 0:
                if pre_process:
                    new_input_ids[i, :first_len] = orig_data[first_start:first_end]
                new_attention_mask_2d[i, :first_len] = 1
                new_position_ids[i, :first_len] = orig_pos[first_start:first_end]

            # Second chunk (from the back): chunk[-(cp_rank+1)]
            # For load balancing with causal masking
            second_start = seq_len_padded - chunk_size * (cp_rank + 1)
            second_end = seq_len_padded - chunk_size * cp_rank
            # Clamp to valid token range
            second_start = max(0, min(second_start, valid_len))
            second_end = max(0, min(second_end, valid_len))
            second_len = max(0, second_end - second_start)

            if second_len > 0:
                if pre_process:
                    new_input_ids[i, half_seq : half_seq + second_len] = orig_data[second_start:second_end]
                new_attention_mask_2d[i, half_seq : half_seq + second_len] = 1
                new_position_ids[i, half_seq : half_seq + second_len] = orig_pos[second_start:second_end]

    # Convert to 4D [batch_size, 1, 1, seq_len] for TransformerEngine FusedAttention BSHD compatibility
    new_attention_mask = new_attention_mask_2d.unsqueeze(1).unsqueeze(1)

    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def postprocess_bshd(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    """
    Postprocess BSHD (Batch-Sequence-Head-Dimension) format sequences.

    When Context Parallelism (CP) is enabled, this function first gathers results
    from all CP ranks using all-gather, reassembles the chunks in the correct order,
    then converts back to the original left-padded format.

    Args:
        result: Model output tensor [batch, seq_len_per_gpu, ...]
        attention_mask: 4D or 2D attention mask from remove_left_padding
        original_attention_mask: Original 2D attention mask before remove_left_padding
        origin_seqlen: Original sequence length before any processing
        post_process: Whether this rank does post-processing

    Returns:
        Tensor in original left-padded format [batch, origin_seqlen, ...]
    """
    if not post_process:
        return result

    cp_size = mpu.get_context_parallel_world_size()

    # Handle 4D attention mask from remove_left_padding [batch, 1, 1, seq_len] -> [batch, seq_len]
    if attention_mask.ndim == 4:
        attention_mask = attention_mask.squeeze(1).squeeze(1)

    shape = list(result.shape)
    batch_size = shape[0]
    seq_len_per_gpu = shape[1]

    if cp_size > 1:
        # All-gather across CP ranks to collect results from all GPUs
        cp_group = mpu.get_context_parallel_group()

        # Gather results from all CP ranks
        # Use detach() for other ranks, then replace current rank with original to preserve gradient
        gathered_results = [torch.empty_like(result) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered_results, result.detach(), group=cp_group)
        gathered_results[mpu.get_context_parallel_rank()] = result

        # Reconstruct full sequence from chunks
        half_seq = seq_len_per_gpu // 2
        full_seq_len = seq_len_per_gpu * cp_size
        chunk_size = full_seq_len // (cp_size * 2)

        full_shape = list(shape)
        full_shape[1] = full_seq_len
        full_result = torch.zeros(dtype=result.dtype, device=result.device, size=full_shape)

        for rank in range(cp_size):
            # First chunk: restore to position chunk_size * rank
            first_dst_start = chunk_size * rank
            full_result[:, first_dst_start : first_dst_start + half_seq] = gathered_results[rank][:, :half_seq]

            # Second chunk: restore to position (full_seq_len - chunk_size * (rank + 1))
            second_dst_start = full_seq_len - chunk_size * (rank + 1)
            full_result[:, second_dst_start : second_dst_start + half_seq] = gathered_results[rank][:, half_seq:]

        result = full_result
        # Update attention_mask to match full sequence
        # For CP, we need to reconstruct full attention mask as well
        full_attention_mask = torch.zeros(
            dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, full_seq_len)
        )
        # Gather attention masks from all ranks (same pattern for consistency)
        gathered_masks = [torch.empty_like(attention_mask) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered_masks, attention_mask.detach(), group=cp_group)
        gathered_masks[mpu.get_context_parallel_rank()] = attention_mask

        for rank in range(cp_size):
            first_dst_start = chunk_size * rank
            full_attention_mask[:, first_dst_start : first_dst_start + half_seq] = gathered_masks[rank][:, :half_seq]
            second_dst_start = full_seq_len - chunk_size * (rank + 1)
            full_attention_mask[:, second_dst_start : second_dst_start + half_seq] = gathered_masks[rank][:, half_seq:]

        attention_mask = full_attention_mask

    # Convert back to original left-padded format
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        valid_len = attention_mask[i].sum().long().item()
        new_result[i, original_attention_mask[i]] = result[i, :valid_len]

    return new_result


def preprocess_bshd_no_padding(input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False):
    """
    Preprocess BSHD sequences for nested tensor inputs.
    Converts nested tensor to dense BSHD format with attention mask and position IDs.

    Args:
        input_ids: Nested tensor input IDs
        pre_process: Whether this rank handles input preprocessing
        need_roll: Whether to roll input_ids by -1 for label creation

    Returns:
        input_ids_bshd: Dense tensor [batch_size, max_seqlen]
        attention_mask: Boolean mask [batch_size, max_seqlen]
        position_ids: Position IDs [batch_size, max_seqlen]
    """
    cp_size = mpu.get_context_parallel_world_size()
    # TODO: support context parallel size > 1
    assert cp_size == 1, "Context parallel size > 1 not yet supported for BSHD no-padding"

    batch_size = input_ids.shape[0]
    seqlens_in_batch = input_ids.offsets().diff()
    max_seqlen = seqlens_in_batch.max().item()
    if mpu.get_tensor_model_parallel_world_size() > 1:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - max_seqlen % sp_world_size) % sp_world_size
        max_seqlen = max_seqlen + pad_size

    attention_mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=input_ids.device)
    input_ids_bshd = torch.zeros(batch_size, max_seqlen, dtype=input_ids.dtype, device=input_ids.device)
    for i in range(batch_size):
        attention_mask[i, : seqlens_in_batch[i]] = True
        input_ids_bshd[i, : seqlens_in_batch[i]] = input_ids[i]
    position_ids = torch.arange(max_seqlen, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids_bshd)
    if need_roll:
        input_ids_bshd = torch.roll(input_ids_bshd, shifts=-1, dims=1)

    return input_ids_bshd, attention_mask, position_ids


def postprocess_bshd_no_padding(
    output: torch.Tensor,
    attention_mask: torch.Tensor,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess BSHD sequences back to nested tensor format.

    Args:
        output: Dense output tensor [batch_size, max_seqlen, ...]
        attention_mask: Boolean mask [batch_size, max_seqlen]
        post_process: Whether this rank handles output postprocessing

    Returns:
        Nested tensor with variable-length sequences
    """
    if not post_process:
        return output

    batch_size = output.shape[0]
    output_new = []

    for i in range(batch_size):
        mask = attention_mask[i].bool()
        output_new.append(output[i][mask])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor


def postprocess_packed_seqs_for_dict_output(
    labels_mask: torch.Tensor,
    output: CausalLMOutputForPPO,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> dict[str, torch.Tensor]:
    """_summary_
    For fused kernels, the output is a dictionary with keys like 'log_probs', 'entropy', etc.
    This function post-processes each tensor in the output dictionary.
    Args:
        output (CausalLMOutputForPPO): _description_
        packed_seq_params (PackedSeqParams): _description_
        attention_mask (torch.Tensor): _description_
        batch_size (int): _description_
        seq_len (int): _description_
        post_process (bool, optional): _description_. Defaults to True.
    Returns:
        CausalLMOutputForPPO: _description_
    """
    ret = {}
    output.entropy = output.entropy.view(1, -1)
    output.log_probs = output.log_probs.view(1, -1)
    output.log_probs = output.log_probs.masked_fill(~labels_mask, 0.0)
    ret["entropy"] = postprocess_packed_seqs(
        output.entropy, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
    )
    ret["log_probs"] = postprocess_packed_seqs(
        output.log_probs, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
    )
    return ret


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# These aliases are provided for backward compatibility with existing code.
# Please use the new function names:
#   - preprocess_thd_no_padding (was preprocess_packed_seqs_no_padding)
#   - postprocess_thd_no_padding (was postprocess_packed_seqs_no_padding)
#   - preprocess_bshd (was remove_left_padding)
#   - postprocess_bshd (was recover_left_padding)
# =============================================================================
remove_left_padding = preprocess_bshd
recover_left_padding = postprocess_bshd
preprocess_packed_seqs_no_padding = preprocess_thd_no_padding
postprocess_packed_seqs_no_padding = postprocess_thd_no_padding
