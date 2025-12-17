# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import warnings
from typing import Optional

import torch

try:
    from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass

from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs
from verl.utils.device import get_device_name
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction

device_name = get_device_name()


# from megatron.core.transformer.transformer_block import get_num_layers_to_build
def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if hasattr(config, "pipeline_model_parallel_layout") and config.pipeline_model_parallel_layout is not None:
        from megatron.core.transformer.enums import LayerType

        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
        assert not (config.account_for_embedding_in_pipeline_split or config.account_for_loss_in_pipeline_split), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        )
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def merge_router_topk_indices(attention_mask, input_ids, mini_layer_topk_idx_list, tf_config, vp_rank=None):
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx.to(torch.uint8))  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).to(device_name)
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = (
            gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
            .unsqueeze(0)
            .contiguous()
        )

        batch_size, seq_len = attention_mask.shape[:2]
        _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        layers_topk_idx = postprocess_packed_seqs(
            layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
        )
        mini_layer_topk_idx_list.append(layers_topk_idx.cpu())


def _preprocess_router_indices_bshd(layers_topk_idx, attention_mask, sequence_parallel=False):
    """
    BSHD format router indices preprocessing. Applies CP/SP processing logic similar to preprocess_bshd.

    Args:
        layers_topk_idx: [bs, max_seq_len, layer_num, topk]
        attention_mask: [bs, max_seq_len]
        sequence_parallel: Whether sequence parallelism is enabled (used for alignment calculation)

    Returns:
        [bs, seq_len_per_gpu, layer_num, topk] - CP split + alignment applied indices
        Padding positions are filled with 255 (sentinel value for uint8 dtype)
    """
    # === DEBUG INPUT ===
    _debug_rank = mpu.get_tensor_model_parallel_rank() if mpu.is_initialized() else 0
    if _debug_rank == 0:
        valid_mask = attention_mask[0].bool()
        _valid_cnt = valid_mask.sum().item()
        _total_nonzero = (layers_topk_idx > 0).sum().item()
        _valid_data = layers_topk_idx[0, valid_mask] if valid_mask.any() else None
        _valid_nonzero = (_valid_data > 0).sum().item() if _valid_data is not None else 0
        _first_v = _valid_data[0, 0, 0].item() if _valid_data is not None and len(_valid_data) > 0 else -1
        _last_v = _valid_data[-1, 0, 0].item() if _valid_data is not None and len(_valid_data) > 0 else -1
        # Find first token with any non-zero value
        _token_has_nonzero = (_valid_data.view(_valid_cnt, -1) > 0).any(dim=1) if _valid_data is not None else None
        _first_nz_pos = _token_has_nonzero.int().argmax().item() if (_token_has_nonzero is not None and _token_has_nonzero.any()) else -1
        print(
            f"[BSHD DEBUG] INPUT | idx={layers_topk_idx.shape}, seq_lens={attention_mask.sum(dim=1).tolist()}, "
            f"total_nz={_total_nonzero}, valid_nz={_valid_nonzero}, "
            f"first_v={_first_v}, last_v={_last_v}, first_nz_pos={_first_nz_pos}"
        )
    # === DEBUG END ===

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    batch_size, max_seq_len = attention_mask.shape
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()

    # Alignment calculation (same logic as preprocess_bshd)
    if cp_size > 1:
        sp_world_size = mpu.get_tensor_model_parallel_world_size() if sequence_parallel else 1
        alignment = cp_size * 2 * max(64, sp_world_size)
    elif sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        alignment = sp_world_size
    else:
        alignment = 64

    pad_size = (alignment - seq_len % alignment) % alignment
    seq_len_padded = seq_len + pad_size

    if cp_size > 1:
        seq_len_per_gpu = seq_len_padded // cp_size
    else:
        seq_len_per_gpu = seq_len_padded

    # Initialize output tensor with 255 (padding sentinel for uint8)
    # 255 is used because: 1) uint8 doesn't support -1, 2) num_experts is typically < 255
    # Router replay patch will detect 255 and compute regular topk for those positions
    PADDING_EXPERT_IDX = 255
    layer_num, topk = layers_topk_idx.shape[2], layers_topk_idx.shape[3]
    result = torch.full(
        (batch_size, seq_len_per_gpu, layer_num, topk),
        fill_value=PADDING_EXPERT_IDX,
        dtype=layers_topk_idx.dtype,
        device=layers_topk_idx.device,
    )

    for i in range(batch_size):
        valid_len = seq_lens[i].item()

        # Extract original data (valid tokens only)
        orig_indices = layers_topk_idx[i, attention_mask[i].bool()]  # [valid_len, layer_num, topk]

        if cp_size <= 1:
            # Simple right-padding conversion
            copy_len = min(valid_len, seq_len_per_gpu)
            result[i, :copy_len] = orig_indices[:copy_len]
        else:
            # CP chunk splitting (same pattern as preprocess_bshd)
            chunk_size = seq_len_padded // (cp_size * 2)
            half_seq = seq_len_per_gpu // 2

            # First chunk: chunk[cp_rank]
            first_start = chunk_size * cp_rank
            first_end = min(first_start + chunk_size, valid_len)
            first_len = max(0, first_end - first_start)
            if first_len > 0:
                result[i, :first_len] = orig_indices[first_start:first_end]

            # Second chunk: chunk[-(cp_rank+1)]
            second_start = seq_len_padded - chunk_size * (cp_rank + 1)
            second_end = seq_len_padded - chunk_size * cp_rank
            second_start = max(0, min(second_start, valid_len))
            second_end = max(0, min(second_end, valid_len))
            second_len = max(0, second_end - second_start)
            if second_len > 0:
                result[i, half_seq : half_seq + second_len] = orig_indices[second_start:second_end]

    # === DEBUG OUTPUT ===
    if _debug_rank == 0:
        _non_sentinel = (result < 255).sum().item()
        _out_nonzero = (result > 0).sum().item()
        _first_out = result[0, 0, 0, 0].item() if result.shape[1] > 0 else -1
        _last_out = result[0, -1, 0, 0].item() if result.shape[1] > 0 else -1
        print(
            f"[BSHD DEBUG] OUTPUT | result={result.shape}, non_sent={_non_sentinel}, "
            f"out_nz={_out_nonzero}, first_out={_first_out}, last_out={_last_out}"
        )
    # === DEBUG END ===

    return result.to(device_name)


def set_router_replay_data(
    layers_topk_idx, attention_mask, tf_config, vp_rank=None, use_sequence_packing=True, sequence_parallel=False
):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        use_sequence_packing (bool): If True, use THD (packed sequences) format. If False, use BSHD format.
            Defaults to True for backward compatibility.
        sequence_parallel (bool): Whether sequence parallelism is enabled. Used for BSHD format to apply
            SP scatter. Defaults to False.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    with torch.no_grad():
        if use_sequence_packing:
            # Existing THD logic
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
            layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

            # 1, dynamic_bs_split, layer_num, topk
            layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
                layers_topk_idx_rmpad.to(device_name).squeeze(dim=0)
            ).unsqueeze(dim=0)
        else:
            # BSHD logic: keep all tokens (including padding with sentinel value 255)
            # Unlike THD which packs sequences, BSHD model processes all tokens including padding
            # Router replay patch will handle padding tokens (255) by computing regular topk
            layers_topk_idx_split = _preprocess_router_indices_bshd(
                layers_topk_idx, attention_mask, sequence_parallel
            )

            # BSHD: [batch, seq_per_gpu, layer_num, topk]
            # Router receives [seq, batch, hidden] and does view(-1, num_experts)
            # So we need to flatten in the same order: transpose then flatten
            # [batch, seq_per_gpu, layer_num, topk] -> [seq_per_gpu, batch, layer_num, topk]
            layers_topk_idx_transposed = layers_topk_idx_split.transpose(0, 1)

            # Apply SP scatter (if SP is enabled)
            if sequence_parallel:
                # [seq_per_gpu, batch, layer_num, topk] -> [seq_per_gpu // sp, batch, layer_num, topk]
                layers_topk_idx_transposed = scatter_to_sequence_parallel_region(layers_topk_idx_transposed)

            # Flatten ALL tokens (including padding) - must match router's token count
            seq_per_gpu_sp, batch_size_local = layers_topk_idx_transposed.shape[:2]
            layers_topk_idx_rmpad_split = layers_topk_idx_transposed.reshape(
                1, seq_per_gpu_sp * batch_size_local, -1, layers_topk_idx_split.shape[-1]
            )

            # === DEBUG BSHD ===
            _debug_rank = mpu.get_tensor_model_parallel_rank() if mpu.is_initialized() else 0
            if _debug_rank == 0:
                _non_sentinel_final = (layers_topk_idx_rmpad_split < 255).sum().item()
                _final_nz = (layers_topk_idx_rmpad_split > 0).sum().item()
                _first_final = layers_topk_idx_rmpad_split[0, 0, 0, 0].item()
                _last_final = layers_topk_idx_rmpad_split[0, -1, 0, 0].item()
                print(
                    f"[BSHD DEBUG] FINAL | transpose={layers_topk_idx_split.transpose(0,1).shape}, "
                    f"sp={sequence_parallel}, final={layers_topk_idx_rmpad_split.shape}, "
                    f"non_sent={_non_sentinel_final}, final_nz={_final_nz}, "
                    f"first={_first_final}, last={_last_final}"
                )
            # === DEBUG END ===

        # Common: set indices for each router
        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, _ = local_rank_info["start"], local_rank_info["end"]
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        for i, router in enumerate(router_instances_list):
            router.set_target_indices(layers_topk_idx_reshape[i + offset].to(torch.int64))


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    for chunk_id in range(vpp_size):
        mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)
    return out


def get_current_rank_layer_info(tf_config, vp_rank=None):
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    if vp_rank is None:
        vp_rank = 0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)
    offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def pp_gather(local_layers_router_map, tf_config):
    # TODO: Consider non-uniform layer allocation cases.
    """
    Gather local router maps from all PP ranks into a global router map.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_num_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    if pp_size <= 1:
        return local_layers_router_map

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    local_layers_router_map = local_layers_router_map.to(device_name)
    layers_topk_idx_global_list = [
        torch.empty(
            size=local_layers_router_map.shape,
            dtype=local_layers_router_map.dtype,
            device=local_layers_router_map.device,
        )
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(
        tensor=local_layers_router_map,
        tensor_list=layers_topk_idx_global_list,
        group=pp_group,
        async_op=False,
    )
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vpp_router_map_offset = [[] for _ in range(pp_size)]
        for pp_stage in range(pp_size):
            vpp_router_map_offset[pp_stage].append(0)
            for vp_stage in range(vp_size):
                num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage, pp_stage)
                vpp_router_map_offset[pp_stage].append(num_layers_to_build + vpp_router_map_offset[pp_stage][-1])
        layers_topk_idx_global = []
        for vp_stage in range(vp_size):
            for pp_stage in range(pp_size):
                piece = slice(vpp_router_map_offset[pp_stage][vp_stage], vpp_router_map_offset[pp_stage][vp_stage + 1])
                layers_topk_idx_global.append(layers_topk_idx_global_list[pp_stage][:, :, piece, :])
        global_router_map = torch.cat(layers_topk_idx_global, dim=2).to("cpu")
    else:
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")

    return global_router_map


class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.router_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        if vp_size is not None:
            vp_rank = 0 if vp_rank is None else vp_rank
            offset = 0
            for pre_vp_stage in range(vp_size):
                if pre_vp_stage == vp_rank:
                    break
                num_layers_to_build = get_num_layers_to_build(tf_config, pre_vp_stage)
                offset += num_layers_to_build
        else:
            offset = 0

        num_layers_to_build = get_num_layers_to_build(tf_config, vp_rank)
        router_instances_list = RouterReplay.router_instances[offset : offset + num_layers_to_build]
        return router_instances_list

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.RECORD

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )
