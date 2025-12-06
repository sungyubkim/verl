# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


import torch
from megatron.core import parallel_state as mpu

from verl.utils.megatron_utils import unwrap_model

from .util import (
    postprocess_packed_seqs,
    postprocess_packed_seqs_no_padding,
    preprocess_packed_seqs,
    preprocess_packed_seqs_no_padding,
    recover_left_padding,
    remove_left_padding,
)


def model_forward_gen(vision_model: bool = False, use_sequence_packing: bool = True):
    def model_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        multi_modal_inputs: dict,
        logits_processor=None,
        logits_processor_args: dict = None,
        value_model=False,
    ):
        """Forward pass for models with optional sequence packing.

        Args:
            use_sequence_packing: If True, uses THD format with packed sequences.
                If False, uses standard BSHD format with attention masks.
        """
        pre_process = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]

        if use_sequence_packing:
            # Sequence packing path (THD format)
            fp8 = unwrap_model(model).config.fp8
            use_fp8_padding = fp8 in ["e4m3", "hybrid"]

            input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=pre_process, use_fp8_padding=use_fp8_padding
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()

            input_args = dict(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids if not vision_model else None,
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )

            if vision_model:
                # workaround for supporting sequence packing with context parallelism
                # cp split with sequence packing will make model lose vision token information, so we need to keep
                # the original input_ids and pack them after vision embedding is calculated,
                # cooporate with mbridge
                input_args["input_ids"] = input_ids
                input_args["attention_mask"] = attention_mask

            output_orig = model(**input_args)
            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_packed_seqs(
                    output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
        else:
            # Non-packing path (standard BSHD format with attention masks)
            # This path is for models that require features incompatible with THD format
            # (e.g., learnable softmax in TransformerEngine)
            sequence_parallel = unwrap_model(model).config.sequence_parallel
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            cp_size_init = mpu.get_context_parallel_world_size()

            # DEBUG: CP=2 hang 디버깅
            print(f"[DEBUG model_forward BSHD] "
                  f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}/{pp_size}, "
                  f"CP_RANK={mpu.get_context_parallel_rank()}/{cp_size_init}, "
                  f"pre_process={pre_process}, post_process={post_process}, "
                  f"batch_size={batch_size}, seq_len={seq_len}", flush=True)

            # For PP>1, use fixed sequence length to ensure consistent P2P buffer shapes.
            # Without this, remove_left_padding computes seq_len dynamically per micro-batch,
            # causing P2P send/recv buffer mismatches and deadlocks.
            if pp_size > 1:
                fixed_seq_len = attention_mask.shape[1]  # Use original padded length
            else:
                fixed_seq_len = None  # Dynamic computation (original behavior)

            # Remove left padding and convert to right-padded format
            new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=sequence_parallel,
                pre_process=pre_process,
                fixed_seq_len=fixed_seq_len,
            )

            input_args = dict(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,  # Pass attention mask (not None)
                position_ids=new_position_ids if not vision_model else None,
                packed_seq_params=None,  # No sequence packing
                **model_kwargs,
            )

            # DEBUG: model forward 전
            print(f"[DEBUG model_forward] Before model(), "
                  f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                  f"input_ids.shape={new_input_ids.shape if pre_process else 'N/A'}", flush=True)

            output_orig = model(**input_args)

            # DEBUG: model forward 후
            print(f"[DEBUG model_forward] After model(), "
                  f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                  f"output_orig.shape={output_orig.shape if post_process else 'hidden'}", flush=True)

            if post_process and logits_processor is not None:
                # Detect batch flattening from Megatron-Core PP scheduler.
                # When variable_seq_lengths=False (i.e., use_sequence_packing=False),
                # PP scheduler flattens [batch, seq, hidden] -> [1, batch*seq, hidden].
                # See: Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py L301
                output_batch = output_orig.shape[0]
                output_seq_len = output_orig.shape[1]
                is_batch_flattened = (output_batch == 1 and batch_size > 1)

                # Handle CP: output_layer does CP all-gather internally, so output_orig has full sequence.
                # But attention_mask and logits_processor_args are still CP-divided.
                # We need to all-gather them to match the output shape.
                cp_size = mpu.get_context_parallel_world_size()

                # DEBUG: CP size 확인
                print(f"[DEBUG model_forward] post_process path, "
                      f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                      f"CP_SIZE={cp_size}, is_batch_flattened={is_batch_flattened}", flush=True)

                if cp_size > 1:
                    import torch.distributed
                    cp_group = mpu.get_context_parallel_group()

                    # DEBUG: CP all-gather 전
                    print(f"[DEBUG model_forward] BEFORE CP all-gather, "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                          f"CP_RANK={mpu.get_context_parallel_rank()}, "
                          f"attention_mask.shape={attention_mask.shape}", flush=True)

                    # All-gather attention_mask [batch, seq/cp] -> reconstruct [batch, seq]
                    gathered_masks = [torch.empty_like(attention_mask) for _ in range(cp_size)]
                    torch.distributed.all_gather(gathered_masks, attention_mask, group=cp_group)

                    # DEBUG: CP all-gather 후
                    print(f"[DEBUG model_forward] AFTER CP all-gather (attention_mask), "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                          f"CP_RANK={mpu.get_context_parallel_rank()}", flush=True)

                    # CP chunk reassembly (same logic as _postprocess_bshd)
                    seq_len_per_gpu = attention_mask.shape[1]
                    full_seq_len = seq_len_per_gpu * cp_size
                    half_seq = seq_len_per_gpu // 2
                    chunk_size = full_seq_len // (cp_size * 2)

                    full_attention_mask = torch.zeros(
                        batch_size, full_seq_len, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    for rank in range(cp_size):
                        first_dst = chunk_size * rank
                        full_attention_mask[:, first_dst : first_dst + half_seq] = gathered_masks[rank][:, :half_seq]
                        second_dst = full_seq_len - chunk_size * (rank + 1)
                        full_attention_mask[:, second_dst : second_dst + half_seq] = gathered_masks[rank][:, half_seq:]

                    # DO NOT all-gather logits_processor_args - keep CP-divided to match output_orig
                    # output_orig is CP-divided (seq_per_gpu), so args must also be CP-divided
                    args = {}
                    for k, v in logits_processor_args.items():  # v: [batch, seq_per_gpu]
                        # Convert to right-padded format using CP-divided attention_mask
                        converted = torch.zeros(batch_size, output_seq_len, dtype=v.dtype, device=v.device)
                        for i in range(batch_size):
                            valid_mask = attention_mask[i].bool()  # CP-divided mask
                            valid_len = min(valid_mask.sum().item(), output_seq_len)
                            converted[i, :valid_len] = v[i, valid_mask][:valid_len]

                        # Handle batch flattening
                        if is_batch_flattened:
                            converted = converted.reshape(1, -1)

                        args[k] = converted

                    # DEBUG: logits_processor 전 (CP>1)
                    print(f"[DEBUG model_forward] BEFORE logits_processor (CP>1), "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                          f"output_orig.shape={output_orig.shape}", flush=True)

                    output_dict = logits_processor(output_orig, **args)

                    # DEBUG: logits_processor 후 (CP>1)
                    print(f"[DEBUG model_forward] AFTER logits_processor (CP>1), "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}", flush=True)

                    # Unflatten and recover to original left-padded format
                    if is_batch_flattened:
                        output = {k: v.reshape(batch_size, -1) for k, v in output_dict.items()}
                    else:
                        # Recover to original left-padded format
                        output = {}
                        for k, v in output_dict.items():
                            recovered = torch.zeros(batch_size, seq_len, dtype=v.dtype, device=v.device)
                            for i in range(batch_size):
                                valid_len = full_attention_mask[i].sum().long().item()
                                orig_valid_len = min(valid_len, seq_len)
                                # Left-padded: place valid tokens at the end
                                recovered[i, -orig_valid_len:] = v[i, :orig_valid_len]
                            output[k] = recovered
                else:
                    # CP=1: use original logic
                    args = {}
                    for k, v in logits_processor_args.items():
                        # Use remove_left_padding for 2D tensors by adding/removing a dimension
                        converted, _, _ = remove_left_padding(
                            v.unsqueeze(-1),  # [batch, seq_len] -> [batch, seq_len, 1]
                            attention_mask,
                            position_ids,
                            sequence_parallel=sequence_parallel,
                            pre_process=True,
                            fixed_seq_len=fixed_seq_len,  # Use same fixed_seq_len for consistency
                        )
                        converted = converted.squeeze(-1)  # [batch, new_seq_len, 1] -> [batch, new_seq_len]

                        # Handle batch flattening: [batch, seq] -> [1, batch*seq]
                        if is_batch_flattened:
                            converted = converted.reshape(1, -1)

                        # Defensive padding/truncation for seq_len mismatch
                        if converted.shape[1] < output_seq_len:
                            padding = torch.zeros(
                                converted.shape[0],
                                output_seq_len - converted.shape[1],
                                dtype=converted.dtype,
                                device=converted.device,
                            )
                            converted = torch.cat([converted, padding], dim=1)
                        elif converted.shape[1] > output_seq_len:
                            converted = converted[:, :output_seq_len]

                        args[k] = converted

                    # DEBUG: logits_processor 전 (CP=1)
                    print(f"[DEBUG model_forward] BEFORE logits_processor (CP=1), "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}, "
                          f"output_orig.shape={output_orig.shape}", flush=True)

                    output_dict = logits_processor(output_orig, **args)

                    # DEBUG: logits_processor 후 (CP=1)
                    print(f"[DEBUG model_forward] AFTER logits_processor (CP=1), "
                          f"PP_RANK={mpu.get_pipeline_model_parallel_rank()}", flush=True)

                    # Unflatten results if batch was flattened
                    if is_batch_flattened:
                        # [1, batch*seq] -> [batch, seq]
                        # Note: seq_len is the original sequence length from attention_mask
                        output = {k: v.reshape(batch_size, -1) for k, v in output_dict.items()}
                    else:
                        # Normal path: recover left padding
                        # Defensive padding for new_attention_mask
                        if new_attention_mask.shape[-1] < output_seq_len:
                            padding_shape = list(new_attention_mask.shape)
                            padding_shape[-1] = output_seq_len - new_attention_mask.shape[-1]
                            mask_padding = torch.zeros(
                                padding_shape,
                                dtype=new_attention_mask.dtype,
                                device=new_attention_mask.device,
                            )
                            new_attention_mask_padded = torch.cat([new_attention_mask, mask_padding], dim=-1)
                        else:
                            new_attention_mask_padded = new_attention_mask

                        output = {
                            k: recover_left_padding(v, new_attention_mask_padded, attention_mask, seq_len, post_process=post_process)
                            for k, v in output_dict.items()
                        }
            else:
                output = recover_left_padding(
                    output_orig, new_attention_mask, attention_mask, seq_len, post_process=post_process
                )

        if value_model and post_process:
            output = output[..., 0]
        return output

    return model_forward


def gptmodel_forward_no_padding(
    model,
    input_ids,
    multi_modal_inputs: dict,
    logits_processor=None,
    logits_processor_args: dict = None,
    value_model=False,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    model_kwargs = {}
    if "pixel_values" in multi_modal_inputs:
        model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
    if "image_grid_thw" in multi_modal_inputs:
        model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)

    batch_size = input_ids.shape[0]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs_no_padding(input_ids, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    output_orig = model(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=None,
        packed_seq_params=packed_seq_params,
        **model_kwargs,
    )

    if post_process and logits_processor is not None:
        args = {
            k: preprocess_packed_seqs_no_padding(v, pre_process=True, need_roll=(k == "label"))[0]
            for k, v in logits_processor_args.items()
        }
        output_dict = logits_processor(output_orig, **args)
        output = {
            k: postprocess_packed_seqs_no_padding(
                v, packed_seq_params, input_ids, batch_size, post_process=post_process
            )
            for k, v in output_dict.items()
        }
    else:
        output = postprocess_packed_seqs_no_padding(
            output_orig, packed_seq_params, input_ids, batch_size, post_process=post_process
        )

    if value_model and post_process:
        # output = output[..., 0]
        # while using nested tensor, the advanced indexing operation above will result in an error at backward, i.e.
        # ValueError: NestedTensor _nested_select_backward_default(grad_output: t, self: jt_all, dim: any, index: any)
        # so we use `squeeze` to remove the last dimension
        output = output.squeeze(-1)

    return output
