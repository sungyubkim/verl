#!/bin/bash
set -x

# Example: Training with weighted dataset sampling
# This script demonstrates how to control the mixing ratio of multiple datasets.
# Dataset ratio control allows you to specify the proportion of samples from each
# dataset during training, which is useful for:
# - Balancing datasets of different sizes
# - Emphasizing certain datasets over others
# - Over-sampling smaller datasets to match larger ones

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# Dataset ratio configuration
# Specify the proportion of samples from each dataset
# Order corresponds to the order in train_files
# Example: [0.7, 0.3] means 70% from GSM8K, 30% from MATH
dataset_ratios="[0.7, 0.3]"

# Optional: Specify epoch size (total samples per epoch)
# If null, uses the size of the largest dataset
# If specified, must be a positive integer
# epoch_size=8000

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gpg \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.dataset_ratios="$dataset_ratios" \
    data.dataloader_num_workers=0 \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_gpg_example_gsm8k_math_weighted' \
    trainer.experiment_name='qwen2_7b_weighted_sampling' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

# Notes on Weighted Dataset Sampling:
#
# 1. dataset_ratios must be specified when using multiple datasets
#    - Must be a list of floats that sum to approximately 1.0
#    - Order corresponds to the order of files in train_files
#    - Example: [0.7, 0.3] for 70% first dataset, 30% second dataset
#
# 2. dataloader_num_workers must be 0
#    - Required to ensure consistent sampling ratios
#    - Prevents data caching that could interfere with sampling
#
# 3. Epoch size behavior:
#    - Default: Uses the size of the largest dataset
#    - Custom: Set data.epoch_size=<number> to specify explicitly
#
# 4. Over-sampling and under-sampling:
#    - Small datasets will be over-sampled (with replacement)
#    - Large datasets will be under-sampled (without replacement)
#    - This ensures the specified ratios are maintained
#
# 5. Reproducibility:
#    - Use data.seed=<number> for reproducible sampling
#
# Example scenarios:
#
# Scenario 1: Equal mixing (50-50)
#   dataset_ratios="[0.5, 0.5]"
#
# Scenario 2: Emphasize first dataset (80-20)
#   dataset_ratios="[0.8, 0.2]"
#
# Scenario 3: Three datasets (50-30-20)
#   train_files="['dataset1.parquet', 'dataset2.parquet', 'dataset3.parquet']"
#   dataset_ratios="[0.5, 0.3, 0.2]"
#
# Scenario 4: Custom epoch size
#   dataset_ratios="[0.6, 0.4]"
#   data.epoch_size=10000
