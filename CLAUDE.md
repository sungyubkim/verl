# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

verl (Volcano Engine Reinforcement Learning) is a flexible and efficient RL training library for large language models. It implements the HybridFlow programming model for distributed RLHF training across hundreds of GPUs.

## Common Commands

### Installation

```bash
# Development installation with vLLM backend
pip install -e .[test,vllm]

# Development installation with SGLang backend
pip install -e .[test,sglang]

# Install with GPU dependencies
pip install -e .[gpu]

# Install with math verification support
pip install -e .[math]
```

### Testing

```bash
# Run specific test file
pytest tests/path/to/test_file.py

# Run tests with verbose output
pytest -v tests/

# Run tests matching a pattern
pytest -k "test_pattern"
```

### Linting and Formatting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on staged changes
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

### Running Training

```bash
# PPO training with function-based reward model
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="['path/to/train.parquet']" \
    data.val_files="['path/to/test.parquet']" \
    actor_rollout_ref.model.path="path/to/model" \
    # ... additional config parameters

# GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.rollout.n=5 \
    # ... additional config parameters

# SFT (Supervised Fine-Tuning)
# See examples/sft/ for SFT training scripts
```

### Building Documentation

```bash
# Ensure verl is installed
pip install -e .[test]

# Install documentation dependencies
pip install -r requirements-docs.txt

# Generate HTML docs
cd docs
make clean
make html

# Preview locally
python -m http.server -d _build/html/
```

## Architecture

### Core Components

**1. Hybrid Programming Model**
- verl uses a hybrid controller programming model that separates control flow (Python) from data flow (distributed workers)
- Controllers orchestrate training via Ray, while workers execute computation on GPUs
- This enables flexible RL algorithm implementation with efficient distributed execution

**2. Main Modules**

- `verl.protocol` - Core data exchange protocol (`DataProto` class) for passing data between workers
  - `DataProto` wraps TensorDict (tensor data) and numpy arrays (non-tensor data) with metadata
  - Supports slicing, chunking, concatenation, and distributed operations
  - Handles serialization/deserialization for Ray remote workers

- `verl.trainer` - Training orchestration layer
  - `main_ppo.py` - Main entry point for PPO/GRPO training with Hydra config
  - `ppo/ray_trainer.py` - Ray-based PPO trainer coordinating distributed workers
  - `config/` - Hydra YAML configs for different training scenarios

- `verl.workers` - Distributed worker implementations
  - `fsdp_workers.py` - FSDP/FSDP2 backend workers (ActorRolloutRefWorker, CriticWorker, RewardModelWorker)
  - `megatron_workers.py` - Megatron-LM backend workers for large-scale training
  - `rollout/` - Rollout workers (vLLM, SGLang, HF Transformers)
  - `actor/`, `critic/`, `reward_model/` - Role-specific worker implementations

- `verl.models` - Model architecture implementations
  - `transformers/` - HuggingFace Transformers integration
  - `llama/`, `qwen2/` - Model-specific implementations
  - `mcore/` - Megatron-LM core integration

- `verl.utils` - Shared utilities
  - `dataset/` - Dataset handling (RLHFDataset, DynamicGenDataset)
  - `device.py` - Device management (CUDA, NPU, CPU detection)
  - `torch_functional.py` - PyTorch distributed utilities
  - `fs.py` - Filesystem operations (HDFS, S3, local)

**3. Training Backends**

- **FSDP/FSDP2**: PyTorch FSDP for training, supports offloading and gradient checkpointing
- **Megatron-LM**: Tensor/pipeline parallelism for very large models (e.g., DeepSeek 671B)
- **Rollout engines**: vLLM (default), SGLang (multi-turn/agentic RL), HF Transformers

**4. Worker Roles**

Training uses specialized Ray remote workers:
- `ActorRolloutRefWorker`: Actor model training + rollout generation + reference policy
- `CriticWorker`: Value function training (for GAE-based algorithms)
- `RewardModelWorker`: Model-based reward computation
- Workers communicate via `DataProto` objects passed through Ray

### Configuration System

Uses Hydra for hierarchical configuration:
- Base configs in `verl/trainer/config/`
- Override via CLI: `python -m verl.trainer.main_ppo actor_rollout_ref.model.path=path/to/model`
- Config structure mirrors code architecture (actor_rollout_ref, critic, reward_model, etc.)

### Key Design Patterns

1. **Resource Pool Management**: GPU allocation via `ResourcePoolManager` mapping roles to GPU pools
2. **Ray Remote Workers**: All heavy computation in Ray actors, lightweight driver coordination
3. **Advantage Estimator Abstraction**: Swap between GAE, GRPO, RLOO via config (`algorithm.adv_estimator`)
4. **Modular Reward Functions**: Supports function-based (verifiable), model-based, and hybrid rewards

## Development Workflow

### Adding New Models

For FSDP backend:
1. Add model class to `verl/models/transformers/` or appropriate subdirectory
2. Register in `verl/models/registry.py`
3. Implement weight loading logic in `verl/models/weight_loader_registry.py`

For Megatron backend:
1. Add model config in `verl/models/mcore/`
2. See documentation for Megatron-specific integration

### Adding New RL Algorithms

1. Implement advantage estimator in `verl/trainer/ppo/core_algos.py`
2. Add config option to `algorithm.adv_estimator` choices
3. Update `verl/trainer/ppo/ray_trainer.py` if needed for algorithm-specific logic
4. See `recipe/` directory for examples (DAPO, PRIME, etc.)

### Adding CI Tests

When adding features:
1. Find relevant workflow in `.github/workflows/` (e.g., `gpu_unit_tests.yml`, `vllm.yml`)
2. Add file path patterns to trigger conditions
3. Minimize test workload (use small models, few steps)
4. Ensure tests pass on GitHub Actions runners

## File Organization

```
verl/
├── verl/                    # Main package
│   ├── protocol.py          # DataProto - core data exchange protocol
│   ├── trainer/             # Training orchestration
│   │   ├── main_ppo.py      # PPO/GRPO entry point
│   │   ├── ppo/             # PPO trainer implementation
│   │   └── config/          # Hydra YAML configs
│   ├── workers/             # Distributed worker implementations
│   │   ├── fsdp_workers.py  # FSDP backend
│   │   ├── megatron_workers.py  # Megatron backend
│   │   ├── actor/, critic/, reward_model/  # Role implementations
│   │   └── rollout/         # Rollout engines (vLLM, SGLang)
│   ├── models/              # Model implementations
│   ├── utils/               # Shared utilities
│   └── single_controller/   # Ray worker group abstractions
├── examples/                # Training examples
│   ├── ppo_trainer/         # PPO examples
│   ├── grpo_trainer/        # GRPO examples
│   ├── sft/                 # Supervised fine-tuning
│   ├── data_preprocess/     # Data preparation scripts
│   └── sglang_multiturn/    # Multi-turn RL examples
├── recipe/                  # Advanced RL algorithms
│   ├── dapo/                # DAPO implementation
│   ├── prime/               # PRIME implementation
│   ├── sppo/                # Self-play preference optimization
│   └── ...
├── tests/                   # Test suite
└── docs/                    # Documentation source
```

## Important Notes

### Claude Code Workflow

**When working on code tasks, always get user confirmation before implementing:**
- Present a plan for code changes and wait for user approval
- Do not implement or fix code until the user explicitly confirms they want to proceed
- Use the TodoWrite tool to outline the planned changes
- Only begin implementation after receiving confirmation

**When working with git, always get user confirmation before committing or pushing:**
- Do not create git commits until the user explicitly requests them
- Do not push changes to remote repositories until the user confirms
- Present a summary of changes before committing
- Wait for explicit approval before running `git commit` or `git push` commands

### DataProto Usage

`DataProto` is the fundamental data structure for worker communication:
- Always use `DataProto.from_dict()` or `DataProto.from_single_dict()` to create instances
- Tensor data goes in `batch` (TensorDict), non-tensor in `non_tensor_batch` (dict of numpy arrays)
- Metadata goes in `meta_info` (dict)
- Operations: `chunk()`, `concat()`, `select()`, `slice()`, `repeat()`, etc.

### Testing Philosophy

- GPU tests run on self-hosted runners (see `.github/workflows/gpu_unit_tests.yml`)
- CPU tests for unit testing logic without GPU dependencies
- vLLM/SGLang-specific tests in separate workflows
- Minimize test duration while ensuring correctness

### Rollout Engine Selection

- `actor_rollout_ref.rollout.name=vllm` (default, fastest for single-turn)
- `actor_rollout_ref.rollout.name=sglang` (better for multi-turn, tool calling)
- `actor_rollout_ref.rollout.name=hf` (HuggingFace Transformers, debugging)

### Memory Optimization

- Enable gradient checkpointing: `actor_rollout_ref.model.enable_gradient_checkpointing=True`
- FSDP offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- Adjust GPU memory utilization: `actor_rollout_ref.rollout.gpu_memory_utilization=0.6`
- Use FSDP2 for better memory efficiency: `actor_rollout_ref.actor.strategy=fsdp2`

## Related Resources

- Documentation: https://verl.readthedocs.io/
- Paper: https://arxiv.org/abs/2409.19256v2 (HybridFlow)
- Examples: `examples/` directory for runnable scripts
- Recipes: `recipe/` directory for advanced algorithms
- Contributing: See CONTRIBUTING.md for development guidelines
