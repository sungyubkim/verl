# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Last verified: 2026-03-29

## Project Overview

verl (Volcano Engine Reinforcement Learning for LLMs)은 ByteDance Seed 팀이 개발한 LLM RL 학습 프레임워크이다.
PPO, GRPO, DAPO 등 30+ 알고리즘을 지원하며, FSDP/Megatron-LM 기반 분산 학습과 vLLM/SGLang 기반 rollout을 제공한다.

**HyperCLOVAX_PostTrainer와의 관계**: `oss/HyperCLOVAX_PostTrainer/`는 이 리포를 fork하여 Naver HyperscaleAI 환경에 맞게 커스터마이징한 것이다. verl의 코드 구조와 개념을 이해하면 PostTrainer의 변경 사항을 파악하는 데 도움이 된다.

## Commands

```bash
# 설치 (extras 선택)
pip install -e .              # Core만
pip install -e ".[gpu,vllm]"  # GPU + vLLM rollout
pip install -e ".[test]"      # 테스트 의존성

# 테스트
pytest tests/ -v
pytest tests/test_protocol_on_cpu.py -v      # CPU 단위 테스트

# Lint
ruff check verl/

# 학습 실행 (Hydra config)
python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
```

## Architecture

### Training Pipeline

```
verl/trainer/main_ppo.py (entrypoint, Hydra config)
  → verl/trainer/ppo/ (training loop)
    → verl/workers/rollout/ (vLLM/SGLang rollout generation)
    → verl/workers/actor/ (policy update)
    → verl/workers/critic/ (value estimation)
    → verl/workers/reward_manager/ (reward computation)
```

### Key Directories

| Directory | Role |
|-----------|------|
| `verl/trainer/` | Training orchestration + Hydra YAML configs |
| `verl/workers/` | Distributed workers (actor, critic, rollout, reward) |
| `verl/models/` | Model backends (HuggingFace Transformers, Megatron-Core) |
| `verl/utils/` | Checkpoint, dataset, vLLM/SGLang integration |
| `verl/experimental/` | Experimental features (agent loop, fully async, VLA) |
| `examples/` | 30+ algorithm examples (PPO, GRPO, DAPO, SFT 등) |
| `tests/` | Unit, e2e, distributed, standalone 테스트 |

### Config System

Hydra 기반. `verl/trainer/config/` 디렉토리에 YAML config 파일이 있으며, CLI에서 dot notation으로 override한다:
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=path/to/data.parquet \
    actor_rollout_ref.model.path=model-name \
    trainer.n_gpus_per_node=8
```

## Git Workflow

이 리포는 **읽기 전용 참조 리포**이다. upstream verl의 코드를 참조/비교하는 용도로만 사용하며, 직접 수정하거나 push하지 않는다.

## Environment

- Python >= 3.10
- Core extras: `[gpu]` (liger-kernel, flash-attn), `[vllm]` (vLLM 0.8.5-0.12.0), `[sglang]` (SGLang 0.5.8), `[mcore]` (Megatron-Core via mbridge)
- Core dependencies: ray>=2.41.0, transformers, hydra-core, tensordict, peft, datasets, wandb
- 테스트: `[test]` extras (pytest, pre-commit, py-spy)
