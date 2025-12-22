# Recipe: Atropos Integration

> Integration with [Atropos](https://github.com/NousResearch/atropos), the RL environment framework from Nous Research.

## Overview

This recipe enables VeRL to train with data from Atropos environments. Atropos provides a microservice-based architecture where environments generate rollouts, score them, and submit to a central Trajectory API. VeRL pulls batches and trains using GRPO.

```
┌──────────────────────────────────────────────────────────────┐
│                              VeRL                            │
│  ┌─────────────────────┐           ┌─────────────────────┐   │
│  │  Inference (SGLang) │           │      Trainer        │   │
│  │  - generates text   │◀──────────│   - pulls batches   │   │
│  │  - exposes HTTP API │  weights  │   - computes GRPO   │   │
│  └──────────┬──────────┘           └──────────▲──────────┘   │
└─────────────┼──────────────────────────────────┼─────────────┘
              │                                  │
              │ /generate                        │ GET /batch
              ▼                                  │
┌─────────────────────────┐           ┌─────────┴─────────┐
│  Atropos Environment    │           │  Trajectory API   │
│  (GSM8K, etc.)          │──────────▶│  (localhost:8000) │
│  - calls VeRL inference │  POST     │                   │
│  - scores responses     │  /scored  │                   │
└─────────────────────────┘  _data    └───────────────────┘
```

## Quick Start

### 1. Start Services

```bash
python -m recipe.atropos.launch_services \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steps 1000 \
    --gpus 1
```

### 2. Start Environment

From the [Atropos repository](https://github.com/NousResearch/atropos):

```bash
python -m environments.gsm8k_server serve \
    --env.rollout_server_url http://localhost:8000 \
    --openai.base_url http://<verl-inference-url>/v1 \
    --openai.model_name Qwen/Qwen2.5-1.5B-Instruct
```

Training begins when the environment starts submitting data.

## Configuration

### Launcher Options

```bash
python -m recipe.atropos.launch_services --help
```

### YAML Configuration

See `config/trainer.yaml` for full options.

## Data Format

| Atropos Field | VeRL Field | Description |
|---------------|------------|-------------|
| `tokens` | `input_ids` | Full token sequence |
| `masks` | `response_mask` | -100 for prompt, token_id for response |
| `scores` | `token_level_scores` | Reward (placed on last response token) |
| `inference_logprobs` | `rollout_log_probs` | Logprobs from generation |
| `advantages` | `atropos_advantages` | Optional per-token advantages |
| `ref_logprobs` | `ref_log_prob` | Reference model logprobs for KL |

## Files

```
recipe/atropos/
├── ray_trainer.py      # GRPO trainer
├── data_source.py      # Trajectory API client
├── fsdp_workers.py     # Weight sync workers
├── launch_services.py  # Service launcher
├── main.py             # Hydra entry point
├── config/
│   └── trainer.yaml
├── environments/
│   ├── gsm8k.py            # Full GSM8K environment
│   └── gsm8k_upstream.py   # Wrapper for upstream Atropos
└── utils/
    ├── env_adapter.py      # Universal environment adapter
    └── debug.py
```

## Environment Options

### Option 1: Upstream with Adapter (gsm8k_upstream.py)

Uses upstream Atropos environments directly with the VeRL adapter.

```bash
git clone https://github.com/NousResearch/atropos
export PYTHONPATH=$PYTHONPATH:$(pwd)/atropos

python -m recipe.atropos.environments.gsm8k_upstream serve \
    --env.rollout_server_url http://localhost:8000
```

### Option 2: Local Copy (gsm8k.py)

Full environment code included in this repo. Good for customization.

```bash
python -m recipe.atropos.environments.gsm8k serve \
    --env.rollout_server_url http://localhost:8000
```

## References

- [Atropos Repository](https://github.com/NousResearch/atropos)
