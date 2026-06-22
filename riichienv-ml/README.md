# riichienv-ml

Mahjong RL training pipeline for RiichiEnv.

## Setup

```sh
uv sync
uv run python scripts/train_grp.py -c src/riichienv_ml/configs/4p/grp.yml

# CQL+PPO
uv run python scripts/train_cql.py -c src/riichienv_ml/configs/4p/cql.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/ppo.yml

# BC+PPO (requires online teacher, not included in repo)
uv run python scripts/train_bc.py -c src/riichienv_ml/configs/4p/bc_model.yml
uv run python scripts/train_ppo.py -c src/riichienv_ml/configs/4p/bc_ppo.yml
```

## Inference with pretrained weights

Run the published [`zangjiucheng/riichippo`](https://huggingface.co/zangjiucheng/riichippo)
checkpoints (PPO / CQL) directly in RiichiEnv. The script downloads the weights,
auto-detects the architecture from the checkpoint, and plays full games:

```sh
uv pip install huggingface_hub  # only needed to download weights
uv run python scripts/infer_riichippo.py --hf-file 2025-model/model_50.pth --games 1 --verbose
uv run python scripts/infer_riichippo.py --hf-file v1.pth --games 50 --vs-random
```

See [`docs/RIICHIPPO.md`](../docs/RIICHIPPO.md) for the checkpoint table and details.
