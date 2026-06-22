# Running `zangjiucheng/riichippo` Weights

[`zangjiucheng/riichippo`](https://huggingface.co/zangjiucheng/riichippo) is a
HuggingFace model repo that publishes policy checkpoints trained with this
`riichienv-ml` pipeline (the training fork lives at
[`zangjiucheng/RiichiEnv-PPO`](https://github.com/zangjiucheng/RiichiEnv-PPO)).
This page shows how to load those weights and play full games in RiichiEnv.

## Available checkpoints

All published checkpoints are **4-player** models for the `4p-red-half`
(hanchan) game mode. Architecture is auto-detected from the checkpoint tensors,
so no training YAML is required to run them.

| File                       | Type                 | Architecture                                  |
|----------------------------|----------------------|-----------------------------------------------|
| `2024-model/model_80.pth`  | PPO (ActorCritic)    | in=74, conv=192, blocks=16, fc=512, act=82    |
| `2025-model/model_50.pth`  | PPO (ActorCritic)    | in=74, conv=192, blocks=16, fc=512, act=82    |
| `v1.pth`                   | PPO (ActorCritic)    | in=74, conv=192, blocks=16, fc=512, act=82    |
| `2024-model/cql_model.pth` | CQL (Dueling QNet)   | + `aux_head` (rank prediction, 4 dims)        |
| `2025-model/cql_model.pth` | CQL (Dueling QNet)   | + `aux_head` (rank prediction, 4 dims)        |
| `*/grp_model.pth`          | GRP reward model     | not a policy — cannot drive play              |

These dimensions match the repo's [`configs/4p/ppo.yml`](../riichienv-ml/src/riichienv_ml/configs/4p/ppo.yml)
and [`configs/4p/cql.yml`](../riichienv-ml/src/riichienv_ml/configs/4p/cql.yml).

## Quick start

```sh
# Build the riichienv extension and install runtime deps
uv sync --dev
uv run maturin develop --release
# (huggingface_hub is needed only for downloading weights)
uv pip install huggingface_hub

# 4p PPO model (2025), one self-play game, print hero decisions
uv run python riichienv-ml/scripts/infer_riichippo.py \
    --hf-file 2025-model/model_50.pth --games 1 --verbose

# CQL model, 20 games, report hero placement distribution
uv run python riichienv-ml/scripts/infer_riichippo.py \
    --hf-file 2025-model/cql_model.pth --games 20

# Strength sanity check: trained hero (seat 0) vs three RandomAgents
uv run python riichienv-ml/scripts/infer_riichippo.py \
    --hf-file v1.pth --games 50 --vs-random

# Use a local checkpoint instead of downloading
uv run python riichienv-ml/scripts/infer_riichippo.py --model path/to/model.pth
```

## Export a replay 牌譜 (standalone HTML)

Add `--export-html` to write the last game as a **self-contained** HTML replay
(the 3D viewer JS is embedded, so the file opens in any modern browser with no
server or extra assets):

```sh
# Four riichippo agents play each other; save the replay 牌譜
uv run python riichienv-ml/scripts/infer_riichippo.py \
    --model weights/riichippo/v1.pth --games 1 --export-html riichippo_selfplay_4p.html
```

Open the resulting `.html` file in a browser to step through the hand with the
interactive 3D board (hands, melds, dora, riichi, waits and win results are all
annotated by the viewer).

### Annotating the model's reasoning (`--explain`)

Add `--explain` (together with `--export-html`) to write the model's policy
**under each tile in the discard river**: a confidence bar for the chosen move,
and a badge whenever a special action was *available but declined* on that draw
— <span style="color:#c0392b">槓 (kan)</span>,
<span style="color:#2471a3">立直 (riichi)</span> or
<span style="color:#1e8449">自摸 (tsumo-agari)</span> — with its probability.
Hovering a tile shows the full ranked action probabilities.

```sh
uv run python riichienv-ml/scripts/infer_riichippo.py \
    --model weights/riichippo/v1.pth --games 1 --explain \
    --export-html riichippo_explained_4p.html
```

This makes decisions like "drew the last 9m but did not kakan" auditable: the
9m's discard turn shows `槓9m NN%`, confirming the kan was offered (the legal
mask is correct) and the policy simply ranked discarding higher. `--explain`
assumes the 4-player (82-action) encoding from
[ENCODING.md](ENCODING.md).

The script downloads the requested file from `zangjiucheng/riichippo`, infers
the model type (`actor_head.*` → ActorCriticNetwork, `a_head.*`/`v_head.*` →
dueling QNetwork) and its dimensions, builds the matching network and the
`feat_v1` encoder, then plays the games and prints per-game scores/ranks plus a
hero placement summary.

## Notes

- **CPU is fine.** Inference defaults to `--device cpu`; pass `--device cuda`
  if a GPU is available.
- **Restricted networks.** The script sets `HF_HUB_DISABLE_XET=1` so downloads
  fall back to the standard `resolve/main` URL when HuggingFace's Xet CDN is
  unreachable. To download manually:

  ```sh
  curl -L -o v1.pth https://huggingface.co/zangjiucheng/riichippo/resolve/main/v1.pth
  uv run python riichienv-ml/scripts/infer_riichippo.py --model v1.pth
  ```

- **Action selection** is greedy argmax over the legal-action mask by default;
  add `--sample --temp 1.0` for stochastic play.

## Verified behavior

Running `v1.pth` at seat 0 against three `RandomAgent`s over 24 hanchan
finishes 1st in every game (avg placement 1.000, avg score ≈ 61,700 vs the
25,000 starting stack), and the verbose decision trace shows expected opening
play (discarding isolated honors, calling chi to advance the hand) — confirming
the weights drive a trained policy rather than random moves.
