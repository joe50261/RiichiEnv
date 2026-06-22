"""Run zangjiucheng/riichippo weights in RiichiEnv.

The HuggingFace repo `zangjiucheng/riichippo` publishes policy checkpoints
trained with this riichienv-ml pipeline (a fork lives at
zangjiucheng/RiichiEnv-PPO). This script downloads one of those checkpoints
(or loads a local one), auto-detects the network architecture directly from
the checkpoint tensors, and plays full games in RiichiEnv.

Architecture is inferred from the checkpoint itself, so the script does not
depend on a training YAML matching the weights:

* ``actor_head.*``            -> ActorCriticNetwork (PPO checkpoints, e.g.
                                 ``model_50.pth``, ``model_80.pth``, ``v1.pth``)
* ``a_head.*`` + ``v_head.*`` -> Dueling QNetwork (CQL checkpoints,
                                 e.g. ``cql_model.pth``)

Examples::

    # 4p PPO model (2025), one self-play game, print hero decisions
    python scripts/infer_riichippo.py --hf-file 2025-model/model_50.pth --verbose

    # CQL model, 20 games, report hero placement distribution
    python scripts/infer_riichippo.py --hf-file 2025-model/cql_model.pth --games 20

    # hero (seat 0) vs three RandomAgents, quick strength sanity check
    python scripts/infer_riichippo.py --hf-file v1.pth --games 50 --vs-random

    # use a local checkpoint instead of downloading
    python scripts/infer_riichippo.py --model weights/riichippo/v1.pth

Notes:
* Inference runs fine on CPU (``--device cpu``); use ``--device cuda`` if a GPU
  is available.
* On networks where HuggingFace's Xet CDN is blocked, this script sets
  ``HF_HUB_DISABLE_XET=1`` so downloads fall back to the standard resolve URL.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Make the package importable when run from a source checkout (uninstalled).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Fall back to the standard resolve URL when the Xet CDN is unreachable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from riichienv import RiichiEnv
from riichienv_ml.config import GAME_PARAMS
from riichienv_ml.features.feat_v1 import ObservationEncoder
from riichienv_ml.models.actor_critic import ActorCriticNetwork
from riichienv_ml.models.q_network import QNetwork

DEFAULT_REPO = "zangjiucheng/riichippo"
DEFAULT_HF_FILE = "2025-model/model_50.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run zangjiucheng/riichippo weights in RiichiEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", type=str, default=None,
                     help="Local checkpoint path (skips download).")
    src.add_argument("--hf-file", type=str, default=DEFAULT_HF_FILE,
                     help="File within the HuggingFace repo to download.")
    p.add_argument("--hf-repo", type=str, default=DEFAULT_REPO,
                   help="HuggingFace repo id to download from.")
    p.add_argument("--device", type=str, default="cpu", help="Torch device.")
    p.add_argument("--games", type=int, default=1, help="Number of games to play.")
    p.add_argument("--seed", type=int, default=None, help="Torch RNG seed (sampling).")
    p.add_argument("--sample", action="store_true",
                   help="Sample actions from the policy instead of greedy argmax.")
    p.add_argument("--temp", type=float, default=1.0, help="Softmax temperature (with --sample).")
    p.add_argument("--vs-random", action="store_true",
                   help="Hero (seat 0) uses the model; other seats play randomly.")
    p.add_argument("--verbose", action="store_true",
                   help="Print hero (seat 0) decisions step-by-step.")
    p.add_argument("--export-html", type=str, default=None,
                   help="Write the last game's replay as a standalone HTML 牌譜 to this path.")
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> str:
    if args.model:
        if not Path(args.model).is_file():
            raise SystemExit(f"--model not found: {args.model}")
        return args.model
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "huggingface_hub is required to download weights; "
            "`pip install huggingface_hub` or pass --model <local path>."
        ) from exc
    print(f"Downloading {args.hf_repo}/{args.hf_file} ...")
    return hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)


def load_state_dict(path: str, device: str) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def detect_arch(state: dict) -> dict:
    """Infer model type and dimensions from the checkpoint tensors."""
    if "backbone.conv_in.weight" not in state:
        raise SystemExit(
            "This checkpoint has no CNN backbone (e.g. grp_model.pth is a reward "
            "model, not a policy) and cannot drive play."
        )
    conv_channels, in_channels, _ = state["backbone.conv_in.weight"].shape
    fc_dim, flat = state["backbone.fc.weight"].shape
    tile_dim = flat // conv_channels
    num_blocks = len({k.split(".")[2] for k in state if k.startswith("backbone.res_blocks.")})
    aux_dims = state["aux_head.weight"].shape[0] if "aux_head.weight" in state else None

    if "actor_head.weight" in state:
        model_type = "actor_critic"
        num_actions = state["actor_head.weight"].shape[0]
    elif "a_head.weight" in state and "v_head.weight" in state:
        model_type = "qnetwork"
        num_actions = state["a_head.weight"].shape[0]
    else:
        raise SystemExit("Unknown head layout; expected actor_head.* or a_head.*/v_head.*")

    return dict(
        model_type=model_type, in_channels=in_channels, conv_channels=conv_channels,
        num_blocks=num_blocks, fc_dim=fc_dim, tile_dim=tile_dim,
        num_actions=num_actions, aux_dims=aux_dims,
    )


def players_from_arch(arch: dict) -> int:
    """Map tile_dim/num_actions back to player count via GAME_PARAMS."""
    for n, params in GAME_PARAMS.items():
        if params["tile_dim"] == arch["tile_dim"] and params["num_actions"] == arch["num_actions"]:
            return n
    # Fall back on tile_dim alone.
    for n, params in GAME_PARAMS.items():
        if params["tile_dim"] == arch["tile_dim"]:
            return n
    raise SystemExit(f"Cannot map architecture to a player count: {arch}")


def build_model(state: dict, arch: dict, device: str) -> torch.nn.Module:
    kwargs = dict(
        in_channels=arch["in_channels"], num_actions=arch["num_actions"],
        conv_channels=arch["conv_channels"], num_blocks=arch["num_blocks"],
        fc_dim=arch["fc_dim"], tile_dim=arch["tile_dim"], aux_dims=arch["aux_dims"],
    )
    cls = ActorCriticNetwork if arch["model_type"] == "actor_critic" else QNetwork
    model = cls(**kwargs).to(device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  [warn] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  [warn] unexpected keys: {result.unexpected_keys}")
    model.eval()
    return model


class RiichippoAgent:
    """Greedy / sampling policy agent wrapping a riichippo checkpoint."""

    def __init__(self, model, encoder, device, sample=False, temp=1.0):
        self.model = model
        self.encoder = encoder
        self.device = torch.device(device)
        self.sample = sample
        self.temp = temp

    def reset(self):
        pass

    @torch.inference_mode()
    def act(self, obs):
        feat = self.encoder.encode(obs).to(self.device).unsqueeze(0)
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
        mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0)

        output = self.model(feat)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.masked_fill(mask_t == 0, -1e9)

        if self.sample:
            probs = torch.softmax(logits / self.temp, dim=1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(logits.argmax(dim=1).item())

        action = obs.find_action(action_idx)
        if action is None:
            legals = obs.legal_actions()
            if not legals:
                raise ValueError(f"No legal action for action_id={action_idx}")
            action = legals[0]
        return action


def wrap_standalone_html(fragment: str, title: str, subtitle: str = "") -> str:
    """Wrap a GameViewer HTML fragment into a self-contained HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #f5f5f5; }}
  header {{ padding: 12px 18px; background: #1b1b1b; color: #eee; }}
  header h1 {{ font-size: 16px; margin: 0 0 4px; }}
  header p {{ font-size: 12px; margin: 0; color: #aaa; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 12px; }}
</style>
</head>
<body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<div class="wrap">
{fragment}
</div>
</body>
</html>
"""


def export_replay_html(env, path: str, title: str, subtitle: str) -> None:
    fragment = env.get_viewer().show().data
    if not isinstance(fragment, str):
        fragment = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(wrap_standalone_html(fragment, title, subtitle), encoding="utf-8")
    print(f"\nReplay 牌譜 written to: {path}")


def play_game(env, agents, hero=0, verbose=False):
    obs_dict = env.reset()
    steps = 0
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            action = agents[pid].act(obs)
            if verbose and pid == hero:
                events = obs.new_events()
                tail = events[-1] if events else ""
                print(f"  [hero] {tail}  ->  {action.to_mjai()}")
            actions[pid] = action
        obs_dict = env.step(actions)
        steps += 1
    return steps, env.scores(), env.ranks()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    path = resolve_checkpoint(args)
    state = load_state_dict(path, args.device)
    arch = detect_arch(state)
    n_players = players_from_arch(arch)
    game_mode = GAME_PARAMS[n_players]["game_mode"]

    print(f"Checkpoint : {path}")
    print(f"Detected   : {arch['model_type']} | {n_players}p | "
          f"in={arch['in_channels']} conv={arch['conv_channels']} "
          f"blocks={arch['num_blocks']} fc={arch['fc_dim']} "
          f"actions={arch['num_actions']} tile_dim={arch['tile_dim']}")

    model = build_model(state, arch, args.device)
    encoder = ObservationEncoder(tile_dim=arch["tile_dim"])
    hero_agent = RiichippoAgent(model, encoder, args.device, sample=args.sample, temp=args.temp)

    if args.vs_random:
        from riichienv.agents import RandomAgent
        agents = {0: hero_agent, **{pid: RandomAgent() for pid in range(1, n_players)}}
        print(f"Opponents  : RandomAgent x{n_players - 1} (hero = seat 0)")
    else:
        agents = {pid: hero_agent for pid in range(n_players)}
        print(f"Opponents  : self-play (model in all {n_players} seats)")

    print(f"Mode       : {game_mode} | device={args.device} | "
          f"{'sampling@T=%.2f' % args.temp if args.sample else 'greedy'}\n")

    env = RiichiEnv(game_mode=game_mode)
    hero_ranks: Counter[int] = Counter()
    hero_score_total = 0
    last_scores: list[int] = []
    last_ranks: list[int] = []
    for g in range(args.games):
        steps, scores, ranks = play_game(env, agents, hero=0, verbose=args.verbose)
        hero_ranks[ranks[0]] += 1
        hero_score_total += scores[0]
        last_scores, last_ranks = scores, ranks
        print(f"game {g + 1:>3}: steps={steps:>4} scores={scores} ranks={ranks}")

    if args.export_html:
        weight_name = Path(args.model or args.hf_file).name
        opp = f"RandomAgent x{n_players - 1}" if args.vs_random else "self-play"
        title = f"RiichiEnv 牌譜 — riichippo {weight_name}"
        subtitle = (f"{arch['model_type']} | {game_mode} | {opp} | "
                    f"final scores={last_scores} ranks={last_ranks}")
        export_replay_html(env, args.export_html, title, subtitle)

    print("\n=== summary ===")
    print(f"games           : {args.games}")
    avg_rank = sum(r * c for r, c in hero_ranks.items()) / args.games
    print(f"hero avg place  : {avg_rank:.3f}")
    print(f"hero avg score  : {hero_score_total / args.games:,.0f}")
    dist = {r: hero_ranks.get(r, 0) for r in range(1, n_players + 1)}
    print(f"hero place dist : {dist}")


if __name__ == "__main__":
    main()
