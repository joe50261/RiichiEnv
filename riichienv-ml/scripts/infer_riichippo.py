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
import html as _html
import json
import os
import sys
from collections import Counter, OrderedDict
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
    p.add_argument("--explain", action="store_true",
                   help="Annotate the exported HTML with the model's per-discard policy "
                        "(written under each tile in the discard river, incl. declined kan/riichi).")
    p.add_argument("--explain-topk", type=int, default=6,
                   help="How many ranked actions to keep per decision for the explanation tooltip.")
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


# 4-player (82-action) encoding helpers (see docs/ENCODING.md).
TILES34 = ([f"{n}{s}" for s in "mps" for n in range(1, 10)]
           + ["E", "S", "W", "N", "P", "F", "C"])


def action_kind_4p(i: int) -> str:
    if 0 <= i <= 36:
        return "discard"
    if i == 37:
        return "riichi"
    if 38 <= i <= 40:
        return "chi"
    if i == 41:
        return "pon"
    if 42 <= i <= 78:
        return "kan"
    if i == 79:
        return "agari"
    if i == 80:
        return "ryukyoku"
    return "pass"


def action_label_4p(i: int) -> str:
    kind = action_kind_4p(i)
    if kind == "discard":
        return f"打{TILES34[i]}" if i < 34 else f"discard#{i}"
    if kind == "kan":
        t = i - 42
        return f"槓{TILES34[t]}" if t < 34 else f"kan#{i}"
    return {"riichi": "立直", "chi": "吃", "pon": "碰",
            "agari": "和了", "ryukyoku": "流局", "pass": "見逃/跳過"}[kind]


class RiichippoAgent:
    """Greedy / sampling policy agent wrapping a riichippo checkpoint."""

    def __init__(self, model, encoder, device, sample=False, temp=1.0,
                 explain=False, explain_topk=6):
        self.model = model
        self.encoder = encoder
        self.device = torch.device(device)
        self.sample = sample
        self.temp = temp
        self.explain = explain
        self.explain_topk = explain_topk
        self.last_decision: dict | None = None

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

        self.last_decision = None
        if self.explain:
            self.last_decision = self._build_decision(mask, logits, action_idx, action)
        return action

    def _build_decision(self, mask, logits, chosen_idx, action) -> dict:
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # report at T=1
        legal = [i for i in range(len(mask)) if mask[i]]
        order = sorted(legal, key=lambda i: -probs[i])
        topk = [(int(i), float(probs[i])) for i in order[:self.explain_topk]]
        specials = [(int(i), action_kind_4p(i), float(probs[i]))
                    for i in legal
                    if action_kind_4p(i) in ("riichi", "kan", "agari", "pon", "chi")]
        mjai = action.to_mjai()
        if isinstance(mjai, str):
            mjai = json.loads(mjai)
        return {
            "chosen_id": int(chosen_idx),
            "chosen_kind": action_kind_4p(int(chosen_idx)),
            "chosen_mjai": mjai,
            "conf": float(probs[chosen_idx]),
            "topk": topk,
            "specials": specials,
        }


def wrap_standalone_html(fragment: str, title: str, subtitle: str = "",
                         extra_css: str = "") -> str:
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
{extra_css}
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


COMMENTARY_CSS = """
  .cmt { margin-top: 18px; }
  .cmt h2 { font-size: 15px; margin: 18px 0 8px; }
  .cmt .legend { font-size: 12px; color: #555; margin-bottom: 10px; }
  .cmt .legend b { padding: 1px 5px; border-radius: 3px; color: #fff; }
  .kyoku { background: #fff; border: 1px solid #e3e3e3; border-radius: 8px;
           padding: 10px 12px; margin-bottom: 12px; }
  .kyoku h3 { font-size: 13px; margin: 0 0 8px; color: #333; }
  .river { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 4px;
           margin: 2px 0 10px; }
  .seat { font-size: 12px; color: #666; width: 64px; flex: none; padding-top: 6px; }
  .tile { position: relative; min-width: 30px; text-align: center; font-size: 13px;
          font-weight: 600; padding: 4px 3px 3px; border: 1px solid #cfcfcf;
          border-radius: 4px; background: #fafafa; }
  .tile.m { color: #c0392b; } .tile.p { color: #2471a3; }
  .tile.s { color: #1e8449; } .tile.z { color: #444; }
  .tile.tsumogiri { opacity: 0.5; }
  .tile.reach { background: #fff3cd; border-color: #e0a800; }
  .conf { height: 3px; margin-top: 3px; background: #2ecc71; border-radius: 2px; }
  .badge { display: block; font-size: 9px; font-weight: 700; margin-top: 2px;
           padding: 0 2px; border-radius: 2px; color: #fff; white-space: nowrap; }
  .b-kan { background: #c0392b; } .b-riichi { background: #2471a3; }
  .b-agari { background: #1e8449; }
"""


def _tile_suit_class(pai: str) -> str:
    p = pai.rstrip("r")
    if p.endswith("m"):
        return "m"
    if p.endswith("p"):
        return "p"
    if p.endswith("s"):
        return "s"
    return "z"


def _fmt_topk(topk: list) -> str:
    return " | ".join(f"{action_label_4p(i)} {p * 100:.1f}%" for i, p in topk)


def _discard_tile_html(rec: dict) -> str:
    mjai = rec["chosen_mjai"]
    pai = mjai.get("pai", "?")
    cls = ["tile", _tile_suit_class(pai)]
    if mjai.get("tsumogiri"):
        cls.append("tsumogiri")
    if mjai.get("reach"):
        cls.append("reach")

    # Declined special actions available on this very draw (kan / riichi / tsumo-agari).
    badges = []
    for i, kind, p in rec.get("specials", []):
        if i == rec["chosen_id"]:
            continue
        if kind == "kan":
            badges.append(f'<span class="badge b-kan">{_html.escape(action_label_4p(i))} {p * 100:.0f}%</span>')
        elif kind == "agari":
            badges.append(f'<span class="badge b-agari">自摸 {p * 100:.0f}%</span>')
        elif kind == "riichi" and p >= 0.08:
            badges.append(f'<span class="badge b-riichi">立直 {p * 100:.0f}%</span>')

    tip = f"{_fmt_topk(rec['topk'])}　(信心 {rec['conf'] * 100:.0f}%)"
    conf_bar = f'<div class="conf" style="width:{max(3, round(rec["conf"] * 28))}px"></div>'
    return (f'<div class="{" ".join(cls)}" title="{_html.escape(tip)}">'
            f'{_html.escape(pai)}{conf_bar}{"".join(badges)}</div>')


def build_commentary_html(decisions: list, n_players: int) -> str:
    """Render per-kyoku discard rivers with the model's policy under each tile."""
    if not decisions:
        return ""
    # Group discard decisions by kyoku (preserve order), then by seat.
    kyokus: "OrderedDict[str, dict]" = OrderedDict()
    for rec in decisions:
        if rec.get("chosen_kind") != "discard":
            continue
        kl = rec.get("kyoku") or "?"
        if kl not in kyokus:
            kyokus[kl] = {"meta": rec.get("kyoku_meta"), "rivers": {p: [] for p in range(n_players)}}
        kyokus[kl]["rivers"][rec["pid"]].append(rec)

    parts = ['<div class="cmt">', "<h2>模型解說 — 每張捨牌的策略機率</h2>",
             '<div class="legend">每張牌下方數字條=該手信心;徽章=當下「可做但沒做」的特殊動作'
             '(<b class="b-kan">槓</b> <b class="b-riichi">立直</b> <b class="b-agari">自摸</b>),'
             '滑鼠移到牌上可看完整動作機率排名。</div>']
    for kl, data in kyokus.items():
        meta = data["meta"] or {}
        head = (f'{_html.escape(str(kl))}　親:P{meta.get("oya", "?")}　'
                f'ドラ表示:{_html.escape(str(meta.get("dora_marker", "?")))}')
        parts.append(f'<div class="kyoku"><h3>{head}</h3>')
        for p in range(n_players):
            tiles = "".join(_discard_tile_html(r) for r in data["rivers"][p])
            parts.append(f'<div class="river"><div class="seat">P{p} 捨て</div>{tiles}</div>')
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def export_replay_html(env, path: str, title: str, subtitle: str,
                       commentary_html: str = "", extra_css: str = "") -> None:
    fragment = env.get_viewer().show().data
    if not isinstance(fragment, str):
        fragment = ""
    fragment = fragment + commentary_html
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        wrap_standalone_html(fragment, title, subtitle, extra_css), encoding="utf-8")
    print(f"\nReplay 牌譜 written to: {path}")


def _scan_kyoku(events, kyoku):
    """Update current-kyoku label/meta from a list of MJAI event JSON strings."""
    for s in events or []:
        try:
            ev = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            continue
        if ev.get("type") == "start_kyoku":
            kyoku["label"] = f'{ev.get("bakaze", "?")}{ev.get("kyoku", "?")}-{ev.get("honba", 0)}'
            kyoku["meta"] = {k: ev.get(k) for k in ("bakaze", "kyoku", "honba", "oya", "dora_marker")}


def play_game(env, agents, hero=0, verbose=False, decisions=None):
    obs_dict = env.reset()
    steps = 0
    kyoku = {"label": None, "meta": None}
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            need_events = verbose and pid == hero
            events = obs.new_events() if (decisions is not None or need_events) else None
            if decisions is not None:
                _scan_kyoku(events, kyoku)
            action = agents[pid].act(obs)
            if need_events:
                tail = events[-1] if events else ""
                print(f"  [hero] {tail}  ->  {action.to_mjai()}")
            if decisions is not None:
                dec = getattr(agents[pid], "last_decision", None)
                if dec is not None:
                    rec = dict(dec)
                    rec["pid"] = pid
                    rec["kyoku"] = kyoku["label"]
                    rec["kyoku_meta"] = kyoku["meta"]
                    decisions.append(rec)
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

    explain = args.explain and bool(args.export_html)
    if args.explain and not args.export_html:
        print("[note] --explain only annotates HTML output; add --export-html PATH.")
    if explain and arch["num_actions"] != 82:
        print("[note] --explain action labels assume the 4p (82-action) encoding; disabling.")
        explain = False

    model = build_model(state, arch, args.device)
    encoder = ObservationEncoder(tile_dim=arch["tile_dim"])
    hero_agent = RiichippoAgent(model, encoder, args.device, sample=args.sample, temp=args.temp,
                                explain=explain, explain_topk=args.explain_topk)

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
    last_decisions: list = []
    for g in range(args.games):
        rec = [] if (explain and g == args.games - 1) else None
        steps, scores, ranks = play_game(env, agents, hero=0, verbose=args.verbose, decisions=rec)
        hero_ranks[ranks[0]] += 1
        hero_score_total += scores[0]
        last_scores, last_ranks = scores, ranks
        if rec is not None:
            last_decisions = rec
        print(f"game {g + 1:>3}: steps={steps:>4} scores={scores} ranks={ranks}")

    if args.export_html:
        weight_name = Path(args.model or args.hf_file).name
        opp = f"RandomAgent x{n_players - 1}" if args.vs_random else "self-play"
        title = f"RiichiEnv 牌譜 — riichippo {weight_name}"
        subtitle = (f"{arch['model_type']} | {game_mode} | {opp} | "
                    f"final scores={last_scores} ranks={last_ranks}")
        commentary = build_commentary_html(last_decisions, n_players) if explain else ""
        export_replay_html(env, args.export_html, title, subtitle,
                           commentary_html=commentary,
                           extra_css=COMMENTARY_CSS if explain else "")

    print("\n=== summary ===")
    print(f"games           : {args.games}")
    avg_rank = sum(r * c for r, c in hero_ranks.items()) / args.games
    print(f"hero avg place  : {avg_rank:.3f}")
    print(f"hero avg score  : {hero_score_total / args.games:,.0f}")
    dist = {r: hero_ranks.get(r, 0) for r in range(1, n_players + 1)}
    print(f"hero place dist : {dist}")


if __name__ == "__main__":
    main()
