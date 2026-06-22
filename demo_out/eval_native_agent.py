"""Evaluate the native-format agent vs RandomAgents (greedy inference).

Compares three heroes against 3 RandomAgents over K hanchan each:
  - trained net  (loads demo_out/native_agent.pth)
  - untrained net (fresh random init, same architecture)
  - random       (RandomAgent baseline; expected ~rank 2.5)
"""

# ruff: noqa: E402  (riichienv-ml is imported via runtime sys.path injection below)
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "riichienv-ml", "src"))

import numpy as np
import torch
from riichienv_ml.features.feat_v1 import ObservationEncoder
from riichienv_ml.models.actor_critic import ActorCriticNetwork

from riichienv import RiichiEnv

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUT_DIR, "native_agent.pth")
ENC = ObservationEncoder(tile_dim=34)
HERO = 0
K = int(os.environ.get("EVAL_GAMES", "80"))


def net_act(net, obs):
    feat = ENC.encode(obs).unsqueeze(0)
    mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).bool()
    with torch.inference_mode():
        logits, _ = net(feat)
    idx = int(logits[0].masked_fill(~mask, -1e9).argmax())
    return obs.find_action(idx)


def run(hero_kind, net=None):
    rng = random.Random(42)
    ranks, scores, firsts = [], [], 0
    for g in range(K):
        env = RiichiEnv(game_mode="4p-red-half", seed=500000 + g)
        od = env.reset()
        while not env.done():
            actions = {}
            for pid in sorted(od):
                obs = od[pid]
                if pid == HERO and hero_kind != "random":
                    actions[pid] = net_act(net, obs)
                else:
                    actions[pid] = rng.choice(obs.legal_actions())
            od = env.step(actions)
        r = env.ranks()[HERO]
        ranks.append(r)
        scores.append(env.scores()[HERO])
        firsts += r == 1
    return sum(ranks) / K, sum(scores) / K, 100 * firsts / K


def main():
    cfg = torch.load(CKPT, map_location="cpu", weights_only=False)
    trained = ActorCriticNetwork(**cfg["model_config"]).eval()
    trained.load_state_dict(cfg["state_dict"])
    torch.manual_seed(123)
    untrained = ActorCriticNetwork(**cfg["model_config"]).eval()

    print(f"Hero vs 3 RandomAgents over {K} hanchan (lower rank = better; first% = win the match):\n")
    print(f"{'hero':14s} {'avg_rank':>9s} {'avg_score':>10s} {'1st_place%':>10s}")
    for name, kind, net in [
        ("trained_net", "net", trained),
        ("untrained_net", "net", untrained),
        ("random", "random", None),
    ]:
        ar, asc, f1 = run(kind, net)
        print(f"{name:14s} {ar:9.3f} {asc:10.0f} {f1:10.1f}")


if __name__ == "__main__":
    main()
