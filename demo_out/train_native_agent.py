"""Train and run a RiichiEnv agent using the project's OWN ML format (riichienv-ml).

No external/community weights and no Mortal: this uses riichienv-ml's native
ActorCriticNetwork + feat_v1 ObservationEncoder, trains a checkpoint by self-play
against RandomAgents (REINFORCE with a critic baseline), saves it in a native
.pth, then plays. Inference mirrors riichienv_ml.agents.Agent.act exactly:
encode(obs) -> net -> mask illegal -> argmax -> obs.find_action(idx).
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
from torch.distributions import Categorical

from riichienv import RiichiEnv

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUT_DIR, "native_agent.pth")

# Native model config (same class/format as riichienv-ml; small for CPU training).
MODEL_CFG = dict(in_channels=74, num_actions=82, conv_channels=32, num_blocks=2, fc_dim=128, tile_dim=34)
ENC = ObservationEncoder(tile_dim=34)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def feat_mask(obs):
    feat = ENC.encode(obs)  # (74,34)
    mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).bool()
    return feat, mask


def play_episode(net, hero, env, rng, train):
    """Play one hanchan; hero uses the net, others RandomAgent. Returns hero step records."""
    od = env.reset()
    steps = []
    while not env.done():
        actions = {}
        for pid in sorted(od):
            obs = od[pid]
            if pid == hero:
                feat, mask = feat_mask(obs)
                logits, value = net(feat.unsqueeze(0))
                masked = logits[0].masked_fill(~mask, -1e9)
                if train:
                    dist = Categorical(logits=masked)
                    idx = dist.sample()
                    steps.append((dist.log_prob(idx), value.squeeze(0), dist.entropy()))
                else:
                    idx = masked.argmax()
                actions[pid] = obs.find_action(int(idx))
            else:
                actions[pid] = rng.choice(obs.legal_actions())
        od = env.step(actions)
    return steps


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    net = ActorCriticNetwork(**MODEL_CFG)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    n_games = int(os.environ.get("TRAIN_GAMES", "240"))
    accum = 8
    hero = 0
    ent_coef, value_coef = 0.01, 0.5

    net.train()
    opt.zero_grad()
    recent = []
    for g in range(1, n_games + 1):
        env = RiichiEnv(game_mode="4p-red-half", seed=1000 + g)
        steps = play_episode(net, hero, env, rng, train=True)
        ret = (env.scores()[hero] - 25000) / 15000.0  # dense score-based return
        recent.append(env.ranks()[hero])
        if not steps:
            continue
        logps = torch.stack([s[0] for s in steps])
        vals = torch.stack([s[1] for s in steps])
        ents = torch.stack([s[2] for s in steps])
        adv = ret - vals.detach()
        loss = (-(adv * logps) + value_coef * (vals - ret) ** 2 - ent_coef * ents).mean()
        loss.backward()
        if g % accum == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            opt.zero_grad()
            avg_rank = sum(recent) / len(recent)
            print(f"  game {g:4d}/{n_games}  avg_rank(last {len(recent)})={avg_rank:.3f}", flush=True)
            recent = []

    torch.save({"model_config": MODEL_CFG, "state_dict": net.state_dict()}, CKPT)
    print(f"saved native checkpoint -> {CKPT} ({os.path.getsize(CKPT) // 1024} KB)")


if __name__ == "__main__":
    main()
