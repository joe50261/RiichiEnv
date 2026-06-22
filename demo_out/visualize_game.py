"""Run one full, reproducible hanchan and export the replay visualization.

Outputs (written next to this script):
  - game_log.jsonl   : the raw MJAI event log (replayable with GameViewer.from_jsonl)
  - replay.html      : a self-contained 3D replay viewer (open in any modern browser)

The 3D viewer JS is embedded (gzip + base64) inside replay.html, so the file is
fully standalone -- no internet, server, or extra assets required.

Reproducibility: the wall is fixed via RiichiEnv(seed=...), and players are
iterated in sorted order so the RandomAgent consumes its RNG deterministically
(obs_dict iteration order is otherwise unspecified).
"""

import json
import os

from riichienv import Action, Action3P, RiichiEnv
from riichienv.agents import RandomAgent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 46  # a lively, decisive hanchan: 91 calls (chi/pon) and a winning hand


def run_game() -> RiichiEnv:
    agent = RandomAgent(seed=SEED)
    env = RiichiEnv(game_mode="4p-red-half", seed=SEED)
    obs_dict = env.reset()
    while not env.done():
        actions: dict[int, Action | Action3P] = {pid: agent.act(obs_dict[pid]) for pid in sorted(obs_dict)}
        obs_dict = env.step(actions)
    return env


def main() -> None:
    env = run_game()

    # 1) Persist the raw MJAI log as JSONL (a reusable, replayable artifact).
    log = env.mjai_log
    jsonl_path = os.path.join(OUT_DIR, "game_log.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in log:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # 2) Build the viewer and use its programmatic-inspection API.
    viewer = env.get_viewer()

    print("=== Game summary (per round) ===")
    rounds = viewer.summary()
    for r in rounds:
        print(
            f"  round {r['round_idx']:>2}: {r['bakaze']}{r['kyoku']} "
            f"honba={r['honba']} oya={r['oya']} scores={r['scores']}"
        )

    print("\n=== Win results (WinResult per round) ===")
    any_win = False
    for idx in range(len(rounds)):
        for res in viewer.get_results(idx):
            any_win = True
            yaku_names = [y.name_en for y in res.yaku_list()]
            print(f"  round {idx}: han={res.han} fu={res.fu} yaku={yaku_names}")
    if not any_win:
        print("  (no winning hands -- all rounds were exhaustive draws)")

    print("\n=== Final ===")
    print("  scores:", env.scores())
    print("  ranks :", env.ranks())

    # 3) Export the self-contained interactive 3D replay as a standalone HTML page.
    raw = viewer.show().data  # IPython HTML -> raw html+js string
    inner_html = raw if isinstance(raw, str) else ""
    html_doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>RiichiEnv Replay (seed={SEED}, 4p-red-half)</title>\n"
        "<style>body{margin:0;background:#101216;font-family:sans-serif;color:#eee;}"
        "h1{font-size:16px;padding:10px 14px;margin:0;}"
        ".wrap{max-width:1100px;margin:0 auto;}</style>\n"
        '</head>\n<body>\n<div class="wrap">\n'
        f"<h1>RiichiEnv &mdash; 3D replay (random agents, seed={SEED}, 4p-red-half)</h1>\n"
        + inner_html
        + "\n</div>\n</body>\n</html>\n"
    )
    html_path = os.path.join(OUT_DIR, "replay.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print("\nWrote:")
    print("  ", jsonl_path)
    print("  ", html_path, "(%.0f KB)" % (os.path.getsize(html_path) / 1024))


if __name__ == "__main__":
    main()
