# Visualization demo

Runs one full, reproducible hanchan with `RandomAgent`s and exports the replay
visualization, demonstrating `riichienv.visualizer.GameViewer`.

```bash
uv run python demo_out/visualize_game.py
```

This writes two artifacts (git-ignored, regenerate any time):

- `replay.html` — a **self-contained** interactive 3D replay viewer. The viewer
  JS is embedded (gzip + base64), so the file needs no server, network, or extra
  assets: just open it in any modern browser. Drag to orbit the camera, use the
  timeline at the bottom to step through the hand.
- `game_log.jsonl` — the raw MJAI event log. Reload it with
  `GameViewer.from_jsonl("demo_out/game_log.jsonl").show()`.

## Notes

- The game is fully reproducible: the wall is fixed via `RiichiEnv(seed=...)`,
  and players are iterated in `sorted()` order so the `RandomAgent` consumes its
  RNG deterministically (the `obs_dict` iteration order is otherwise unspecified).
- In a Jupyter notebook you can skip the HTML export and call `env.get_viewer().show()`
  directly to render the viewer inline (see the project README).
