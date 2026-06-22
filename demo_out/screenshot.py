"""Render still screenshots of the GameViewer 3D replay with headless Chrome.

The viewer is a WebGL scene, so capturing it needs a browser with WebGL. This
script drives "Chrome for Testing" (software WebGL via SwiftShader) through
Playwright and writes PNG stills for chosen (step, perspective) pairs.

Prerequisites (one-time, in this environment):
    uv pip install playwright
    # Chrome for Testing (no apt/snap needed); pick a known-good version:
    #   https://storage.googleapis.com/chrome-for-testing-public/<ver>/linux64/chrome-linux64.zip
    # then install its runtime libs (libnss3, libgbm1, libasound2t64, ...).
    # Point CHROME_BIN at the extracted `chrome` binary.

Usage:
    CHROME_BIN=/opt/cft/chrome-linux64/chrome uv run python demo_out/screenshot.py

Outputs PNGs into demo_out/shots/ (git-ignored). `step` is an index into the
MJAI event list (viewer totalSteps == events.length); `perspective` is the seat
(0-3) the camera looks from, or None for the default (seat 0).
"""

import os
import tempfile

from playwright.sync_api import sync_playwright

from riichienv.visualizer import GameViewer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOTS_DIR = os.path.join(OUT_DIR, "shots")
LOG_PATH = os.path.join(OUT_DIR, "game_log.jsonl")
CHROME_BIN = os.environ.get("CHROME_BIN", "/opt/cft/chrome-linux64/chrome")

# Software-WebGL flags so the scene renders without a GPU.
CHROME_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--hide-scrollbars",
    "--force-color-profile=srgb",
]

# (name, step, perspective, render_wait_ms) -- tuned for the seed=46 game log.
SHOTS = [
    ("A_start", 1, None, 4000),  # E1 deal
    ("B_midgame", 430, None, 4000),  # E3: a Pon call + discard pools
    ("C_win", 1549, None, 5500),  # W2 ron win, score overlay (dealt-in seat view)
    ("D_win_seat2", 1549, 2, 5500),  # same win seen from seat 2 (camera rotation)
]


def _wrap(inner_html: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset=utf-8>"
        "<style>html,body{margin:0;background:#0e1014;}#stage{width:1280px;height:800px;}</style>"
        "</head><body><div id=stage>" + inner_html + "</div></body></html>"
    )


def main() -> None:
    os.makedirs(SHOTS_DIR, exist_ok=True)
    viewer = GameViewer.from_jsonl(LOG_PATH)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_BIN, args=CHROME_ARGS)
        page = browser.new_page(viewport={"width": 1320, "height": 840}, device_scale_factor=2)
        for name, step, perspective, wait_ms in SHOTS:
            raw = viewer.show(step=step, perspective=perspective, freeze=True).data
            inner = raw if isinstance(raw, str) else ""
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir="/tmp") as f:
                f.write(_wrap(inner))
                tmp_path = f.name
            page.goto("file://" + tmp_path)
            page.wait_for_timeout(wait_ms)  # let WebGL render the frozen frame
            out = os.path.join(SHOTS_DIR, f"{name}.png")
            page.locator("#stage").screenshot(path=out)
            os.unlink(tmp_path)
            print(f"{name}: -> {out} ({os.path.getsize(out) // 1024} KB)")
        browser.close()


if __name__ == "__main__":
    main()
