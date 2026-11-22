"""Look at one episode: saves a contact-sheet PNG and a GIF to results/figures/.

    python data/preview.py --gap 10 --policy oracle
    python data/preview.py --gap 30 --policy directed --seed 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.cue_recall import COLOR_NAMES, rollout  # noqa: E402


def pick_indices(T, gap, max_frames=14):
    if T <= max_frames:
        return list(range(T))
    idx = {0, 1, 2, gap // 2, gap - 1, gap, gap + 1, gap + 2, gap + 3, T - 3, T - 2, T - 1}
    return sorted(i for i in idx if 0 <= i < T)[:max_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--policy", choices=["random", "directed", "oracle"], default="oracle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--distractor-rate", type=float, default=0.0)
    ap.add_argument("--distractor-size", type=int, default=1)
    a = ap.parse_args()

    ep = rollout(a.gap, a.policy, a.seed, distractor_rate=a.distractor_rate, distractor_size=a.distractor_size)
    obs, T = ep["obs"], len(ep["obs"])
    print(f"cue={COLOR_NAMES[ep['cue']]} gap={a.gap} frames={T} "
          f"reached={'none' if ep['reached'] < 0 else COLOR_NAMES[ep['reached']]} success={ep['success']} "
          f"total_reward={ep['rewards'].sum():.0f}")

    scale, per_row = 3, 7
    idx = pick_indices(T, a.gap)
    size = 64 * scale
    rows = (len(idx) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (size + 4), rows * (size + 18)), (60, 60, 60))
    draw = ImageDraw.Draw(sheet)
    for n, t in enumerate(idx):
        im = Image.fromarray(obs[t]).resize((size, size), Image.NEAREST)
        x, y = (n % per_row) * (size + 4), (n // per_row) * (size + 18)
        sheet.paste(im, (x, y + 14))
        draw.text((x + 2, y + 1), f"t={t}", fill=(255, 255, 255))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cue_recall_gap{a.gap}_{a.policy}_s{a.seed}" + (f"_d{a.distractor_rate}x{a.distractor_size}" if a.distractor_rate else "")
    sheet.save(out / f"{stem}.png")
    gif = [Image.fromarray(f).resize((256, 256), Image.NEAREST) for f in obs]
    gif[0].save(out / f"{stem}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)
    print(f"saved {out / (stem + '.png')} and {out / (stem + '.gif')}")


if __name__ == "__main__":
    main()
