"""Cue-recall accuracy vs. gap length for a trained world model.

For every episode that ends on a goal, take the reward logit at the final transition and measure
how well it separates correct from wrong goals (AUC; 0.5 = no memory of the cue, 1.0 = perfect).
Prints one row per gap with a 95% bootstrap confidence interval. Runs the encoder, core and
reward head only (no decoder), in fp32, so even 500-step episodes fit in VRAM.

    python eval/gap_sweep.py --ckpt checkpoints/gru_big_warm_s0/best.pt --data data_test
    python eval/gap_sweep.py --ckpt checkpoints/memoryless_warm_s0/best.pt --data data_test   # negative control
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import EpisodeDataset, collate_episodes  # noqa: E402
from models.world_model import build_world_model  # noqa: E402
from train import auc  # noqa: E402


def bootstrap_ci(scores, labels, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(len(scores), size=len(scores))
        v = auc(scores[i], labels[i])
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (float("nan"),) * 2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=1200, help="frames per forward pass (lower if out of VRAM)")
    ap.add_argument("--csv", help="also write the per-gap table to this CSV file")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_world_model(ck["cfg"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"{a.ckpt} (step {ck.get('step', '?')}) on {a.data}/{a.split}")
    print(f"{'gap':>5} {'reached':>8} {'correct':>8} {'AUC':>6}   95% CI")

    all_s, all_y, rows = [], [], []
    base = EpisodeDataset(a.split, a.data)
    for g in sorted(set(base.meta["gap"].tolist())):
        ds = EpisodeDataset(a.split, a.data, gaps=[g])
        bs = max(1, a.max_frames // (g + 30))
        scores, labels = [], []
        for s in range(0, len(ds), bs):
            batch = collate_episodes([ds[i] for i in range(s, min(s + bs, len(ds)))])
            batch = {k: v.to(dev) for k, v in batch.items()}
            x = model._embed(batch["obs"][:, :-1], batch["act"][:, :-1].clamp(min=0))
            h, _ = model.core(x)
            logit = model.reward_head(h).squeeze(-1).float().cpu()
            for b in range(len(batch["length"])):
                if batch["reached"][b].item() >= 0:
                    scores.append(logit[b, batch["length"][b].item() - 2].item())
                    labels.append(batch["success"][b].item())
        scores, labels = np.array(scores), np.array(labels, dtype=bool)
        lo, hi = bootstrap_ci(scores, labels)
        print(f"{g:>5} {len(scores):>8} {int(labels.sum()):>8} {auc(scores, labels):>6.3f}   [{lo:.2f}, {hi:.2f}]")
        all_s.append(scores); all_y.append(labels)
        rows.append(dict(gap=g, reached=len(scores), correct=int(labels.sum()), auc=auc(scores, labels), lo=lo, hi=hi))
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'saved {a.csv}')


if __name__ == "__main__":
    main()
