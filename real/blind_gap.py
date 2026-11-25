"""Blind-gap evaluation: how well does a model predict the scene after its camera is switched off?

For each blind span N: the model sees `s0` real frames, then N steps in which the frame is withheld (it still receives
the actions), and must predict the frames that follow. Horizon h = number of steps since the last frame it saw:
h = 1 is ordinary one-step prediction, h = N + 1 is the first frame after the span. Errors are pixel MSE in [0, 1].
Reference 1: "copy the last frame it saw" (exact pixels, so it is biased in the baseline's favour: the model rebuilds
frames through a small latent bottleneck). Reference 2: the model's own one-step error; the column "vs 1-step"
(blind mean / one-step error) says how much worse it gets when it cannot see. Near 1 = blindness costs nothing.

    python real/blind_gap.py --ckpt gru=checkpoints/real_gru_s0/best.pt transformer=checkpoints/real_transformer_s0/best.pt
    python real/blind_gap.py --ckpt gru=... --spans 8 16 32 64 --split val
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real.dataset import DroidEpisodes, build_windows, window_batch  # noqa: E402
from real.encoder import to_float  # noqa: E402
from real.world_model import build_real_model  # noqa: E402


@torch.no_grad()
def evaluate_blind(model, ds, spans, s0, device, max_windows=200, bs=8, bf16=False):
    """Returns {N: dict(n=windows, model=per-horizon error (N+1,), copy=per-horizon baseline error (N+1,))}."""
    was_training = model.training
    model.eval()
    results = {}
    for N in spans:
        F = s0 + N + 1  # s0 visible frames, N blind steps, one frame after the span
        wins = build_windows(ds, F, max_windows)
        if not wins:
            continue
        em, ec = np.zeros(N + 1), np.zeros(N + 1)
        for i in range(0, len(wins), bs):
            obs, act = window_batch(ds, wins[i:i + bs], F)
            obs, act = torch.from_numpy(obs).to(device), torch.from_numpy(act).to(device)
            blind = torch.zeros(obs.shape[0], F - 1, dtype=torch.bool, device=device)
            blind[:, s0:s0 + N] = True  # inputs s0 .. s0+N-1 are blind
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device == "cuda"):
                out = model(obs[:, :-1], act[:, :-1], blind)
            err = ((out["next_frame"].float() - to_float(obs[:, 1:])) ** 2).mean(dim=(2, 3, 4))  # (B, F-1)
            em += err[:, s0 - 1:s0 + N].sum(0).cpu().numpy()  # positions s0-1 .. s0+N-1 -> horizons 1 .. N+1
            last = to_float(obs[:, s0 - 1])
            ec += ((to_float(obs[:, s0:s0 + N + 1]) - last[:, None]) ** 2).mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
        results[N] = dict(n=len(wins), model=em / len(wins), copy=ec / len(wins))
    model.train(was_training)
    return results


def summarize(results):
    """Per span: windows, one-step error, mean error over the blind horizons, error at the first frame after the span,
    and the same two errors for the copy-last-frame baseline."""
    rows = []
    for N, r in results.items():
        rows.append(dict(span=N, windows=r["n"], one_step=r["model"][0], blind_mean=r["model"][1:].mean(),
                         blind_last=r["model"][-1], copy_mean=r["copy"][1:].mean(), copy_last=r["copy"][-1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--data", default=None, help="folder with droid_<split>_*.npy (default: the one in the checkpoint)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--spans", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--s0", type=int, default=8, help="visible frames before the blind span")
    ap.add_argument("--max-windows", type=int, default=200)
    ap.add_argument("--out", default="results/real_blind_gap.csv")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_rows = []
    for item in a.ckpt:
        label, path = item.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        model = build_real_model(ck["cfg"]).to(device)
        model.load_state_dict(ck["model"])
        ds = DroidEpisodes(a.split, a.data or ck["cfg"]["data"]["dir"], act_stats=tuple(ck["act_stats"]))
        res = evaluate_blind(model, ds, a.spans, a.s0, device, a.max_windows)
        print(f"\n{label} ({path}, step {ck.get('step', '?')}), {a.split} split, {a.s0} visible frames first")
        print(f"{'span':>5}{'windows':>9}{'one-step':>10}{'blind mean':>12}{'copy mean':>11}{'ratio':>7}"
              f"{'blind last':>12}{'copy last':>11}{'ratio':>7}{'vs 1-step':>11}")
        for r in summarize(res):
            print(f"{r['span']:>5}{r['windows']:>9}{r['one_step']:>10.5f}{r['blind_mean']:>12.5f}{r['copy_mean']:>11.5f}"
                  f"{r['blind_mean'] / r['copy_mean']:>7.2f}{r['blind_last']:>12.5f}{r['copy_last']:>11.5f}"
                  f"{r['blind_last'] / r['copy_last']:>7.2f}{r['blind_mean'] / r['one_step']:>11.2f}")
        for N, r in res.items():
            for h in range(N + 1):
                csv_rows.append(dict(label=label, span=N, horizon=h + 1, windows=r["n"],
                                     model_err=r["model"][h], copy_err=r["copy"][h]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
