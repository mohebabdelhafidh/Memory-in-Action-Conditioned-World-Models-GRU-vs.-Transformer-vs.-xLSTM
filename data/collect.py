"""Collect cue-recall episodes and write them to disk (flat, memory-mappable arrays).

Examples (run from the project root):
    python data/collect.py --split train --per-gap 5 --out data      # tiny smoke test
    python data/collect.py --split train
    python data/collect.py --split val
    python data/collect.py --split test

Files written per split (all matched by the existing .gitignore: data/*.npy, data/*.npz):
    data/cue_recall_<split>_obs.npy   (total_frames, 64, 64, 3) uint8, all episodes concatenated
    data/cue_recall_<split>_act.npy   (total_frames,) int8     action taken at that frame, -1 on the last
    data/cue_recall_<split>_rew.npy   (total_frames,) float32  reward received on arriving at that frame
    data/cue_recall_<split>_meta.npz  per-episode: starts (E+1 offsets), cue, gap, policy, success, reached

Episode e is obs[starts[e]:starts[e+1]].

Splits are by episode with disjoint seeds. The test split alone contains the long gaps
(100, 200, 500) so they measure extrapolation. Training data uses only cue-blind policies
(random + directed), so actions never reveal the cue.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.cue_recall import rollout  # noqa: E402

SPLITS = {
    "train": dict(gaps=[10, 30, 60], per_gap=700, seed=0),
    "val": dict(gaps=[10, 30, 60], per_gap=70, seed=1),
    "test": dict(gaps=[10, 30, 60, 100, 200, 500], per_gap=100, seed=2),
}
POLICIES = ["random", "directed"]  # ids 0 and 1


def collect(split, gaps, per_gap, seed, directed_frac, out_dir, cue_always=False,
            distractor_rate=0.0, distractor_size=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"cue_recall_{split}"
    tmp = Path(f"{prefix}_obs.tmp")

    jobs = [(g, i) for g in gaps for i in range(per_gap)]
    np.random.default_rng(seed).shuffle(jobs)

    lengths, cues, gap_l, pol_l, succ_l, reached_l = [], [], [], [], [], []
    acts, rews = [], []
    t0 = time.time()
    with open(tmp, "wb") as f:
        for k, (gap, _) in enumerate(jobs):
            ep_seed = seed * 1_000_000 + k
            pol = int(np.random.default_rng([seed, k, 7]).random() < directed_frac)
            ep = rollout(gap, POLICIES[pol], ep_seed, cue_always=cue_always,
                         distractor_rate=distractor_rate, distractor_size=distractor_size)
            f.write(ep["obs"].tobytes())
            acts.append(ep["actions"]); rews.append(ep["rewards"])
            lengths.append(len(ep["obs"])); cues.append(ep["cue"]); gap_l.append(gap)
            pol_l.append(pol); succ_l.append(ep["success"]); reached_l.append(ep["reached"])
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} episodes ({time.time() - t0:.0f}s)")

    lengths = np.array(lengths)
    total = int(lengths.sum())
    shape = (total, 64, 64, 3)
    src = np.memmap(tmp, dtype=np.uint8, mode="r", shape=shape)
    dst = np.lib.format.open_memmap(f"{prefix}_obs.npy", mode="w+", dtype=np.uint8, shape=shape)
    for s in range(0, total, 2048):
        dst[s : s + 2048] = src[s : s + 2048]
    dst.flush()
    del src, dst
    tmp.unlink()

    np.save(f"{prefix}_act.npy", np.concatenate(acts))
    np.save(f"{prefix}_rew.npy", np.concatenate(rews))
    starts = np.concatenate([[0], np.cumsum(lengths)])
    meta = dict(
        starts=starts, cue=np.array(cues), gap=np.array(gap_l), policy=np.array(pol_l),
        success=np.array(succ_l), reached=np.array(reached_l),
    )
    np.savez(f"{prefix}_meta.npz", **meta)
    summarize(split, meta, lengths, total)


def summarize(split, m, lengths, total):
    reached = m["reached"] >= 0
    print(f"\n[{split}] {len(lengths)} episodes, {total} frames, "
          f"{total * 64 * 64 * 3 / 1e9:.2f} GB of uint8 frames")
    print("gap  episodes  mean_len")
    for g in np.unique(m["gap"]):
        sel = m["gap"] == g
        print(f"{g:>4} {sel.sum():>8} {lengths[sel].mean():>9.1f}")
    print("cue balance (should be ~25% each):",
          np.round(np.bincount(m["cue"], minlength=4) / len(m["cue"]), 3))
    for p, name in enumerate(POLICIES):
        sel = m["policy"] == p
        r = sel & reached
        rate = m["success"][r].mean() if r.any() else float("nan")
        print(f"policy {name:<8}: {sel.sum():>5} eps | reached a goal {r.sum() / max(sel.sum(), 1):.2f} | "
              f"correct given reached {rate:.2f}  (leak check: should be ~0.25)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--gaps", type=int, nargs="+", help="override the split's gap list")
    ap.add_argument("--per-gap", type=int, help="episodes per gap value")
    ap.add_argument("--seed", type=int, help="override the split's base seed")
    ap.add_argument("--directed-frac", type=float, default=0.5, help="fraction of directed episodes")
    ap.add_argument("--out", default="data")
    ap.add_argument("--distractor-rate", type=float, default=0.0,
                    help="expected distractor squares per gap frame (0 = original task)")
    ap.add_argument("--distractor-size", type=int, default=1, help="1 = small dots, 2 = cue-sized blocks")
    ap.add_argument("--cue-always", action="store_true",
                    help="positive control: cue visible in every frame (use a separate --out folder)")
    a = ap.parse_args()
    d = SPLITS[a.split]
    collect(a.split, a.gaps or d["gaps"], a.per_gap or d["per_gap"],
            d["seed"] if a.seed is None else a.seed, a.directed_frac, a.out, a.cue_always,
            a.distractor_rate, a.distractor_size)


if __name__ == "__main__":
    main()
