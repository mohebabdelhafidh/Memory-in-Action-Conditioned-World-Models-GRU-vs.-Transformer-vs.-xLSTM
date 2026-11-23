"""E3 (first version): planning success vs. gap length, per core, with baselines.

Every agent faces the SAME episodes (same seeds): the cue is shown, the agent wanders for `gap` blank steps
(random actions, optionally with distractors), then the four goals appear and the agent must reach the one that
matches the cue.

  oracle  knows the cue and walks to the matching goal (upper bound; should be 100%)
  blind   walks to a uniformly random goal, cue-blind (what a model with NO memory should score: ~25%)
  <label> a trained world model: watches the episode, imagines the four walk-to-goal plans from its belief state,
          executes the plan with the highest predicted reward (see planning/goal_planner.py)

Episodes in which the agent already stands on a goal when the goals appear are skipped (the outcome would be decided
before any policy could act), so the oracle's success rate is 100%.

    python eval/planning_success.py \
        --ckpt gru=checkpoints/gru_d06b_v2_s0/best.pt transformer=checkpoints/transformer_d06b_v2_s0/best.pt \
               xlstm=checkpoints/xlstm_d06b_v2_s0/best.pt
    python eval/planning_success.py            # baselines only (no GPU or checkpoints needed)
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.visualize import make_context, real_branch  # noqa: E402
from planning.goal_planner import candidate_path, choose_goal, observe  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="label=path pairs")
    ap.add_argument("--gaps", type=int, nargs="+", default=[10, 30, 60, 100, 200, 500])
    ap.add_argument("--n", type=int, default=100, help="episodes per gap")
    ap.add_argument("--distractor-rate", type=float, default=0.6)
    ap.add_argument("--distractor-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=9_000_000, help="base seed (disjoint from train/val/test seeds)")
    ap.add_argument("--out", default="results/planning_success.csv")
    a = ap.parse_args()

    models, device = {}, "cpu"
    if a.ckpt:
        import torch
        from models.world_model import build_world_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for item in a.ckpt:
            label, path = item.split("=", 1)
            ck = torch.load(path, map_location=device)
            m = build_world_model(ck["cfg"]).to(device)
            m.load_state_dict(ck["model"])
            models[label] = m.eval()
            print(f"loaded {label}: {path} (step {ck.get('step', '?')})")

    agents = ["oracle", "blind"] + list(models)
    wins = {g: {k: 0 for k in agents} for g in a.gaps}
    margin = {g: {k: [] for k in models} for g in a.gaps}
    print(f"distractors: rate {a.distractor_rate}, size {a.distractor_size} | {a.n} episodes per gap")

    for gap in a.gaps:
        t0 = time.time()
        for e in range(a.n):
            env, O, A, seed = make_context(gap, a.seed + e, a.distractor_rate, a.distractor_size)
            rng = np.random.default_rng([seed, 4242])
            wins[gap]["oracle"] += int(real_branch(env, candidate_path(env, env.cue))[2])
            wins[gap]["blind"] += int(real_branch(env, candidate_path(env, int(rng.integers(4))))[2])
            for label, model in models.items():
                state = None
                for t in range(len(A)):  # watch the episode up to the last blank frame
                    state = observe(model, O[t], A[t], state, device)
                best, path, scores = choose_goal(model, device, state, env, O[-1])
                wins[gap][label] += int(real_branch(env, path)[2])
                others = [v for c, v in scores.items() if c != env.cue]
                margin[gap][label].append(scores[env.cue] - float(np.mean(others)))
        print(f"  gap {gap} done in {time.time() - t0:.0f}s")

    print(f"\nsuccess rate (reaching the cue-matched goal), {a.n} episodes per gap, 95% Wilson interval")
    print(f"{'gap':>5}" + "".join(f"{k:>24}" for k in agents))
    rows = []
    for g in a.gaps:
        cells = []
        for k in agents:
            lo, hi = wilson(wins[g][k], a.n)
            cells.append(f"{100 * wins[g][k] / a.n:>6.1f}% [{100 * lo:.0f},{100 * hi:.0f}]".rjust(24))
            rows.append(dict(gap=g, agent=k, n=a.n, successes=wins[g][k], rate=wins[g][k] / a.n, lo=lo, hi=hi,
                             mean_margin=float(np.mean(margin[g][k])) if k in models else ""))
        print(f"{g:>5}" + "".join(cells))
    if models:
        print("\nmean logit margin (cue-matched goal minus the mean of the other three); 0 = no preference")
        print(f"{'gap':>5}" + "".join(f"{k:>14}" for k in models))
        for g in a.gaps:
            print(f"{g:>5}" + "".join(f"{np.mean(margin[g][k]):>14.2f}" for k in models))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
