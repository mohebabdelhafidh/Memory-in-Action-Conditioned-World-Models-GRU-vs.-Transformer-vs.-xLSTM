"""Summarize the per-seed AUC sweeps: mean, standard deviation and per-seed values for each core and gap.

Reads results/seeds/<core>_s<seed>_<best|last>.csv (written by run_recipe.py / eval/gap_sweep.py --csv).

    python eval/aggregate_seeds.py
    python eval/aggregate_seeds.py --dir results/seeds --md results/seeds/summary.md
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"^(?P<core>[a-z]+)_s(?P<seed>\d+)_(?P<ckpt>best|last)\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/seeds")
    ap.add_argument("--md", help="also save the tables as markdown to this file")
    a = ap.parse_args()

    data = defaultdict(lambda: defaultdict(dict))  # (ckpt, core) -> seed -> {gap: auc}
    for path in sorted(Path(a.dir).glob("*.csv")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                data[(m["ckpt"], m["core"])][int(m["seed"])][int(r["gap"])] = float(r["auc"])
    if not data:
        raise SystemExit(f"no sweep CSVs found in {a.dir}")

    out = []
    for ckpt in ("best", "last"):
        cores = sorted(c for (k, c) in data if k == ckpt)
        if not cores:
            continue
        gaps = sorted({g for c in cores for s in data[(ckpt, c)].values() for g in s})
        out.append(f"\n### {ckpt}.pt (AUC, mean \u00b1 std over seeds; n = number of seeds)\n")
        out.append("| core | n | " + " | ".join(str(g) for g in gaps) + " |")
        out.append("|---|---|" + "---|" * len(gaps))
        for c in cores:
            seeds = data[(ckpt, c)]
            cells = []
            for g in gaps:
                v = np.array([seeds[s][g] for s in seeds if g in seeds[s]])
                cells.append(f"{v.mean():.2f} \u00b1 {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if len(v) else "-"))
            out.append(f"| {c} | {len(seeds)} | " + " | ".join(cells) + " |")
        out.append("\nPer-seed values:\n")
        for c in cores:
            for s in sorted(data[(ckpt, c)]):
                out.append(f"- {c} seed {s}: " + ", ".join(f"{g}: {data[(ckpt, c)][s].get(g, float('nan')):.2f}" for g in gaps))
    text = "\n".join(out)
    print(text)
    out_note = ("\nEach per-seed value carries its own sampling error of about \u00b10.05 (about 50 correct-goal episodes per gap), "
                "and with 3 seeds the standard deviation is itself uncertain.")
    print(out_note)
    if a.md:
        Path(a.md).write_text(text + "\n" + out_note + "\n")
        print(f"\nsaved {a.md}")


if __name__ == "__main__":
    main()
