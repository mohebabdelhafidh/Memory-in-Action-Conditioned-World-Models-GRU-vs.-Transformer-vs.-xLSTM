#!/usr/bin/env python
"""Run the full three-stage recipe (+ AUC sweeps) for several cores and seeds, unattended.

For each (seed, core) it runs, in order, with the same seed in every stage:
    stage 1  configs/<core>_ctrl.yaml      cue visible in every frame
    stage 2  configs/<core>_warm.yaml      --init stage 1 best, gaps 0 and 2
    stage 3  configs/<core>_d06b_v2.yaml   --init stage 2 best, decoy data, selection by pooled AUC
    sweep    gap_sweep on the decoy test set for best.pt and last.pt -> results/seeds/<core>_s<seed>_<ckpt>.csv

Finished stages leave a DONE marker and are skipped when you rerun, so you can stop (Ctrl+C) and continue later.
An interrupted stage restarts from its last checkpoint (--resume). Each stage's output goes to
results/seeds/logs/<run>.log. Everything runs sequentially (one GPU job at a time).

    python run_recipe.py --stages sweep --seeds 0          # seed 0 already trained: just produce its sweep CSVs
    python run_recipe.py --seeds 1 2                        # train + sweep, all three cores, seeds 1 and 2
    python run_recipe.py --seeds 1 --cores gru --dry-run    # print the commands only
    nohup python run_recipe.py --seeds 1 2 > results/seeds/run.log 2>&1 &    # unattended
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

STAGES = [("ctrl", None), ("warm", "ctrl"), ("d06b_v2", "warm")]  # (config suffix, stage it starts from)
REQUIRED_DIRS = ["data_ctrl", "data_short", "data_d06b", "data_d06b_val", "data_d06b_test"]


def run_cmd(cmd, log_path):
    with open(log_path, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def last_val_line(log_path):
    try:
        for line in reversed(Path(log_path).read_text().splitlines()):
            if "[val @" in line:
                return line.strip()
    except FileNotFoundError:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", default=["gru", "transformer", "xlstm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--stages", nargs="+", default=["train", "sweep"], choices=["train", "sweep"])
    ap.add_argument("--test-data", default="data_d06b_test")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    missing = [d for d in REQUIRED_DIRS if not Path(d).is_dir()]
    missing += [f"configs/{c}_{s}.yaml" for c in a.cores for s, _ in STAGES if not Path(f"configs/{c}_{s}.yaml").exists()]
    if missing and not a.dry_run:
        sys.exit("missing required folders/configs: " + ", ".join(missing))
    logs = Path("results/seeds/logs")
    logs.mkdir(parents=True, exist_ok=True)

    t_start, failed = time.time(), []
    for seed in a.seeds:
        for core in a.cores:
            job, ok = f"{core} seed {seed}", True
            if "train" in a.stages:
                for suffix, init in STAGES:
                    run = f"{core}_{suffix}_s{seed}"
                    ckdir = Path("checkpoints") / run
                    if (ckdir / "DONE").exists():
                        print(f"[{job}] {suffix}: already done, skipping")
                        continue
                    cmd = [sys.executable, "train.py", "--config", f"configs/{core}_{suffix}.yaml", "--seed", str(seed)]
                    if init:
                        cmd += ["--init", f"checkpoints/{core}_{init}_s{seed}/best.pt"]
                    if (ckdir / "last.pt").exists():
                        cmd += ["--resume"]
                    print(f"[{job}] {suffix}: starting at {time.strftime('%H:%M:%S')}")
                    if a.dry_run:
                        print("    " + " ".join(cmd))
                        continue
                    t0 = time.time()
                    if run_cmd(cmd, logs / f"{run}.log") != 0:
                        print(f"[{job}] {suffix}: FAILED, see {logs / (run + '.log')}")
                        failed.append(f"{job} / {suffix}")
                        ok = False
                        break
                    ckdir.mkdir(parents=True, exist_ok=True)
                    (ckdir / "DONE").write_text(time.ctime())
                    print(f"[{job}] {suffix}: done in {(time.time() - t0) / 60:.1f} min | {last_val_line(logs / (run + '.log'))}")
            if ok and "sweep" in a.stages:
                for ck in ("best", "last"):
                    out = Path(f"results/seeds/{core}_s{seed}_{ck}.csv")
                    cmd = [sys.executable, "eval/gap_sweep.py", "--ckpt", f"checkpoints/{core}_d06b_v2_s{seed}/{ck}.pt",
                           "--data", a.test_data, "--csv", str(out)]
                    if out.exists():
                        print(f"[{job}] sweep {ck}: already done, skipping")
                        continue
                    print(f"[{job}] sweep {ck}")
                    if a.dry_run:
                        print("    " + " ".join(cmd))
                        continue
                    if run_cmd(cmd, logs / f"{core}_sweep_{ck}_s{seed}.log") != 0:
                        print(f"[{job}] sweep {ck}: FAILED")
                        failed.append(f"{job} / sweep {ck}")
    print(f"\nfinished in {(time.time() - t_start) / 3600:.2f} h" + (f" | FAILED: {failed}" if failed else " | no failures"))


if __name__ == "__main__":
    main()
