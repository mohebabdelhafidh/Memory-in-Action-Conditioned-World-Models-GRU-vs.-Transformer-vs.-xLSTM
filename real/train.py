"""Train a world model (any core) on the converted real data, with random blind spans.

    python real/train.py --config configs/real/gru.yaml --steps 200       # quick smoke test
    python real/train.py --config configs/real/gru.yaml                   # real run
    python real/train.py --config configs/real/gru.yaml --resume

The loss is pixel MSE on the next frame. At random "blind" steps the model gets the action but not the image, so it
must carry the scene in its belief state. Validation runs the blind-gap evaluation (real/blind_gap.py); the best
checkpoint is the one with the lowest mean blind-step error over the configured spans.
There is no staged recipe here: the signal is dense (every frame), unlike the sparse reward in the simulated task.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real.blind_gap import evaluate_blind, summarize  # noqa: E402
from real.dataset import DroidEpisodes, random_blind  # noqa: E402
from real.encoder import to_float  # noqa: E402
from real.world_model import build_real_model, count_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    tc = cfg["train"]
    if a.steps:
        tc["steps"] = a.steps
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(tc.get("bf16", True)) and device == "cuda"

    run = f"{cfg['name']}_s{seed}"
    ckpt_dir, log_dir = Path("checkpoints") / run, Path("results/logs") / run
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DroidEpisodes("train", cfg["data"]["dir"], seq_len=tc["seq_len"])
    val_ds = DroidEpisodes("val", cfg["data"]["dir"], act_stats=(train_ds.mean, train_ds.std))
    act_stats = (train_ds.mean, train_ds.std)
    print(f"run {run} | device {device} | bf16 {use_bf16} | train episodes {train_ds.n} | val episodes {val_ds.n}")
    cfg["model"]["size"] = list(train_ds.size)
    cfg["model"]["act_dim"] = int(train_ds.act.shape[1])

    model = build_real_model(cfg).to(device)
    print(f"params (M): total {count_params(model):.2f} | core {count_params(model.core):.2f} | "
          f"encoder {count_params(model.encoder):.2f} | decoder {count_params(model.decoder):.2f}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total, warmup = tc["steps"], tc["warmup"]
    lr_lambda = lambda s: (s + 1) / warmup if s < warmup else 0.1 + 0.45 * (
        1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start, best = 0, float("inf")
    if a.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["step"], ck["best"]
        print(f"resumed from step {start}")

    bcfg, ecfg = tc["blind"], tc["eval"]
    log_file = open(log_dir / "log.csv", "a", newline="")
    writer, wrote_header = csv.writer(log_file), (log_dir / "log.csv").stat().st_size > 0
    model.train()
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(start + 1, total + 1):
        b = train_ds.sample(rng, tc["batch_size"])
        obs, act, mask = (torch.from_numpy(b[k]).to(device) for k in ("obs", "act", "mask"))
        blind = torch.from_numpy(random_blind(obs.shape[0], obs.shape[1] - 1, rng, **bcfg)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(obs[:, :-1], act[:, :-1], blind)
        per = ((out["next_frame"].float() - to_float(obs[:, 1:])) ** 2).mean(dim=(2, 3, 4))  # (B, T-1)
        valid = mask[:, 1:].float()
        loss = (per * valid).sum() / valid.sum().clamp(min=1)
        bv = valid * blind.float()
        loss_blind = (per * bv).sum() / bv.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); sched.step()

        if step % tc["log_every"] == 0 or step == 1:
            vram = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
            ms = (time.time() - t0) / (step - start) * 1000
            print(f"step {step:>5} | mse {loss.item():.5f} | blind-step mse {loss_blind.item():.5f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {ms:.0f} ms/step | peak {vram:.1f} GB")

        if step % tc["eval_every"] == 0 or step == total:
            res = evaluate_blind(model, val_ds, ecfg["spans"], ecfg["s0"], device, ecfg["max_windows"], bf16=use_bf16)
            rows = summarize(res)
            if not rows:
                print("  [val] no validation window long enough for the configured spans; skipping")
                continue
            score = float(np.mean([r["blind_mean"] for r in rows]))
            print(f"  [val @ {step}] one-step mse {np.mean([r['one_step'] for r in rows]):.5f} | " + " | ".join(
                f"span {r['span']}: {r['blind_mean']:.5f} (copy {r['copy_mean']:.5f})" for r in rows))
            row = dict(step=step, train_mse=loss.item(), score=score)
            for r in rows:
                row[f"blind_mean_{r['span']}"], row[f"copy_mean_{r['span']}"] = r["blind_mean"], r["copy_mean"]
            if not wrote_header:
                writer.writerow(row.keys()); wrote_header = True
            writer.writerow(row.values()); log_file.flush()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(model=model.state_dict(), cfg=cfg, step=step, act_stats=act_stats)
            if score < best:
                best = score
                torch.save(payload, ckpt_dir / "best.pt")
            torch.save(dict(payload, opt=opt.state_dict(), sched=sched.state_dict(), best=best), ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
