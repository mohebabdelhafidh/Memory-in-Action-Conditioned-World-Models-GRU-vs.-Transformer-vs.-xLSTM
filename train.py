"""Train a world model (any core) on the cue-recall data.

    python train.py --config configs/gru.yaml --overfit      # 1) sanity: overfit 8 episodes, loss should -> ~0
    python train.py --config configs/gru.yaml                # 2) real run
    python train.py --config configs/memoryless.yaml         # 3) control: must stay at chance on final_auc
    python train.py --config configs/gru.yaml --resume       # continue from checkpoints/<name>_s<seed>/last.pt

Per step t the model sees (frame_t, action_t) and predicts frame_{t+1} (MSE) and reward_{t+1} (weighted BCE).
Validation also reports `final_auc`: at the last transition of episodes that reached a goal, how well the
reward logit separates correct from wrong goals (0.5 = no memory of the cue, 1.0 = perfect), per gap length.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import EpisodeDataset, collate_episodes
from models.encoder import to_float
from models.world_model import build_world_model, count_params


# ---------------------------------------------------------------- helpers
def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def run_model(model, batch):
    """Inputs are frames/actions 0..T-2; outputs at step t predict frame/reward t+1."""
    return model(batch["obs"][:, :-1], batch["act"][:, :-1].clamp(min=0))


def compute_losses(out, batch, tc, pos_weight):
    m = batch["mask"][:, 1:].float()  # a target is valid if that frame really exists
    n = m.sum().clamp(min=1)
    target = to_float(batch["obs"][:, 1:])
    frame_mse = ((out["next_frame"].float() - target) ** 2).mean(dim=(2, 3, 4))
    frame_loss = (frame_mse * m).sum() / n
    logit, rew = out["reward_logit"].float(), batch["rew"][:, 1:]
    bce = F.binary_cross_entropy_with_logits(logit, rew, reduction="none")
    bce_w = F.binary_cross_entropy_with_logits(logit, rew, reduction="none", pos_weight=pos_weight)
    rew_loss = (bce_w * m).sum() / n
    total = tc["frame_weight"] * frame_loss + tc["reward_weight"] * rew_loss
    return dict(total=total, frame=frame_loss, rew_w=rew_loss, rew_bce=(bce * m).sum() / n, n=n)


def auc(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def evaluate(model, loader, tc, device, pos_weight, use_bf16):
    model.eval()
    sums, n_total = dict(total=0.0, frame=0.0, rew_bce=0.0), 0.0
    scores, labels, gaps = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = run_model(model, batch)
        ls = compute_losses(out, batch, tc, pos_weight)
        for k in sums:
            sums[k] += ls[k].item() * ls["n"].item()
        n_total += ls["n"].item()
        logit = out["reward_logit"].float().cpu()
        for b in range(len(batch["length"])):
            if batch["reached"][b].item() >= 0:  # episode ended on a goal
                t = batch["length"][b].item() - 2  # step that produced the final frame
                scores.append(logit[b, t].item())
                labels.append(batch["success"][b].item())
                gaps.append(batch["gap"][b].item())
    model.train()
    res = {k: v / max(n_total, 1) for k, v in sums.items()}
    res["final_auc"] = auc(scores, labels)
    gaps = np.array(gaps)
    for g in sorted(set(gaps.tolist())):
        sel = gaps == g
        res[f"auc_gap{g}"] = auc(np.array(scores)[sel], np.array(labels)[sel])
    return res


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--overfit", action="store_true", help="train on 8 episodes only and skip eval/checkpoints")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="load model weights only from this checkpoint (warm start; fresh optimizer)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        cfg["seed"] = a.seed
    tc = cfg["train"]
    if a.steps:
        tc["steps"] = a.steps
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(tc.get("bf16", True)) and device == "cuda"

    run = f"{cfg['name']}_s{seed}" + ("_overfit" if a.overfit else "")
    ckpt_dir, log_dir = Path("checkpoints") / run, Path("results/logs") / run
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EpisodeDataset(cfg["data"]["train"], cfg["data"]["dir"])
    val_ds = EpisodeDataset(cfg["data"]["val"], cfg["data"].get("val_dir", cfg["data"]["dir"]))  # val_dir: optional separate folder
    bs = tc["batch_size"]
    if a.overfit:
        train_ds, bs = Subset(train_ds, range(8)), 8
        if not a.steps:
            tc["steps"] = 400
    nw = tc.get("num_workers", 2)
    loader_args = dict(collate_fn=collate_episodes, num_workers=nw, pin_memory=device == "cuda",
                       persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=not a.overfit, drop_last=not a.overfit,
                              generator=torch.Generator().manual_seed(seed), **loader_args)
    val_bs = tc.get("val_batch_size", max(1, bs // 2))  # long-gap val episodes are big, so use smaller batches
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, **loader_args)

    model = build_world_model(cfg).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)["model"])
        print(f"warm start: loaded weights from {a.init}")
    print(f"run {run} | device {device} | bf16 {use_bf16}")
    print(f"params (M): total {count_params(model):.2f} | core {count_params(model.core):.2f} | "
          f"encoder {count_params(model.encoder):.2f} | decoder {count_params(model.decoder):.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps, warmup = tc["steps"], tc["warmup"]

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    pos_weight = torch.tensor(float(tc["reward_pos_weight"]), device=device)

    start, best = 0, float("inf")
    if a.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["step"], ck["best"]
        print(f"resumed from step {start}")

    log_file = open(log_dir / "log.csv", "a", newline="")
    writer, wrote_header = csv.writer(log_file), (log_dir / "log.csv").stat().st_size > 0
    it = cycle(train_loader)
    model.train()
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(start + 1, total_steps + 1):
        batch = to_device(next(it), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = run_model(model, batch)
        ls = compute_losses(out, batch, tc, pos_weight)
        opt.zero_grad(set_to_none=True)
        ls["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()

        log_every = 25 if a.overfit else tc["log_every"]
        if step % log_every == 0 or step == 1:
            vram = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            ms = (time.time() - t0) / (step - start) * 1000
            print(f"step {step:>5} | loss {ls['total'].item():.4f} | frame {ls['frame'].item():.5f} | "
                  f"rew_bce {ls['rew_bce'].item():.5f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"{ms:.0f} ms/step | peak {vram:.1f} GB")

        if not a.overfit and (step % tc["eval_every"] == 0 or step == total_steps):
            res = evaluate(model, val_loader, tc, device, pos_weight, use_bf16)
            aucs = " ".join(f"{k[3:]}={v:.2f}" for k, v in res.items() if k.startswith("auc_gap"))
            print(f"  [val @ {step}] loss {res['total']:.4f} | frame {res['frame']:.5f} | "
                  f"rew_bce {res['rew_bce']:.5f} | final_auc {res['final_auc']:.3f} ({aucs})")
            row = dict(step=step, train_loss=ls["total"].item(), **res)
            if not wrote_header:
                writer.writerow(row.keys()); wrote_header = True
            writer.writerow(row.values()); log_file.flush()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # model selection: lowest val loss (default) or highest pooled final_auc (tc["select"] = "auc")
            score = res["total"] if tc.get("select", "loss") == "loss" else -res["final_auc"]
            if not math.isnan(score) and score < best:
                best = score
                torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), ckpt_dir / "best.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=step, best=best, cfg=cfg), ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()