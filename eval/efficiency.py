"""E4: efficiency of the three dynamics cores vs. context length.

Two modes, both on random inputs with the cores built from configs/{gru,transformer,xlstm}.yaml
(untrained weights: cost does not depend on training):

  step      one-step inference with explicit state, the way planning (CEM-MPC) uses the model.
            The context is first built up to length L, then `core.step` is timed.
            Reports the size of the state, peak allocated VRAM and ms per step.
            Run for several batch sizes: batch = number of rollouts held in parallel
            (a planner with K candidate action sequences keeps K caches).
  parallel  one forward pass over a whole sequence of length L (batch 1), the way training, probes and the
            long-context sweeps use it. Reports peak VRAM and time.

When a core runs out of memory at some context length, that length is recorded as OOM (a result, not a failure).
Only the dynamics core is measured. The encoder/decoder add a constant cost that is identical for all variants.

    python eval/efficiency.py
    python eval/efficiency.py --batches 1 64 256 --lengths 64 256 1024 4096 16384
"""
import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.world_model import build_world_model  # noqa: E402

MB = 2 ** 20


def sync():
    torch.cuda.synchronize()


def peak_mb():
    return torch.cuda.max_memory_allocated() / MB


def state_bytes(s):
    if torch.is_tensor(s):
        return s.numel() * s.element_size()
    if isinstance(s, dict):
        return sum(state_bytes(v) for v in s.values())
    if isinstance(s, (list, tuple)):
        return sum(state_bytes(v) for v in s)
    return 0


def build_core(name, max_len, dev):
    cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
    cfg["core"]["context_length"] = max_len  # only read by the xlstm core
    return build_world_model(cfg).core.to(dev).eval()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def row(core, mode, batch, context, **kw):
    return dict(core=core, mode=mode, batch=batch, context=context, state_mb=kw.get("state_mb", ""),
                peak_mb=kw.get("peak_mb", ""), ms=kw.get("ms", ""), status=kw.get("status", "ok"))


@torch.no_grad()
def bench_step(core, name, B, lengths, n_time):
    rows, x = [], torch.randn(B, core.dim, device="cuda")
    if hasattr(core, "blocks"):  # transformer: the cache grows, so build each context separately (chunked prefill)
        for L in lengths:
            state = None
            cleanup()
            try:
                done = 0
                while done < L:
                    c = min(512, L - done)
                    _, state = core(torch.randn(B, c, core.dim, device="cuda"), state)
                    done += c
                _, state = core.step(x, state)  # warm-up
                cleanup(); torch.cuda.reset_peak_memory_stats()
                sync(); t0 = time.perf_counter()
                for _ in range(n_time):
                    _, state = core.step(x, state)
                sync(); ms = (time.perf_counter() - t0) / n_time * 1000
                rows.append(row(name, "step", B, L, state_mb=state_bytes(state) / MB, peak_mb=peak_mb(), ms=ms))
            except torch.cuda.OutOfMemoryError:
                rows.append(row(name, "step", B, L, status="OOM"))
                state = None
                break
            state = None
    else:  # recurrent cores: one pass, state stays the same size however long the context gets
        state, t = None, 0
        cleanup(); torch.cuda.reset_peak_memory_stats()
        try:
            for L in lengths:
                while t < L:
                    _, state = core.step(x, state); t += 1
                sync(); t0 = time.perf_counter()
                for _ in range(n_time):
                    _, state = core.step(x, state)
                t += n_time
                sync(); ms = (time.perf_counter() - t0) / n_time * 1000
                rows.append(row(name, "step", B, L, state_mb=state_bytes(state) / MB, peak_mb=peak_mb(), ms=ms))
        except torch.cuda.OutOfMemoryError:
            rows.append(row(name, "step", B, L, status="OOM"))
        state = None
    cleanup()
    return rows


@torch.no_grad()
def bench_parallel(core, name, lengths):
    rows = []
    core(torch.randn(1, 64, core.dim, device="cuda"))  # warm-up (kernel compilation etc.)
    for L in lengths:
        cleanup()
        try:
            x = torch.randn(1, L, core.dim, device="cuda")
            sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            core(x)
            sync(); ms = (time.perf_counter() - t0) * 1000
            rows.append(row(name, "parallel", 1, L, peak_mb=peak_mb(), ms=ms))
        except torch.cuda.OutOfMemoryError:
            rows.append(row(name, "parallel", 1, L, status="OOM"))
            break
        except Exception as e:  # e.g. a sequence longer than the core supports
            rows.append(row(name, "parallel", 1, L, status=f"error: {str(e)[:50]}"))
            break
    cleanup()
    return rows


def plot(rows, batch, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the plot (the CSV has everything)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    spec = [("step", batch, "peak_mb", f"step mode, batch {batch}: peak VRAM (MB)"),
            ("step", batch, "ms", f"step mode, batch {batch}: ms per step"),
            ("parallel", 1, "peak_mb", "parallel forward, batch 1: peak VRAM (MB)")]
    for ax, (mode, b, key, title) in zip(axes, spec):
        for core in sorted({r["core"] for r in rows}):
            pts = [r for r in rows if r["core"] == core and r["mode"] == mode and r["batch"] == b]
            ok = [r for r in pts if r["status"] == "ok"]
            line, = ax.plot([r["context"] for r in ok], [r[key] for r in ok], marker="o", label=core)
            for r in pts:
                if r["status"] != "ok":
                    ax.axvline(r["context"], color=line.get_color(), ls=":", alpha=0.6)
                    ax.text(r["context"], ax.get_ylim()[1], f" {core} fails", color=line.get_color(),
                            rotation=90, va="top", fontsize=8)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("context length"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", default=["gru", "transformer", "xlstm"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 256])
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--n-time", type=int, default=20)
    ap.add_argument("--skip-parallel", action="store_true")
    ap.add_argument("--out", default="results/efficiency")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this benchmark needs a CUDA GPU"
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"GPU: {gpu} ({total:.1f} GB) | fp32 | lengths {a.lengths} | batches {a.batches}")

    rows = []
    for name in a.cores:
        core = build_core(name, max(a.lengths), "cuda")
        print(f"\n== {name} core ({sum(p.numel() for p in core.parameters()) / 1e6:.2f}M params) ==")
        for B in a.batches:
            rows += bench_step(core, name, B, a.lengths, a.n_time)
            print(f"  step mode, batch {B}: done")
        if not a.skip_parallel:
            rows += bench_parallel(core, name, a.lengths)
            print("  parallel mode: done")
        del core
        cleanup()

    for mode, batches in [("step", a.batches), ("parallel", [1])]:
        for B in batches:
            print(f"\n--- {mode} mode, batch {B} ---")
            print(f"{'core':<12}{'context':>8}{'state MB':>10}{'peak MB':>10}{'ms':>9}  status")
            for r in rows:
                if r["mode"] == mode and r["batch"] == B:
                    f = lambda v, d=1: f"{v:.{d}f}" if v != "" else "-"
                    print(f"{r['core']:<12}{r['context']:>8}{f(r['state_mb'], 2):>10}{f(r['peak_mb']):>10}"
                          f"{f(r['ms'], 2):>9}  {r['status']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.parent / (out.name + "_" + gpu.replace(" ", "_").replace("/", "") + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {csv_path}")
    plot(rows, max(a.batches), "results/figures/e4_efficiency.png")


if __name__ == "__main__":
    main()
