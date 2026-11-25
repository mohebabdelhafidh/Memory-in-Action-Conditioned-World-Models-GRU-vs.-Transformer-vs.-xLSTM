"""Inspect a DROID-format (RLDS/TFDS) sample before building anything on it.

Prints the schema, per-step field shapes, episode lengths, action statistics and language instructions, and saves a
contact sheet of frames from every camera-like field. It reads the schema from the dataset, so it does not assume
field names.

Run it in a SEPARATE environment with TensorFlow (the training venv on Python 3.14 cannot install it):
    uv venv --python 3.11 .venv-real && source .venv-real/bin/activate
    uv pip install tensorflow-cpu tensorflow-datasets pillow numpy gsutil
    gsutil -m cp -r gs://gresearch/robotics/droid_100 data_real/
    python real/inspect_droid.py --root data_real/droid_100
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def flatten(d, prefix=""):
    """{'a': {'b': x}} -> {'a/b': x}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        else:
            out[key] = v
    return out


def stack_episode(steps):
    """list of per-step nested dicts -> {key: array stacked over time}"""
    flat = [flatten(s) for s in steps]
    return {k: np.stack([f[k] for f in flat]) for k in flat[0]}


def is_image(arr):
    return arr.dtype == np.uint8 and arr.ndim == 4 and arr.shape[-1] in (1, 3)


def decode(x):
    return x.decode("utf-8", "replace") if isinstance(x, (bytes, np.bytes_)) else str(x)


def find_builder_dir(root):
    for p in Path(root).rglob("dataset_info.json"):
        return p.parent
    sys.exit(f"no dataset_info.json under {root}: is this the folder you downloaded?")


def contact_sheet(ep, path, n=8):
    from PIL import Image, ImageDraw
    keys = [k for k, v in ep.items() if is_image(v)]
    if not keys:
        print("no image-like fields found, skipping the contact sheet")
        return
    T, rows = len(next(iter(ep.values()))), []
    idx = np.linspace(0, T - 1, n).round().astype(int)
    for k in keys:
        frames = [Image.fromarray(ep[k][i][..., :3] if ep[k].shape[-1] != 1 else np.repeat(ep[k][i], 3, -1))
                  for i in idx]
        h = 120
        frames = [f.resize((int(f.width * h / f.height), h)) for f in frames]
        row = Image.new("RGB", (sum(f.width for f in frames), h + 14), (40, 40, 40))
        x = 0
        for f in frames:
            row.paste(f, (x, 14)); x += f.width
        ImageDraw.Draw(row).text((2, 1), k, fill=(255, 255, 255))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"saved {path} (rows: {', '.join(keys)}; columns: {n} frames spread over the first episode)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder you downloaded (searched for dataset_info.json)")
    ap.add_argument("--n", type=int, default=20, help="episodes to read for the statistics")
    ap.add_argument("--out", default="results/figures/droid_sample.png")
    a = ap.parse_args()

    import tensorflow_datasets as tfds  # imported late so the helpers above can be tested without TensorFlow
    bdir = find_builder_dir(a.root)
    builder = tfds.builder_from_directory(str(bdir))
    print(f"dataset at {bdir}")
    print(builder.info.features)
    print("splits:", {k: v.num_examples for k, v in builder.info.splits.items()})

    ds = tfds.as_numpy(builder.as_dataset(split="train"))
    lengths, actions, texts, first = [], [], [], None
    for i, episode in enumerate(ds):
        if i >= a.n:
            break
        ep = stack_episode(list(episode["steps"]))
        if first is None:
            first = ep
            print("\nper-step fields of the first episode (T = %d):" % len(next(iter(ep.values()))))
            for k, v in ep.items():
                print(f"  {k:55s} {str(v.shape):22s} {v.dtype}")
        lengths.append(len(next(iter(ep.values()))))
        if "action" in ep:
            actions.append(ep["action"])
        for k, v in ep.items():
            if "language_instruction" in k and len(texts) < 5 and decode(v[0]).strip():
                texts.append(decode(v[0]))
                break
    lengths = np.array(lengths)
    print(f"\nepisode length over {len(lengths)} episodes: min {lengths.min()} | mean {lengths.mean():.0f} | max {lengths.max()}")
    if actions:
        A = np.concatenate(actions)
        np.set_printoptions(precision=3, suppress=True)
        print(f"action dims: {A.shape[1]} | min {A.min(0)} | max {A.max(0)} | mean {A.mean(0)} | std {A.std(0)}")
    print("example instructions:", texts)
    contact_sheet(first, a.out)


if __name__ == "__main__":
    main()
