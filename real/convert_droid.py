"""Convert DROID (RLDS/TFDS) episodes into the flat format our training code reads.

Per episode: keep one camera, subsample in time (--stride), resize (default 64 x 112, which keeps the 16:9 shape),
and build actions from the robot state: action[t] = [change in the 7 joint angles from frame t to t+1, gripper
position at t+1] (8 numbers; the last frame of each episode gets zeros). This avoids relying on the raw `action`
field, whose contents did not match its description in the sample we inspected.

Run it in the TensorFlow environment (.venv-real):
    python real/convert_droid.py --root data_real/droid_100 --out data_real/droid_sample

Files written (flat, like data/collect.py; episode e is [starts[e]:starts[e+1]]):
    <out>/droid_<split>_obs.npy   (N, H, W, 3) uint8
    <out>/droid_<split>_act.npy   (N, 8) float32
    <out>/droid_<split>_meta.npz  starts, language (one instruction per episode)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_droid import decode, find_builder_dir, stack_episode  # noqa: E402

MB = 2 ** 20


def process_episode(ep, camera, stride, size, min_frames=8):
    """Returns (frames uint8 (n,H,W,3), actions float32 (n,8), instruction) or None if too short."""
    T = len(ep["observation/joint_position"])
    idx = np.arange(0, T, stride)
    if len(idx) < min_frames:
        return None
    h, w = size
    frames = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.BOX))
                       for f in ep[f"observation/{camera}"][idx]])
    q = ep["observation/joint_position"][idx].astype(np.float32)
    g = ep["observation/gripper_position"][idx].astype(np.float32)
    acts = np.zeros((len(idx), 8), np.float32)
    acts[:-1, :7] = q[1:] - q[:-1]
    acts[:-1, 7] = g[1:, 0]
    text = ""
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if key in ep and decode(ep[key][0]).strip():
            text = decode(ep[key][0]).strip()
            break
    return frames, acts, text


class Writer:
    """Streams frames to a temp file, then writes the final .npy files (no need to hold everything in RAM)."""

    def __init__(self, prefix, h, w):
        self.prefix, self.h, self.w = str(prefix), h, w
        self.tmp = open(f"{prefix}_obs.tmp", "wb")
        self.acts, self.lengths, self.texts = [], [], []

    def add(self, frames, acts, text):
        self.tmp.write(np.ascontiguousarray(frames).tobytes())
        self.acts.append(acts); self.lengths.append(len(frames)); self.texts.append(text)

    def close(self):
        self.tmp.close()
        tmp = Path(f"{self.prefix}_obs.tmp")
        total = int(sum(self.lengths))
        if total == 0:
            tmp.unlink()
            return 0, 0
        shape = (total, self.h, self.w, 3)
        src = np.memmap(tmp, dtype=np.uint8, mode="r", shape=shape)
        dst = np.lib.format.open_memmap(f"{self.prefix}_obs.npy", mode="w+", dtype=np.uint8, shape=shape)
        for s in range(0, total, 2048):
            dst[s:s + 2048] = src[s:s + 2048]
        dst.flush()
        del src, dst
        tmp.unlink()
        np.save(f"{self.prefix}_act.npy", np.concatenate(self.acts))
        np.savez(f"{self.prefix}_meta.npz", starts=np.concatenate([[0], np.cumsum(self.lengths)]),
                 language=np.array(self.texts))
        return len(self.lengths), total


def diagnose_action_field(ep):
    """Compare the raw `action` field with other fields of the first episode, to see what it really holds."""
    a = ep["action"]
    print("\ncheck of the raw `action` field (max absolute difference over the first episode; ~0 means it matches):")
    for name in ("action_dict/cartesian_position", "observation/cartesian_position", "action_dict/joint_velocity",
                 "action_dict/cartesian_velocity"):
        if name in ep and ep[name].shape[1] >= 6:
            print(f"  action[:, :6] vs {name}[:, :6]: {np.abs(a[:, :6] - ep[name][:, :6]).max():.4f}")
    if "action_dict/gripper_position" in ep:
        print(f"  action[:, 6] vs action_dict/gripper_position: {np.abs(a[:, 6] - ep['action_dict/gripper_position'][:, 0]).max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data_real/droid_sample")
    ap.add_argument("--camera", default="exterior_image_1_left")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--size", type=int, nargs=2, default=[64, 112], metavar=("H", "W"))
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(find_builder_dir(a.root)))
    ds = tfds.as_numpy(builder.as_dataset(split="train"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: Writer(out / f"droid_{s}", *a.size) for s in ("train", "val")}
    rng = np.random.default_rng(a.seed)
    kept = skipped = 0
    for i, episode in enumerate(ds):
        if a.max_episodes and i >= a.max_episodes:
            break
        ep = stack_episode(list(episode["steps"]))
        if i == 0:
            diagnose_action_field(ep)
        r = process_episode(ep, a.camera, a.stride, a.size)
        if r is None:
            skipped += 1
            continue
        writers["val" if rng.random() < a.val_frac else "train"].add(*r)
        kept += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} episodes read")
    for s, w in writers.items():
        n_ep, n_fr = w.close()
        print(f"{s}: {n_ep} episodes, {n_fr} frames")
    print(f"kept {kept}, skipped {skipped} (too short after striding)")
    acts = np.load(out / "droid_train_act.npy")
    np.set_printoptions(precision=3, suppress=True)
    print("train action mean:", acts.mean(0), "\ntrain action std: ", acts.std(0))
    meta = np.load(out / "droid_train_meta.npz")
    print("episode length in frames (train): min %d | mean %.0f | max %d" % (
        np.diff(meta["starts"]).min(), np.diff(meta["starts"]).mean(), np.diff(meta["starts"]).max()))
    print("example instruction:", meta["language"][0])


if __name__ == "__main__":
    main()
