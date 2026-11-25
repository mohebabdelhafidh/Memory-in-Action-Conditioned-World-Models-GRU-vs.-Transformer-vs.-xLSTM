"""Loader for the flat files written by real/convert_droid.py (numpy only; no torch needed).

Conventions (per episode of L frames): obs[t] is frame t; act[t] is the normalized action taken at frame t (change of
the 7 joint angles to frame t+1, plus the next gripper position); the last frame has no action (its row is ignored).
Training uses random windows of up to `seq_len` frames; evaluation uses fixed windows (see build_windows).
"""
from pathlib import Path

import numpy as np


def compute_action_stats(act, starts):
    """Mean and std over valid action rows (the last frame of each episode has no action)."""
    keep = np.ones(len(act), dtype=bool)
    keep[np.asarray(starts[1:]) - 1] = False
    a = act[keep]
    return a.mean(0).astype(np.float32), (a.std(0) + 1e-6).astype(np.float32)


class DroidEpisodes:
    def __init__(self, split, data_dir, seq_len=64, act_stats=None):
        p = Path(data_dir) / f"droid_{split}"
        self.obs = np.load(f"{p}_obs.npy", mmap_mode="r")
        self.act = np.load(f"{p}_act.npy")
        meta = np.load(f"{p}_meta.npz")
        self.starts, self.language = meta["starts"], meta["language"]
        self.n, self.seq_len = len(self.starts) - 1, seq_len
        self.size = tuple(self.obs.shape[1:3])
        self.mean, self.std = act_stats if act_stats is not None else compute_action_stats(self.act, self.starts)

    def length(self, e):
        return int(self.starts[e + 1] - self.starts[e])

    def window(self, e, start, length):
        s = int(self.starts[e]) + start
        t = min(s + length, int(self.starts[e + 1]))
        return np.array(self.obs[s:t]), (self.act[s:t] - self.mean) / self.std

    def sample(self, rng, batch_size):
        """Random windows, padded to the longest in the batch. Returns numpy arrays."""
        wins = []
        for _ in range(batch_size):
            e = int(rng.integers(self.n))
            start = int(rng.integers(0, max(1, self.length(e) - self.seq_len + 1)))
            wins.append(self.window(e, start, self.seq_len))
        return collate(wins)


def collate(wins):
    B, T = len(wins), max(len(o) for o, _ in wins)
    H, W = wins[0][0].shape[1:3]
    obs = np.zeros((B, T, H, W, 3), np.uint8)
    act = np.zeros((B, T, wins[0][1].shape[1]), np.float32)
    mask = np.zeros((B, T), bool)
    for b, (o, a) in enumerate(wins):
        obs[b, :len(o)], act[b, :len(a)], mask[b, :len(o)] = o, a, True
    return dict(obs=obs, act=act, mask=mask)


def build_windows(ds, frames, max_windows):
    """Non-overlapping windows of exactly `frames` frames: [(episode, start), ...], capped at max_windows."""
    wins = [(e, st) for e in range(ds.n) for st in range(0, ds.length(e) - frames + 1, frames)]
    return wins[:max_windows]


def window_batch(ds, wins, frames):
    obs, act = zip(*[ds.window(e, st, frames) for e, st in wins])
    return np.stack(obs), np.stack(act)


def random_blind(batch, steps, rng, p=0.8, max_span=16, min_visible=4):
    """Random blind masks (True = frame withheld): with probability p a sequence gets 1 or 2 blind spans of random
    length 1..max_span, never within the first `min_visible` steps."""
    m = np.zeros((batch, steps), bool)
    if steps <= min_visible + 1:
        return m
    for b in range(batch):
        if rng.random() < p:
            for _ in range(int(rng.integers(1, 3))):
                span = int(rng.integers(1, max_span + 1))
                s = int(rng.integers(min_visible, max(min_visible + 1, steps - span + 1)))
                m[b, s:s + span] = True
    return m
