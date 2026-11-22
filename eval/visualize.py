"""Visualize what a trained world model remembers, by imagination.

For one episode (the cue is shown at t=0 and then hidden for `gap` steps):
  1. The model watches the real frames up to the moment the four goals appear (teacher-forced),
     building its belief state h_t.
  2. From there it IMAGINES walking to each of the four goals: it is fed its own predicted
     frames (open loop), with the action sequence we give it, and it never sees the cue again.
  3. For each goal we show the real simulator's outcome (top row) next to the model's imagined
     frames (bottom row) and the model's predicted reward logit at the final step.

If the model remembers the cue, the reward logit should be highest for the goal that matches the
hidden cue, and the imagined final frame should get the white (correct) border for that goal only.
Note: training upweights positive rewards (x20), so logits are shifted up. Compare them across the
four goals; don't read them as probabilities.

    python eval/visualize.py --ckpt checkpoints/gru_big_warm_s0/best.pt --gap 30 --seed 1
    python eval/visualize.py --ckpt checkpoints/gru_big_warm_s0/best.pt --gap 200 --n 100 --save 2
"""
import argparse
import copy
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.cue_recall import COLOR_NAMES, COLORS, MOVES, NUM_ACTIONS, CueRecallEnv  # noqa: E402


# ------------------------------------------------------------ environment side
def make_context(gap, seed, rate=0.0, size=1):
    """Random-walk through the cue frame and the gap until the goals appear. Retries seeds if the
    agent happens to land on a goal at the very first goal frame (episode would end immediately)."""
    while True:
        env = CueRecallEnv(gap=gap, distractor_rate=rate, distractor_size=size)
        rng = np.random.default_rng([seed, 99])
        O, A = [env.reset(seed=seed)], []
        done = False
        for _ in range(gap + 1):
            a = int(rng.integers(NUM_ACTIONS))
            obs, _, done, _ = env.step(a)
            O.append(obs); A.append(a)
        if not done:
            return env, O, A, seed
        seed += 1000


def bfs_path(start, goal, blocked, grid):
    """Shortest action sequence from start to goal that never steps on another goal cell."""
    prev, q = {start: None}, deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for a in range(NUM_ACTIONS):
            nxt = (cur[0] + int(MOVES[a][0]), cur[1] + int(MOVES[a][1]))
            if 0 <= nxt[0] < grid and 0 <= nxt[1] < grid and nxt not in blocked and nxt not in prev:
                prev[nxt] = (cur, a)
                q.append(nxt)
    actions, node = [], goal
    while prev[node] is not None:
        node, a = prev[node][0], prev[node][1]
        actions.append(a)
    return actions[::-1]


def real_branch(env, path):
    env = copy.deepcopy(env)
    frames, reward, info = [env._render()], 0.0, {}
    for a in path:
        obs, reward, _, info = env.step(a)
        frames.append(obs)
    return frames, reward, info["success"]


# ------------------------------------------------------------------ model side
def load_model(ckpt_path, device):
    import torch
    from models.world_model import build_world_model
    ck = torch.load(ckpt_path, map_location=device)
    model = build_world_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def imagine(model, device, O, A, paths):
    """Returns {color: (imagined_frames_uint8, final_reward_logit)}."""
    import torch
    from models.decoder import Decoder

    def frame_t(f):
        return torch.from_numpy(np.ascontiguousarray(f)).unsqueeze(0).to(device)

    out = {}
    with torch.no_grad():
        state = None
        for t in range(len(A)):  # watch the real context: cue, gap, goals appear
            o = model.step(frame_t(O[t]), torch.tensor([A[t]], device=device), state)
            state = o["state"]
        for c, path in paths.items():
            st, frame, imag, logit = copy.deepcopy(state), frame_t(O[-1]), [O[-1]], float("nan")
            for a in path:
                o = model.step(frame, torch.tensor([a], device=device), st)
                st, logit = o["state"], o["reward_logit"].float().item()
                frame = o["next_frame"].float().clamp(0, 1)  # feed its own prediction back in
                imag.append(Decoder.to_uint8(frame)[0].cpu().numpy())
            out[c] = (imag, logit)
    return out


# --------------------------------------------------------------------- drawing
def _font(size=13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def compose(env, O, gap, cue, rows, title):
    """rows: list of dicts(color, real_frames, imag_frames, success, logit)."""
    S, pad, font = 2, 6, _font()
    cell = 64 * S
    width = 5 * (cell + pad) + pad + 46  # right margin for the real/model tags
    header_h, block_h = 150, 22 + 2 * (cell + pad) + 10
    img = Image.new("RGB", (width, header_h + len(rows) * block_h), (45, 45, 45))
    d = ImageDraw.Draw(img)
    d.text((pad, 4), title, fill=(255, 255, 255), font=font)
    d.text((pad, 22), f"hidden cue: {COLOR_NAMES[cue]}  (shown only at t=0, then {gap} blank steps)",
           fill=(255, 255, 255), font=font)
    for i, (t, label) in enumerate([(0, "t=0: cue shown"), (gap, f"t={gap}: last blank"),
                                    (gap + 1, f"t={gap + 1}: goals appear")]):
        x = pad + i * (cell + pad)
        img.paste(Image.fromarray(O[t]).resize((cell, cell), Image.NEAREST), (x, 42))
        d.text((x, 42 + cell + 2), label, fill=(210, 210, 210), font=font)
    for b, r in enumerate(rows):
        y0 = header_h + b * block_h
        d.rectangle([pad, y0 + 3, pad + 14, y0 + 17], fill=tuple(int(v) for v in COLORS[r["color"]]))
        verdict = "CORRECT goal" if r["success"] else "wrong goal"
        d.text((pad + 20, y0 + 2), f"walk to {COLOR_NAMES[r['color']]}: real outcome = {verdict} | "
               f"model reward logit {r['logit']:+.2f}", fill=(255, 255, 255), font=font)
        n = len(r["real_frames"]) - 1
        idx = sorted(set(np.linspace(0, n, 5).round().astype(int).tolist()))
        for row, (frames, tag) in enumerate([(r["real_frames"], "real"), (r["imag_frames"], "model")]):
            y = y0 + 22 + row * (cell + pad)
            for col, k in enumerate(idx):
                x = pad + col * (cell + pad)
                img.paste(Image.fromarray(frames[k]).resize((cell, cell), Image.NEAREST), (x, y))
            d.text((width - 42, y + cell // 2 - 8), tag, fill=(255, 200, 80), font=font)
    return img


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=1, help="number of episodes to run")
    ap.add_argument("--save", type=int, default=1, help="save images for the first K episodes")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--distractor-rate", type=float, default=0.0)
    ap.add_argument("--distractor-size", type=int, default=1)
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, device)
    name = Path(a.ckpt).parent.name
    best_is_correct = 0
    for e in range(a.n):
        env, O, A, seed = make_context(a.gap, a.seed + e, a.distractor_rate, a.distractor_size)
        paths = {c: bfs_path(env.pos, env.goal_cell[c],
                             {cell for k, cell in env.goal_cell.items() if k != c}, env.grid)
                 for c in range(4)}
        imag = imagine(model, device, O, A, paths)
        rows = []
        for c in range(4):
            real_frames, _, success = real_branch(env, paths[c])
            rows.append(dict(color=c, real_frames=real_frames, imag_frames=imag[c][0],
                             success=success, logit=imag[c][1]))
        logits = [r["logit"] for r in rows]
        best_is_correct += int(int(np.argmax(logits)) == env.cue)
        if e < a.save:
            title = f"{name} (step {ck.get('step', '?')}) | gap {a.gap} | seed {seed}"
            img = compose(env, O, a.gap, env.cue, rows, title)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            path = Path(a.out) / f"imagination_{name}_gap{a.gap}_s{seed}.png"
            img.save(path)
            print(f"saved {path}")
            print("  logits by goal color:", {COLOR_NAMES[c]: round(l, 2) for c, l in enumerate(logits)},
                  "| hidden cue:", COLOR_NAMES[env.cue])
    if a.n > 1:
        print(f"\ngap {a.gap}: the goal matching the hidden cue got the highest reward logit in "
              f"{best_is_correct}/{a.n} episodes ({100 * best_is_correct / a.n:.0f}%; chance = 25%)")


if __name__ == "__main__":
    main()
