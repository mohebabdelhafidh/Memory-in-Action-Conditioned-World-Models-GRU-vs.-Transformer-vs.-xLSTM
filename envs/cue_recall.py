"""Cue-recall grid world (Tier 1).

One episode, with gap N:
    obs_0           : a colored CUE block is shown in the centre (the only time it is ever shown)
    obs_1 .. obs_N  : GAP. Blank grid, only the agent is visible. Nothing says which cue it was.
    obs_{N+1} ..    : the four colored GOALS appear in the corners (colors shuffled every episode).
                      The agent must reach the goal whose color matches the cue.
    The episode ends when the agent steps on a goal (or after `max_goal_steps`).
    The final frame gets a 2px border: white = correct goal, magenta = wrong goal.

Why the design looks like this:
  * Goal colors are shuffled per episode, so "go to the top-left corner" cannot replace remembering the cue.
  * The reward (and the border flash) depend on the cue, so forgetting it costs loss. Pure next-frame
    prediction would not force memory, because nothing visible later depends on the cue.
  * Agent movement is visible but irrelevant during the gap (Option A: isolates memory).
  * Optional distractors (distractor_rate > 0): during the gap frames only, Poisson(rate) colored squares per
    frame appear at random cells with random colors (all four cue colors included). size 1 = small dots
    (same size as the agent), size 2 = big blocks the size of the cue. They are not interactive and last one frame.
    The model must keep the t=0 cue while its input keeps changing. rate 0 reproduces the original task exactly.

Data conventions (used by data/collect.py), for an episode with T actions:
    obs     : T+1 frames, uint8, (64, 64, 3)
    actions : T+1 entries; actions[t] is taken when seeing obs[t]; the last entry is -1 (no action)
    rewards : T+1 entries; rewards[t] is received on arriving at obs[t]; rewards[0] = 0
"""
from __future__ import annotations

import numpy as np

COLORS = np.array(
    [[220, 40, 40], [40, 90, 230], [40, 190, 70], [240, 210, 40]], dtype=np.uint8
)  # red, blue, green, yellow
COLOR_NAMES = ["red", "blue", "green", "yellow"]
BG = np.array([25, 25, 25], dtype=np.uint8)
AGENT_COLOR = np.array([255, 255, 255], dtype=np.uint8)
FLASH_OK = np.array([255, 255, 255], dtype=np.uint8)
FLASH_BAD = np.array([220, 60, 220], dtype=np.uint8)
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # 0 up, 1 down, 2 left, 3 right
NUM_ACTIONS = 4
NUM_CUES = 4


class CueRecallEnv:
    def __init__(self, gap: int = 10, grid: int = 8, cell: int = 8, max_goal_steps: int = 24,
                 cue_always: bool = False, distractor_rate: float = 0.0, distractor_size: int = 1):
        self.distractor_rate, self.distractor_size = float(distractor_rate), int(distractor_size)
        self.dots = []
        self.cue_always = cue_always  # positive control: cue stays visible in every frame (no memory needed)
        self.gap = int(gap)
        self.grid = grid
        self.cell = cell
        self.size = grid * cell  # 64 by default
        self.max_goal_steps = max_goal_steps
        g = grid - 1
        self.corners = [(0, 0), (0, g), (g, 0), (g, g)]
        self.cue_lo = grid // 2 - 1  # cue block covers cells cue_lo .. cue_lo+1 in both axes
        self.done = True

    # ------------------------------------------------------------------ api
    @property
    def goals_visible(self) -> bool:
        return self.t > self.gap

    def reset(self, seed: int | None = None, cue: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.cue = int(self.rng.integers(NUM_CUES)) if cue is None else int(cue)
        perm = self.rng.permutation(NUM_CUES)  # perm[k] = color sitting in corner k
        self.goal_cell = {int(perm[k]): self.corners[k] for k in range(4)}  # color -> cell
        while True:  # start away from corners and the cue block
            r, c = (int(v) for v in self.rng.integers(self.grid, size=2))
            in_cue = self.cue_lo <= r < self.cue_lo + 2 and self.cue_lo <= c < self.cue_lo + 2
            if (r, c) not in self.corners and not in_cue:
                break
        self.pos = (r, c)
        self.t = 0
        self.dots = []
        self.done = False
        return self._render()

    def step(self, action: int):
        assert not self.done, "episode finished, call reset()"
        dr, dc = MOVES[int(action)]
        r = int(np.clip(self.pos[0] + dr, 0, self.grid - 1))
        c = int(np.clip(self.pos[1] + dc, 0, self.grid - 1))
        self.pos = (r, c)
        self.t += 1
        self.dots = []  # distractors only on the blank gap frames 1..gap (rng is only touched when rate > 0)
        if self.distractor_rate > 0 and self.t <= self.gap:
            s = self.distractor_size
            for _ in range(int(self.rng.poisson(self.distractor_rate))):
                r0, c0 = (int(v) for v in self.rng.integers(self.grid - s + 1, size=2))
                self.dots.append((r0, c0, int(self.rng.integers(NUM_CUES))))

        reward, flash, reached = 0.0, None, -1
        if self.goals_visible:
            for color, cell in self.goal_cell.items():
                if cell == self.pos:
                    reached = color
            if reached >= 0:
                self.done = True
                reward = float(reached == self.cue)
                flash = FLASH_OK if reached == self.cue else FLASH_BAD
            elif self.t - self.gap >= self.max_goal_steps:
                self.done = True  # timeout, reward 0
        obs = self._render(flash)
        info = dict(
            cue=self.cue, gap=self.gap, t=self.t, reached=reached,
            success=bool(reached == self.cue), goals_visible=self.goals_visible,
        )
        return obs, reward, self.done, info

    # ------------------------------------------------------- helper actions
    def action_toward(self, cell, rng) -> int:
        dr, dc = cell[0] - self.pos[0], cell[1] - self.pos[1]
        options = []
        if dr < 0: options.append(0)
        if dr > 0: options.append(1)
        if dc < 0: options.append(2)
        if dc > 0: options.append(3)
        if not options:
            return int(rng.integers(NUM_ACTIONS))
        return int(options[int(rng.integers(len(options)))])

    # ------------------------------------------------------------ rendering
    def _fill(self, img, r0, c0, r1, c1, color):
        k = self.cell
        img[r0 * k : r1 * k, c0 * k : c1 * k] = color

    def _render(self, flash=None) -> np.ndarray:
        img = np.empty((self.size, self.size, 3), dtype=np.uint8)
        img[:] = BG
        k = self.cell
        if self.t == 0 or self.cue_always:  # cue: only on the very first frame (unless cue_always)
            lo = self.cue_lo
            self._fill(img, lo, lo, lo + 2, lo + 2, COLORS[self.cue])
        if self.goals_visible:
            for color, (r, c) in self.goal_cell.items():
                self._fill(img, r, c, r + 1, c + 1, COLORS[color])
        for r0, c0, col in self.dots:  # distractors, drawn under the agent
            if self.distractor_size == 1:
                img[r0 * k + 2 : (r0 + 1) * k - 2, c0 * k + 2 : (c0 + 1) * k - 2] = COLORS[col]
            else:
                self._fill(img, r0, c0, r0 + self.distractor_size, c0 + self.distractor_size, COLORS[col])
        r, c = self.pos  # agent: white square inset by 2px
        img[r * k + 2 : (r + 1) * k - 2, c * k + 2 : (c + 1) * k - 2] = AGENT_COLOR
        if flash is not None:
            img[:2], img[-2:], img[:, :2], img[:, -2:] = flash, flash, flash, flash
        return img


# ---------------------------------------------------------------- policies
def make_policy(name: str, env: CueRecallEnv, rng: np.random.Generator, eps: float = 0.1):
    """Returns a function act() -> int, to call right after reset / step.

    random   : uniform random actions.
    directed : random walk during the gap; once goals are visible, walks toward a goal chosen
               UNIFORMLY AT RANDOM (cue-blind, so its behaviour never reveals the cue).
    oracle   : knows the cue; goes to the right goal. Only for demos and as an upper bound
               in planning evaluation. Never use it to generate training data (it leaks the cue
               through the actions).
    """
    rand = lambda: int(rng.integers(NUM_ACTIONS))
    if name == "random":
        return rand
    if name == "directed":
        target = int(rng.integers(NUM_CUES))  # independent of env.cue

        def act():
            if not env.goals_visible or rng.random() < eps:
                return rand()
            return env.action_toward(env.goal_cell[target], rng)

        return act
    if name == "oracle":

        def act():
            if not env.goals_visible:
                return rand()
            return env.action_toward(env.goal_cell[env.cue], rng)

        return act
    raise ValueError(f"unknown policy {name!r}")


def rollout(gap: int, policy: str, seed: int, eps: float = 0.1, cue_always: bool = False,
            distractor_rate: float = 0.0, distractor_size: int = 1) -> dict:
    """Run one full episode and return arrays in the conventions documented above."""
    env = CueRecallEnv(gap=gap, cue_always=cue_always, distractor_rate=distractor_rate,
                       distractor_size=distractor_size)
    rng = np.random.default_rng([seed, 12345])
    frames = [env.reset(seed=seed)]
    act = make_policy(policy, env, rng, eps)
    actions, rewards, info = [], [0.0], {}
    done = False
    while not done:
        a = act()
        obs, r, done, info = env.step(a)
        actions.append(a)
        frames.append(obs)
        rewards.append(r)
    actions.append(-1)
    return dict(
        obs=np.stack(frames),
        actions=np.array(actions, dtype=np.int8),
        rewards=np.array(rewards, dtype=np.float32),
        cue=info["cue"], gap=gap, reached=info["reached"], success=info["success"],
    )
