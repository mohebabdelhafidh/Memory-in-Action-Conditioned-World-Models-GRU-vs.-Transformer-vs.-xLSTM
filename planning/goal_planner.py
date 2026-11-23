"""Plan by imagination over structured candidates (first version of E3).

When the goals appear, the agent has a belief state built from everything it has seen. It considers four
candidate plans, "walk to the <color> goal" (the shortest path that avoids the other goals), imagines each one
inside the world model (open loop: the model is fed its own predicted frames), scores each by the model's predicted
reward at the end of the imagined walk, and executes the best plan.

What the planner is and is not given:
  * given: the positions of the four goals (they are visible in the frame, so this is not hidden information)
  * NOT given: the cue. It only reaches the planner through the belief state, which is what we want to test.
This checks decision quality, not whether a planner can *find* good action sequences; full CEM over raw actions
(planning/cem_mpc.py) is the next step.
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.visualize import bfs_path  # noqa: E402


def _frame(f, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(f)).unsqueeze(0).to(device)


def observe(model, frame, action, state, device):
    """Advance the belief state with one real (frame, action). Skips the decoder and heads, which are not needed."""
    import torch
    with torch.no_grad():
        x = model._embed(_frame(frame, device), torch.tensor([int(action)], device=device))
        _, state = model.core.step(x, state)
    return state


def imagine_final_logit(model, state, first_frame, path, device):
    """Imagine walking `path` from `state`; return the reward logit at its final step."""
    import torch
    st, frame, logit = copy.deepcopy(state), _frame(first_frame, device), float("nan")
    with torch.no_grad():
        for a in path:
            o = model.step(frame, torch.tensor([int(a)], device=device), st)
            st, logit = o["state"], o["reward_logit"].float().item()
            frame = o["next_frame"].float().clamp(0, 1)  # feed its own prediction back in
    return logit


def candidate_path(env, color):
    others = {cell for c, cell in env.goal_cell.items() if c != color}
    return bfs_path(env.pos, env.goal_cell[color], others, env.grid)


def choose_goal(model, device, state, env, last_frame):
    """Returns (best_color, its_path, {color: score})."""
    paths = {c: candidate_path(env, c) for c in range(4)}
    scores = {c: imagine_final_logit(model, state, last_frame, paths[c], device) for c in range(4)}
    best = max(scores, key=scores.get)
    return best, paths[best], scores
