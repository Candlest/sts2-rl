"""Minimal headless env smoke test + single-env throughput baseline (no torch needed)."""

import sys
import time

import numpy as np

from sts2_gym import Sts2CombatEnv, Sts2RunEnv


def rollout(env_cls, label, seed, max_steps=1000):
    env = env_cls()
    obs, info = env.reset(seed=seed)
    masks = env.action_masks()
    steps = 0
    t0 = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        valid = np.flatnonzero(masks)
        action = int(np.random.choice(valid)) if len(valid) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        masks = env.action_masks()
        steps += 1
    dt = time.perf_counter() - t0
    print(
        f"[{label}] seed={seed} steps={steps} done={terminated} "
        f"trunc={truncated} time={dt:.2f}s -> {steps / dt:.0f} steps/s"
    )
    env.close()
    return steps / dt


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("combat", "both"):
        for s in (0, 1):
            rollout(Sts2CombatEnv, "combat", s, max_steps=50)
    if which in ("run", "both"):
        for s in (0, 1):
            rollout(Sts2RunEnv, "run", s, max_steps=1000)
