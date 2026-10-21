"""Single-process end-to-end self-check: random policy through the real native
engine, verifying obs/mask shapes and determinism. Mirrors scripts/train.py
--check but without torch/sb3 so it runs in the lean env image too.
"""

from __future__ import annotations

import numpy as np


def run() -> None:
    from sts2_gym import Sts2CombatEnv, Sts2RunEnv

    for cls, cap, label in (
        (Sts2CombatEnv, 50, "combat"),
        (Sts2RunEnv, 1000, "run"),
    ):
        env = cls()
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs), f"{label}: obs out of space"
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,), f"{label}: bad mask shape"
        assert mask.any(), f"{label}: empty action mask"
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < cap:
            valid = np.flatnonzero(mask)
            obs, _, terminated, truncated, _ = env.step(int(np.random.choice(valid)))
            mask = env.action_masks()
            steps += 1
        env.close()
        print(f"[smoke] {label}: {steps} steps, terminated={terminated}", flush=True)

    # determinism: same seed, same first-valid-policy trajectory
    def trace(seed: int) -> np.ndarray:
        env = Sts2RunEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs.copy()]
        term = trunc = False
        while not (term or trunc):
            mask = env.action_masks()
            obs, _, term, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
            out.append(obs.copy())
        env.close()
        return np.array(out)

    same = np.array_equal(trace(7), trace(7))
    diff = not np.array_equal(trace(7), trace(8))
    assert same and diff, "determinism check failed"
    print("[smoke] determinism OK", flush=True)
    print("[smoke] all checks passed", flush=True)
