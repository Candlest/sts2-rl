"""Multi-process scaling benchmark: aggregate env steps/s vs worker count.

Each worker is a plain Python process loading the NativeAOT dll via sts2_gym —
no Godot, no rendering, ~tens of MB per worker. This mirrors how SubprocVecEnv
will be deployed for MaskablePPO training.

Usage: python scale_env.py [run|combat] [worker_counts...]
"""

import multiprocessing as mp
import sys
import time

import numpy as np

STEPS_PER_WORKER = 2000


def worker(env_kind, seed, steps, q):
    if env_kind == "run":
        from sts2_gym import Sts2RunEnv

        env = Sts2RunEnv()
        max_steps = 1000
    else:
        from sts2_gym import Sts2CombatEnv

        env = Sts2CombatEnv()
        max_steps = 50

    episodes = 0
    t0 = time.perf_counter()
    done_steps = 0
    while done_steps < steps:
        obs, _ = env.reset(seed=seed + episodes)
        terminated = truncated = False
        ep_steps = 0
        while not (terminated or truncated) and ep_steps < max_steps:
            masks = env.action_masks()
            valid = np.flatnonzero(masks)
            action = int(np.random.choice(valid)) if len(valid) else 0
            obs, _, terminated, truncated, _ = env.step(action)
            ep_steps += 1
        done_steps += ep_steps
        episodes += 1
    dt = time.perf_counter() - t0
    env.close()
    q.put((done_steps, dt, episodes))


def bench(env_kind, n_workers):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(env_kind, 1000 * i, STEPS_PER_WORKER, q))
        for i in range(n_workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    total_steps = 0
    total_eps = 0
    for _ in procs:
        steps, _, eps = q.get()
        total_steps += steps
        total_eps += eps
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    rate = total_steps / wall
    print(
        f"{env_kind:7s} workers={n_workers:3d}  wall={wall:6.2f}s  "
        f"episodes={total_eps:5d}  aggregate={rate:9.0f} steps/s  "
        f"per-worker={rate / n_workers:7.0f}"
    )
    return rate


if __name__ == "__main__":
    env_kind = sys.argv[1] if len(sys.argv) > 1 else "run"
    counts = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 12, 15]
    print(f"=== {env_kind} env scaling (steps/worker={STEPS_PER_WORKER}) ===")
    for n in counts:
        bench(env_kind, n)
