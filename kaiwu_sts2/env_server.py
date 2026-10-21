"""kaiwu_env equivalent: zmq REP server wrapping the NativeAOT StS2 environments.

v2 — pipelined resets (double buffering):
    Each slot owns an active env and a standby env. Background resetter
    threads keep the standby pre-reset with the slot's next seed, so the
    critical-path "reset" is an O(us) swap. The engine is deterministic per
    seed, so precomputed initial states are always valid.

Protocol (msgpack over zmq REP, sequential requests):
    {"type": "meta"} -> {"obs_dim": int, "n_actions": int, "kind": str}
    {"type": "reset", "slot": k, "seed": s|null}
        seed=null -> deterministic slot sequence (fast swap path)
        -> {"obs": [..], "mask": [..], "seed": int}
    {"type": "step", "slot": k, "action": a}
        -> {"obs": [..], "reward": float, "done": bool, "truncated": bool,
            "mask": [..], "won": bool, "episode_steps": int}
"""

from __future__ import annotations

import queue
import signal
import sys
import threading

import numpy as np
import zmq

from .codec import packb, unpackb
from .config import get

RESET_WORKERS = 3


class Slot:
    """Active + standby env with prefetch coordination.

    Invariant: at most one background reset job per slot is in flight.
    `ready` is set by the worker under `lock` when the standby's stash is
    populated; the swap also happens under `lock`.
    """

    def __init__(self, make_env, slot_id: int):
        self.id = slot_id
        self.active = make_env()
        self.standby = make_env()
        self.episode_idx = 0
        self.base_seed = int(get("BASE_SEED")) + slot_id * 1_000_003
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.pending = False

    def seed_for(self, episode_idx: int) -> int:
        return self.base_seed + episode_idx


def _reset_worker(jobs: "queue.Queue[tuple[Slot, object, int]]") -> None:
    while True:
        slot, env, seed = jobs.get()
        if env is None:  # poison pill
            return
        # The 45ms map-generation cost is paid here, off the critical path.
        obs, _ = env.reset(seed=seed)
        mask = env.action_masks()
        env._stash = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
        )
        with slot.lock:
            slot.pending = False
            slot.ready.set()


def serve() -> None:
    kind = str(get("ENV_KIND"))
    n_slots = int(get("ENV_SLOTS"))
    addr = f"tcp://{get('ENV_IP')}:{get('ENV_PORT')}"

    def make_env():
        if kind == "run":
            from sts2_gym import Sts2RunEnv

            return Sts2RunEnv()
        from sts2_gym import Sts2CombatEnv

        return Sts2CombatEnv()

    slots = [Slot(make_env, k) for k in range(n_slots)]

    # Probe metadata (and warm the JIT/AOT paths) without disturbing slot 0's
    # deterministic sequence.
    probe = make_env()
    probe_obs, _ = probe.reset(seed=0)
    meta = {
        "obs_dim": int(np.asarray(probe_obs).shape[0]),
        "n_actions": int(probe.action_space.n),
        "kind": kind,
        "slots": n_slots,
    }
    probe.close()

    jobs: "queue.Queue[tuple[Slot, object, int]]" = queue.Queue()
    for _ in range(RESET_WORKERS):
        threading.Thread(target=_reset_worker, args=(jobs,), daemon=True).start()

    def prime(slot: Slot) -> None:
        """Queue a background reset of slot.standby with its next seed."""
        with slot.lock:
            if slot.pending:
                return
            slot.pending = True
            slot.ready.clear()
            jobs.put((slot, slot.standby, slot.seed_for(slot.episode_idx + 1)))

    # Prime every slot: episode 0 becomes active (synchronous one-time cost),
    # episode 1 is prepared in the standby.
    for s in slots:
        obs, _ = s.active.reset(seed=s.seed_for(0))
        s.active._stash = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(s.active.action_masks(), dtype=np.float32),
        )
        prime(s)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(addr)
    print(
        f"[env_server] kind={kind} slots={n_slots} pipelined-resets bound {addr}",
        flush=True,
    )

    stop = {"flag": False}

    def _sigterm(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while not stop["flag"]:
        try:
            req = unpackb(sock.recv())
        except zmq.ZMQError:
            break
        t = req["type"]
        if t == "meta":
            sock.send(packb(meta))
            continue
        slot = slots[int(req["slot"])]
        if t == "reset":
            explicit = req.get("seed")
            if explicit is not None:
                # Testing path: synchronous reset of the active env.
                obs, _ = slot.active.reset(seed=int(explicit))
                mask = slot.active.action_masks()
                used_seed = int(explicit)
            else:
                # Fast path: swap in the pre-reset standby.
                if not slot.ready.wait(timeout=30):
                    print(f"[env_server] slot {slot.id} standby timeout", flush=True)
                with slot.lock:
                    slot.active, slot.standby = slot.standby, slot.active
                    slot.episode_idx += 1
                prime(slot)
                obs, mask = slot.active._stash
                del slot.active._stash
                used_seed = slot.seed_for(slot.episode_idx)
            sock.send(
                packb(
                    {
                        "obs": np.asarray(obs).tolist(),
                        "mask": np.asarray(mask).astype(int).tolist(),
                        "seed": used_seed,
                    }
                )
            )
        elif t == "step":
            obs, reward, done, truncated, info = slot.active.step(int(req["action"]))
            resp = {
                "obs": np.asarray(obs).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "mask": np.asarray(slot.active.action_masks()).astype(int).tolist(),
                "won": bool(done and info.get("won", False)),
                "episode_steps": int(info.get("episode_steps", 0)),
            }
            sock.send(packb(resp))
        else:
            sock.send(packb({"error": f"unknown type {t}"}))

    for _ in range(RESET_WORKERS):
        jobs.put((None, None, -1))
    print("[env_server] shutdown", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    serve()
