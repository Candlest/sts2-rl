"""aisrv equivalent: rollout orchestrator.

Drives ENV_SLOTS parallel episodes against the env server in its pod:
    predict (batched, via actor) -> env step -> collect transitions ->
    on episode end, PUSH the full episode to the learner (kaiwu on-policy path).

Episode message: {
    "obs_dim": int, "n_actions": int,            # on first message only
    "obs": [[T, obs_dim]], "actions": [T], "logp": [T], "values": [T],
    "rewards": [T], "dones": [T], "final_value": float,
    "won": bool, "seed": int,
}
"""

from __future__ import annotations

import signal
import time

import numpy as np
import zmq

from .codec import packb, unpackb
from .config import get


class RemoteEnv:
    """Thin client for the env_server REP socket in the same pod."""

    def __init__(self):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.RCVTIMEO = 30000
        self.sock.connect(f"tcp://{get('ENV_IP')}:{get('ENV_PORT')}")

    def _rpc(self, req: dict) -> dict:
        self.sock.send(packb(req))
        return unpackb(self.sock.recv())

    def meta(self) -> dict:
        return self._rpc({"type": "meta"})

    def reset(self, slot: int, seed: int | None = None) -> dict:
        return self._rpc({"type": "reset", "slot": slot, "seed": seed})

    def step(self, slot: int, action: int) -> dict:
        return self._rpc({"type": "step", "slot": slot, "action": action})


def serve() -> None:
    n_slots = int(get("ENV_SLOTS"))
    env = RemoteEnv()
    meta = env.meta()
    obs_dim, n_actions = int(meta["obs_dim"]), int(meta["n_actions"])
    print(f"[aisrv] env kind={meta['kind']} obs_dim={obs_dim} n_actions={n_actions}", flush=True)

    ctx = zmq.Context.instance()
    actor = ctx.socket(zmq.REQ)
    actor.RCVTIMEO = 30000
    actor.connect(f"tcp://{get('ACTOR_IP')}:{get('ACTOR_PORT')}")
    learner = ctx.socket(zmq.PUSH)
    learner.connect(f"tcp://{get('LEARNER_IP')}:{get('LEARNER_SAMPLE_PORT')}")

    stop = {"flag": False}

    def _sigterm(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    # Per-slot episode state. Seeds are owned by the env server (deterministic
    # per-slot sequences); aisrv just logs the seed it got back.
    eps = [env.reset(slot=k) for k in range(n_slots)]
    obs = [np.asarray(e["obs"], dtype=np.float32) for e in eps]
    mask = [np.asarray(e["mask"], dtype=np.float32) for e in eps]
    # obs fed to the network = raw obs + action mask appended (nets.py convention).
    obs = [np.concatenate([o, m]) for o, m in zip(obs, mask)]

    traj = [dict(obs=[], actions=[], logp=[], values=[], rewards=[], dones=[]) for _ in range(n_slots)]
    seeds = [int(e["seed"]) for e in eps]
    episode_idx = n_slots
    sent_meta = False
    steps = 0
    t0 = time.monotonic()

    while not stop["flag"]:
        # Batch-predict one action per slot.
        actor.send(packb({"obs": np.stack(obs).tolist(), "n_actions": n_actions}))
        pred = unpackb(actor.recv())
        actions = pred["action"]

        for k in range(n_slots):
            resp = env.step(k, actions[k])
            o = np.asarray(resp["obs"], dtype=np.float32)
            m = np.asarray(resp["mask"], dtype=np.float32)
            t = traj[k]
            t["obs"].append(obs[k])
            t["actions"].append(actions[k])
            t["logp"].append(pred["logp"][k])
            t["values"].append(pred["value"][k])
            t["rewards"].append(resp["reward"])
            t["dones"].append(bool(resp["done"] or resp["truncated"]))
            steps += 1

            if resp["done"] or resp["truncated"]:
                # Bootstrap value of the terminal observation (0 on real termination).
                final_value = 0.0
                if resp["truncated"] and not resp["done"]:
                    actor.send(
                        packb({"obs": [np.concatenate([o, m]).tolist()], "n_actions": n_actions})
                    )
                    final_value = float(unpackb(actor.recv())["value"][0])
                msg = {
                    "obs": np.stack(t["obs"]).tolist(),
                    "actions": t["actions"],
                    "logp": t["logp"],
                    "values": t["values"],
                    "rewards": t["rewards"],
                    "dones": t["dones"],
                    "final_value": final_value,
                    "won": bool(resp["won"]),
                    "seed": seeds[k],
                }
                if not sent_meta:
                    msg["obs_dim"] = obs_dim
                    msg["n_actions"] = n_actions
                    sent_meta = True
                learner.send(packb(msg))
                if episode_idx % 50 == 0:
                    rate = steps / (time.monotonic() - t0)
                    print(
                        f"[aisrv] episodes_sent={episode_idx} rollout_steps/s={rate:.0f}",
                        flush=True,
                    )
                traj[k] = dict(obs=[], actions=[], logp=[], values=[], rewards=[], dones=[])
                r = env.reset(k)
                seeds[k] = int(r["seed"])
                episode_idx += 1
                o = np.asarray(r["obs"], dtype=np.float32)
                m = np.asarray(r["mask"], dtype=np.float32)

            obs[k] = np.concatenate([o, m])

    print(f"[aisrv] shutdown after {steps} rollout steps", flush=True)


if __name__ == "__main__":
    serve()
