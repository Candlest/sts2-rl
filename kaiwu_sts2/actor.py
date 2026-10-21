"""actor equivalent: batch inference service (kaiwu remote_actor_predict mode).

zmq REP on ACTOR_PORT. Request: {"obs": [[...] x B], "n_actions": int}
Response: {"action": [B], "logp": [B], "value": [B], "version": int}

Polls modelpool for newer model versions every MODEL_SYNC_INTERVAL_S.
Model dims are learned from the first request; weights load as soon as the
learner publishes version 1 (actor blocks on the first request until then).
"""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import torch
import zmq

from .codec import packb, unpackb
from .config import get, modelpool_base
from .nets import MaskedMLP, load_state_from_zip_bytes, masked_action


def _fetch_model(base: str, current: int) -> tuple[bytes, int] | None:
    """Return (zip_bytes, version) if modelpool has a newer version than current."""
    try:
        with urllib.request.urlopen(f"{base}/latest", timeout=5) as r:
            latest = int(json.loads(r.read())["version"])
        if latest <= current:
            return None
        with urllib.request.urlopen(f"{base}/download?version={latest}", timeout=30) as r:
            return r.read(), latest
    except Exception as e:  # modelpool may be briefly unreachable
        print(f"[actor] modelpool poll failed: {e}", flush=True)
        return None


def serve() -> None:
    torch.set_num_threads(int(get("TORCH_THREADS")))
    addr = f"tcp://0.0.0.0:{get('ACTOR_PORT')}"
    base = modelpool_base()
    sync_interval = float(get("MODEL_SYNC_INTERVAL_S"))
    hidden = int(get("HIDDEN_SIZE"))

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(addr)
    print(f"[actor] serving on {addr}, modelpool={base}", flush=True)

    model: MaskedMLP | None = None
    version = 0
    pending: tuple[bytes, int] | None = None
    last_sync = 0.0

    while True:
        if time.monotonic() - last_sync > sync_interval or (model is None and pending is None):
            got = _fetch_model(base, version)
            if got is not None:
                pending = got
                print(f"[actor] fetched model v{got[1]} from modelpool", flush=True)
            last_sync = time.monotonic()

        req = unpackb(sock.recv())
        obs = np.asarray(req["obs"], dtype=np.float32)

        if model is None:
            model = MaskedMLP(obs.shape[1], int(req["n_actions"]), hidden)
        if pending is not None:
            data, new_version = pending
            load_state_from_zip_bytes(model, data)
            version = new_version
            pending = None
            print(f"[actor] loaded model v{version}", flush=True)

        actions, logp, values = masked_action(model, obs)
        sock.send(
            packb(
                {
                    "action": actions.tolist(),
                    "logp": logp.tolist(),
                    "value": values.tolist(),
                    "version": version,
                }
            )
        )


if __name__ == "__main__":
    serve()
