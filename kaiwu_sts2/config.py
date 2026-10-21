"""Configuration: env vars override /app/conf/configure.toml which overrides defaults.

Faithful to kaiwudrl's `conf/kaiwudrl/configure.toml` + per-role tomls, minus
the sed-rewriting step — containers get everything through environment.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULTS: dict[str, object] = {
    # --- service discovery (kaiwu: aisrv.toml actor_addrs / learner_addrs) ---
    "MODELPOOL_IP": "modelpool",
    "MODELPOOL_PORT": 10014,
    "LEARNER_IP": "learner",
    "LEARNER_SAMPLE_PORT": 9998,  # kaiwu on-policy zmq PUSH port
    "ACTOR_IP": "actor",
    "ACTOR_PORT": 8888,
    "ENV_IP": "127.0.0.1",  # env server lives in the same pod as aisrv
    "ENV_PORT": 5566,  # kaiwu kaiwu_env port
    # --- environment ---
    "ENV_KIND": "run",  # "run" (full run) | "combat"
    "ENV_SLOTS": 8,  # parallel env instances per aisrv pod
    "BASE_SEED": 100000,
    # --- model ---
    "HIDDEN_SIZE": 256,
    # --- PPO (learner) ---
    "BATCH_SIZE": 16384,
    "MINIBATCH_SIZE": 4096,
    "EPOCHS": 4,
    "LR": 3e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_RANGE": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TORCH_THREADS": 4,
    # --- model sync (kaiwu: model_file_sync/load per minutes) ---
    "MODEL_SYNC_INTERVAL_S": 30,
    "MODEL_DIR": "/models",
    "SAVE_INTERVAL_UPDATES": 10,
}

_CONFIG_PATH = Path(os.environ.get("KAIWU_CONFIG", "/app/conf/configure.toml"))


def _load_toml() -> dict[str, object]:
    if _CONFIG_PATH.is_file():
        with open(_CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        flat: dict[str, object] = {}
        for section in data.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat
    return {}


_TOML = _load_toml()


def get(key: str):
    if key in os.environ:
        raw = os.environ[key]
        default = DEFAULTS.get(key)
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        return raw
    if key in _TOML:
        return _TOML[key]
    return DEFAULTS.get(key)


def modelpool_base() -> str:
    return f"http://{get('MODELPOOL_IP')}:{get('MODELPOOL_PORT')}"
