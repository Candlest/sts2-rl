"""Standalone test for the pipelined env_server: reset latency + determinism."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

env = {
    **os.environ,
    "PYTHONPATH": f"{ROOT};{ROOT / 'emulator' / 'src'}",
    "STS2_LIB_PATH": str(ROOT / "emulator" / "out"),
    "ENV_KIND": "run",
    "ENV_SLOTS": "4",
}


def main() -> None:
    server = subprocess.Popen(
        [str(ROOT / "venv/Scripts/python.exe"), "-m", "kaiwu_sts2.env_server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        time.sleep(6)  # startup: prime all slots (4 x 45ms resets on workers)

        sys.path.insert(0, str(ROOT))
        from kaiwu_sts2.aisrv import RemoteEnv

        envc = RemoteEnv()
        meta = envc.meta()
        print("meta:", meta)

        t0 = time.perf_counter()
        r = envc.reset(0)
        cold = time.perf_counter() - t0
        for i in range(5):
            t0 = time.perf_counter()
            r = envc.reset(0)
            print(f"reset#{i}: {(time.perf_counter() - t0)*1000:.2f}ms seed={r['seed']}")
        print(f"cold first reset: {cold*1000:.2f}ms")
        print("(compare: synchronous reset measured earlier = ~45ms)")

        # determinism: explicit-seed reset must match the pipelined sequence.
        base = 100000  # BASE_SEED default
        slot = 1
        explicit = envc._rpc({"type": "reset", "slot": slot, "seed": base + slot * 1_000_003 + 1})
        pipelined = envc.reset(slot)  # episode_idx now 1 -> same seed
        same = np.allclose(np.array(explicit["obs"]), np.array(pipelined["obs"]))
        print(f"determinism (explicit == pipelined): {same}")
        assert same, "pipelined reset diverged from explicit-seed reset"

        resp = envc.step(0, 0)
        print(f"step ok: done={resp['done']} mask_sum={sum(resp['mask'])}")
        print("ENV_SERVER TEST PASS")
    finally:
        server.terminate()
        out, _ = server.communicate(timeout=15)
        print("----- env_server log -----")
        print("\n".join(out.splitlines()[-4:]))


if __name__ == "__main__":
    main()
