"""Local end-to-end test of the kaiwu_sts2 framework on the Windows host.

Runs all five roles as local processes against the win-x64 NativeAOT engine,
waits for a few learner updates, prints verdict. Same code path as the
Docker deployment (Linux differs only in the native .so).

Usage: python bench/run_local_cluster.py [seconds_to_run]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMULATOR = ROOT / "emulator"
VENV_PY = ROOT / "venv/Scripts/python.exe"

COMMON = {
    "PYTHONPATH": f"{ROOT};{EMULATOR / 'src'}",
    "STS2_LIB_PATH": str(EMULATOR / "out"),
    "MODELPOOL_IP": "127.0.0.1",
    "LEARNER_IP": "127.0.0.1",
    "ACTOR_IP": "127.0.0.1",
    "MODEL_DIR": str(ROOT / "runs/modelpool-local"),
    "ENV_SLOTS": "4",
    "BATCH_SIZE": "4096",
    "MINIBATCH_SIZE": "1024",
    "TORCH_THREADS": "2",
}


def launch(role: str, log: Path) -> subprocess.Popen:
    env = {**os.environ, **COMMON, "PYTHONUNBUFFERED": "1"}
    f = open(log, "w")
    return subprocess.Popen(
        [str(VENV_PY), "-m", "kaiwu_sts2.start", "--role", role],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
    )


def main() -> None:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    logdir = ROOT / "runs/logs-local"
    logdir.mkdir(parents=True, exist_ok=True)

    roles = ["modelpool", "learner", "actor", "aisrv"]
    procs = {}
    for role in roles:
        procs[role] = launch(role, logdir / f"{role}.log")
        time.sleep(1.5)  # startup ordering: pool -> learner -> actor -> aisrv

    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            if any(p.poll() is not None for p in procs.values()):
                print("A ROLE PROCESS EXITED EARLY")
                break
            time.sleep(2)
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    for role in roles:
        log = (logdir / f"{role}.log").read_text(errors="replace")
        print(f"===== {role}.log (last 6 lines) =====")
        print("\n".join(log.splitlines()[-6:]))

    learner_log = (logdir / "learner.log").read_text(errors="replace")
    ok = "version=" in learner_log
    print(f"\nVERDICT: {'PASS - learner published models' if ok else 'FAIL - no model versions'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
