"""Single entry point dispatching by role — kaiwu's `utils/start.sh` pattern.

Roles:
    modelpool                 HTTP model file pool
    env_server                zmq env server (K slots)
    actor                     zmq batch inference service
    learner                   on-policy PPO learner
    aisrv                     env_server subprocess + rollout orchestrator (one pod)
    smoke                     single-process end-to-end self-check (no network)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=os.environ.get("ROLE"))
    args = parser.parse_args()
    role = args.role
    if not role:
        parser.error("--role or ROLE env is required")

    if role == "smoke":
        from . import smoke

        smoke.run()
        return

    if role == "aisrv":
        # Same-pod pattern: env server on localhost, aisrv talks to it.
        env_proc = subprocess.Popen(
            [sys.executable, "-m", "kaiwu_sts2.env_server"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        time.sleep(2)  # let env_server bind and probe metadata
        try:
            from . import aisrv

            aisrv.serve()
        finally:
            env_proc.terminate()
            env_proc.wait(timeout=10)
        return

    mod = __import__(f"kaiwu_sts2.{role}", fromlist=["serve"])
    mod.serve()


if __name__ == "__main__":
    # mp not needed today, but spawn is the safe default for future env pools.
    mp.set_start_method("spawn", force=True)
    main()
