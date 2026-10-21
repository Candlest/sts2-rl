"""learner equivalent: on-policy PPO training over zmq PUSH samples.

- PULL socket on LEARNER_SAMPLE_PORT receives whole episodes from aisrv.
- Accumulates until BATCH_SIZE transitions, then runs PPO epochs over
  minibatches (GAE computed per episode, kaiwu-style on-policy path without
  reverb — matches kaiwudrl's zmq PUSH -> learner_server mode).
- Pushes each new model version to modelpool as a zip (kaiwu ModelFileSync).
"""

from __future__ import annotations

import json
import signal
import time
import urllib.request

import numpy as np
import torch
import zmq

from .codec import packb, unpackb
from .config import get, modelpool_base
from .nets import MaskedMLP, obs_to_tensor, state_to_zip_bytes


def upload_model(base: str, version: int, data: bytes) -> None:
    req = urllib.request.Request(
        f"{base}/upload?version={version}", data=data, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def compute_gae(rewards, values, dones, final_value, gamma, lam):
    """Returns (advantages, returns) for one episode."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = float(final_value)
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


class Learner:
    def __init__(self):
        torch.set_num_threads(int(get("TORCH_THREADS")))
        self.gamma = float(get("GAMMA"))
        self.lam = float(get("GAE_LAMBDA"))
        self.clip = float(get("CLIP_RANGE"))
        self.ent_coef = float(get("ENT_COEF"))
        self.vf_coef = float(get("VF_COEF"))
        self.max_grad_norm = float(get("MAX_GRAD_NORM"))
        self.epochs = int(get("EPOCHS"))
        self.batch_size = int(get("BATCH_SIZE"))
        self.minibatch_size = int(get("MINIBATCH_SIZE"))
        self.model_dir = str(get("MODEL_DIR"))

        self.model: MaskedMLP | None = None
        self.opt = None
        self.version = 0
        self.updates = 0

        # rollout accumulation
        self.pending_obs: list[np.ndarray] = []
        self.pending_act: list[int] = []
        self.pending_logp: list[float] = []
        self.pending_ret: list[float] = []
        self.pending_adv: list[float] = []
        self.n_pending = 0
        self.obs_dim = None
        self.n_actions = None

        self.episodes = 0
        self.wins = 0
        self.transitions = 0
        self.t_start = time.monotonic()

    def ingest(self, msg: dict) -> None:
        if self.obs_dim is None:
            self.obs_dim = int(msg["obs_dim"])
            self.n_actions = int(msg["n_actions"])
            # aisrv feeds the network raw obs concatenated with the action mask,
            # so the network input dim is obs_dim + n_actions.
            self.model = MaskedMLP(
                self.obs_dim + self.n_actions, self.n_actions, int(get("HIDDEN_SIZE"))
            )
            self.opt = torch.optim.Adam(self.model.parameters(), lr=float(get("LR")))
            print(f"[learner] model created obs_dim={self.obs_dim} n_actions={self.n_actions}", flush=True)

        rewards = [float(r) for r in msg["rewards"]]
        values = [float(v) for v in msg["values"]]
        dones = [bool(d) for d in msg["dones"]]
        adv, ret = compute_gae(rewards, values, dones, msg["final_value"], self.gamma, self.lam)

        obs = np.asarray(msg["obs"], dtype=np.float32)
        self.pending_obs.append(obs)
        self.pending_act.extend(int(a) for a in msg["actions"])
        self.pending_logp.extend(float(lp) for lp in msg["logp"])
        self.pending_ret.extend(ret.tolist())
        self.pending_adv.extend(adv.tolist())
        self.n_pending += len(rewards)
        self.episodes += 1
        self.wins += 1 if msg["won"] else 0

    def maybe_update(self) -> None:
        if self.n_pending < self.batch_size:
            return
        assert self.model is not None and self.opt is not None

        obs = np.concatenate(self.pending_obs)
        act = np.asarray(self.pending_act, dtype=np.int64)
        old_logp = torch.as_tensor(self.pending_logp, dtype=torch.float32)
        ret = torch.as_tensor(self.pending_ret, dtype=torch.float32)
        adv = torch.as_tensor(self.pending_adv, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]

        idx = np.arange(n)
        clipfracs = []
        approx_kls = []
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                logits, values = self.model(obs_to_tensor(obs[mb]))
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(torch.as_tensor(act[mb]))
                entropy = dist.entropy().mean()

                logratio = new_logp - old_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - logratio).mean().item())
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip).float().mean().item()
                    )

                mb_adv = adv[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio, -mb_adv * ratio.clamp(1 - self.clip, 1 + self.clip)
                ).mean()
                v_loss = 0.5 * (values - ret[mb]).pow(2).mean()
                loss = pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

        self.updates += 1
        self.transitions += n
        self._clear_pending()

        # publish model
        self.version += 1
        data = state_to_zip_bytes(self.model)
        try:
            upload_model(modelpool_base(), self.version, data)
        except Exception as e:
            print(f"[learner] modelpool upload failed: {e}", flush=True)

        elapsed = time.monotonic() - self.t_start
        sps = self.transitions / elapsed
        print(
            f"[learner] update={self.updates} version={self.version} transitions={self.transitions} "
            f"train_sps={sps:.0f} episodes={self.episodes} win_rate={self.wins / max(1, self.episodes):.3f} "
            f"approx_kl={np.mean(approx_kls):.4f} clipfrac={np.mean(clipfracs):.3f}",
            flush=True,
        )

    def _clear_pending(self) -> None:
        self.pending_obs.clear()
        self.pending_act.clear()
        self.pending_logp.clear()
        self.pending_ret.clear()
        self.pending_adv.clear()
        self.n_pending = 0


def serve() -> None:
    addr = f"tcp://0.0.0.0:{get('LEARNER_SAMPLE_PORT')}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.bind(addr)
    print(f"[learner] pulling samples on {addr}", flush=True)

    learner = Learner()
    stop = {"flag": False}

    def _sigterm(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    while not stop["flag"]:
        try:
            msg = unpackb(sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            learner.maybe_update()
            time.sleep(0.001)
            continue
        learner.ingest(msg)
        learner.maybe_update()

    learner.maybe_update()
    print("[learner] shutdown", flush=True)


if __name__ == "__main__":
    serve()
