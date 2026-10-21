"""Masked MLP policy/value network shared by actor and learner."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import torch
from torch import nn

NEG_INF = -1e9


class MaskedMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """Returns (masked_logits, value). Mask column convention: obs[:, -n_actions:]."""
        h = self.backbone(obs)
        logits = self.pi(h)
        mask = obs[:, -self.n_actions :].bool()
        logits = logits.masked_fill(~mask, NEG_INF)
        return logits, self.v(h).squeeze(-1)


def obs_to_tensor(obs) -> torch.Tensor:
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32)


def masked_action(model: MaskedMLP, obs_batch: np.ndarray):
    """Sample actions for a batch of observations.

    Returns (actions, logp, values) numpy arrays.
    """
    with torch.no_grad():
        obs_t = obs_to_tensor(obs_batch)
        logits, values = model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return (
            actions.cpu().numpy(),
            dist.log_prob(actions).cpu().numpy(),
            values.cpu().numpy(),
        )


def state_to_zip_bytes(model: MaskedMLP) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def load_state_from_zip_bytes(model: MaskedMLP, data: bytes) -> None:
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    model.load_state_dict(state)


def save_model_zip(model: MaskedMLP, path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.pt", state_to_zip_bytes(model))


def load_model_zip(model: MaskedMLP, path) -> None:
    with zipfile.ZipFile(path) as zf:
        load_state_from_zip_bytes(model, zf.read("model.pt"))
