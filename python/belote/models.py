"""Neural architectures for M2 Belgian Belote card play:
- CardQMLP: fast split-trunk action-as-input Q-network (MLP sanity baseline).
- CardQGRU: split-trunk action-as-input Q-network with chronological ply GRU.
- BeliefNet: hidden-card distribution predictor B_phi(card -> rel_seat in {1,2,3}).
- FastPolicy: compact 256x128 ReLU MLP distilled for ultra-fast C++ rollouts.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from belote.encoder import ACTION_DIM, PLY_DIM, STATE_DIM


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.silu(x + h)


class CardQMLP(nn.Module):
    """Split-trunk action-as-input Q-network: computes state trunk once per obs."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 512):
        super().__init__()
        self.arch = "mlp"
        self.state_in = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            ResBlock(hidden),
        )
        self.act_in = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def encode_trunk(
        self,
        state: torch.Tensor,
        history: torch.Tensor | None = None,
        n_plies: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.state_in(state)

    def score_actions(self, trunk: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, action_dim) or (B, K, action_dim)
        if actions.ndim == 2:
            ae = self.act_in(actions)
            return self.head(torch.cat([trunk, ae], dim=-1)).squeeze(-1)
        b, k, _ = actions.shape
        ae = self.act_in(actions)
        te = trunk.unsqueeze(1).expand(b, k, -1)
        return self.head(torch.cat([te, ae], dim=-1)).squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        history: torch.Tensor | None = None,
        n_plies: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.score_actions(self.encode_trunk(state), actions)


class HistoryEncoder(nn.Module):
    """Packs chronological ply history into a fixed hidden vector via GRU."""

    def __init__(self, ply_dim: int = PLY_DIM, hidden: int = 192):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(ply_dim, hidden, num_layers=1, batch_first=True)
        self.empty_emb = nn.Parameter(torch.zeros(hidden))

    def forward(self, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        # history: (B, T, ply_dim), n_plies: (B,) int64 in [0..T]
        bsz = history.shape[0]
        max_len = int(n_plies.max().item()) if bsz > 0 else 0
        if max_len == 0:
            return self.empty_emb.unsqueeze(0).expand(bsz, -1)

        trimmed = history[:, :max_len, :]
        out, _ = self.gru(trimmed)  # (B, max_len, H)
        has_plies = n_plies > 0
        gather_idx = (n_plies - 1).clamp(min=0).long()
        picked = out[torch.arange(bsz, device=history.device), gather_idx]
        empty = self.empty_emb.unsqueeze(0).expand(bsz, -1)
        return torch.where(has_plies.unsqueeze(-1), picked, empty)


class CardQGRU(nn.Module):
    """Action-as-input Q-GRU blueprint (PLAN.md §3.1, M2)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        ply_dim: int = PLY_DIM,
        action_dim: int = ACTION_DIM,
        state_hidden: int = 384,
        gru_hidden: int = 192,
    ):
        super().__init__()
        self.arch = "gru"
        trunk_dim = state_hidden + gru_hidden
        self.state_in = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.LayerNorm(state_hidden),
            nn.SiLU(),
        )
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk = ResBlock(trunk_dim)
        self.act_in = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(trunk_dim + 128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def encode_trunk(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        n_plies: torch.Tensor,
    ) -> torch.Tensor:
        se = self.state_in(state)
        he = self.hist_enc(history, n_plies)
        return self.trunk(torch.cat([se, he], dim=-1))

    def score_actions(self, trunk: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            ae = self.act_in(actions)
            return self.head(torch.cat([trunk, ae], dim=-1)).squeeze(-1)
        b, k, _ = actions.shape
        ae = self.act_in(actions)
        te = trunk.unsqueeze(1).expand(b, k, -1)
        return self.head(torch.cat([te, ae], dim=-1)).squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        history: torch.Tensor,
        n_plies: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_actions(self.encode_trunk(state, history, n_plies), actions)


class BeliefNet(nn.Module):
    """Hidden-card distribution network B_phi: predicts P(rel_seat in {1,2,3}) per canon card."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        ply_dim: int = PLY_DIM,
        state_hidden: int = 256,
        gru_hidden: int = 128,
    ):
        super().__init__()
        trunk_dim = state_hidden + gru_hidden
        self.state_in = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.LayerNorm(state_hidden),
            nn.SiLU(),
        )
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk = ResBlock(trunk_dim)
        self.head = nn.Linear(trunk_dim, 32 * 3)

    def forward(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        n_plies: torch.Tensor,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.trunk(torch.cat([self.state_in(state), self.hist_enc(history, n_plies)], dim=-1))
        logits = self.head(h).view(-1, 32, 3)
        if allowed is not None:
            logits = logits.masked_fill(allowed == 0, -1e9)
        return logits


class FastPolicy(nn.Module):
    """Compact 256x128 ReLU policy MLP distilled from Q-GRU for fast C++ play-outs."""

    def __init__(self, state_dim: int = STATE_DIM, h1: int = 256, h2: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def export_bin(self, path: str | Path) -> None:
        """Export flat little-endian FP32 weights for zero-dependency C++ inference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            # Magic "BFP1" + dims (state_dim, h1, h2, 32)
            f.write(b"BFP1")
            f.write(struct.pack("<4i", self.state_dim, self.h1, self.h2, 32))
            for layer in (self.fc1, self.fc2, self.fc3):
                w = layer.weight.detach().cpu().contiguous().numpy().astype("<f4")
                b = layer.bias.detach().cpu().contiguous().numpy().astype("<f4")
                f.write(w.tobytes())
                f.write(b.tobytes())


def build_q_model(arch: str) -> nn.Module:
    if arch == "mlp":
        return CardQMLP()
    if arch == "gru":
        return CardQGRU()
    raise ValueError(f"Unknown Q architecture: {arch}")
