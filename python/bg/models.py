"""Manille v1.2.3 neural architectures for Bulgarian Bridge-Belot:
- BgCardQGRU: Dueling action-as-input Q-GRU blueprint with 2-block residual trunk
  and training-only auxiliary 32x3 card-ownership head (discarded at export -> 0 ms).
- BgBeliefNet: Residual BeliefNet B_phi predicting logit deltas over exact
  DealSampler::exactMarginals() (guaranteed >= DealSampler on every trick 1-8).
- BgFastPolicy: compact 256x128 ReLU MLP distilled from converged Q-GRU for
  ultra-fast C++ rollout play-outs.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import bg_cpp as bgc


def _write_linear(f, layer: nn.Linear) -> None:
    w = layer.weight.detach().cpu().contiguous().numpy().astype("<f4")
    b = layer.bias.detach().cpu().contiguous().numpy().astype("<f4")
    f.write(w.tobytes())
    f.write(b.tobytes())


def _write_ln(f, layer: nn.LayerNorm) -> None:
    w = layer.weight.detach().cpu().contiguous().numpy().astype("<f4")
    b = layer.bias.detach().cpu().contiguous().numpy().astype("<f4")
    f.write(w.tobytes())
    f.write(b.tobytes())


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

    def export_weights(self, f) -> None:
        _write_linear(f, self.fc1)
        _write_ln(f, self.ln1)
        _write_linear(f, self.fc2)
        _write_ln(f, self.ln2)


class HistoryEncoder(nn.Module):
    """Chronological 1-layer GRU over played cards."""

    def __init__(self, ply_dim: int = bgc.PLY_DIM, hidden: int = 192):
        super().__init__()
        self.ply_dim = ply_dim
        self.hidden = hidden
        self.gru = nn.GRU(ply_dim, hidden, num_layers=1, batch_first=True)
        self.empty_emb = nn.Parameter(torch.zeros(hidden))

    def forward(self, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        bsz = history.shape[0]
        max_len = int(n_plies.max().item()) if bsz > 0 else 0
        if max_len == 0:
            return self.empty_emb.unsqueeze(0).expand(bsz, -1)
        trimmed = history[:, :max_len, :]
        out, _ = self.gru(trimmed)
        has_plies = n_plies > 0
        gather_idx = (n_plies - 1).clamp(min=0).long()
        picked = out[torch.arange(bsz, device=history.device), gather_idx]
        empty = self.empty_emb.unsqueeze(0).expand(bsz, -1)
        return torch.where(has_plies.unsqueeze(-1), picked, empty)

    def export_weights(self, f) -> None:
        # PyTorch GRU gate order: reset (r), update (z), new (n).
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            t = getattr(self.gru, name).detach().cpu().contiguous().numpy().astype("<f4")
            f.write(t.tobytes())
        e = self.empty_emb.detach().cpu().contiguous().numpy().astype("<f4")
        f.write(e.tobytes())


class DuelingQHead(nn.Module):
    """Dueling V(s) + centered A(s, a) over all K legal candidate actions."""

    def __init__(
        self,
        trunk_dim: int,
        action_dim: int = bgc.ACTION_DIM,
        act_hidden: int = 128,
        adv_hidden: int = 256,
    ):
        super().__init__()
        self.trunk_dim = trunk_dim
        self.action_dim = action_dim
        self.act_hidden = act_hidden
        self.adv_hidden = adv_hidden
        self.v_fc1 = nn.Linear(trunk_dim, adv_hidden)
        self.v_ln1 = nn.LayerNorm(adv_hidden)
        self.v_fc2 = nn.Linear(adv_hidden, 1)

        self.act_fc = nn.Linear(action_dim, act_hidden)
        self.act_ln = nn.LayerNorm(act_hidden)

        self.adv_fc1 = nn.Linear(trunk_dim + act_hidden, adv_hidden)
        self.adv_ln1 = nn.LayerNorm(adv_hidden)
        self.adv_res = ResBlock(adv_hidden)
        self.adv_fc2 = nn.Linear(adv_hidden, 1)

    def forward(
        self,
        trunk: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = self.v_fc2(F.silu(self.v_ln1(self.v_fc1(trunk))))  # (B, 1)
        b, k_full, _ = actions.shape
        if valid_mask is not None and valid_mask.any():
            k_active = int(valid_mask.any(dim=0).nonzero(as_tuple=False).max().item()) + 1
            act_in = actions[:, :k_active, :]
            vm = valid_mask[:, :k_active].to(v.dtype)
        else:
            k_active = k_full
            act_in = actions
            vm = None

        ae = F.silu(self.act_ln(self.act_fc(act_in)))
        te = trunk.unsqueeze(1).expand(b, k_active, -1)
        h = F.silu(self.adv_ln1(self.adv_fc1(torch.cat([te, ae], dim=-1))))
        h = self.adv_res(h)
        adv = self.adv_fc2(h).squeeze(-1)
        if vm is not None:
            mean_adv = (adv * vm).sum(dim=-1, keepdim=True) / vm.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            mean_adv = adv.mean(dim=-1, keepdim=True)
        q_active = v + (adv - mean_adv)
        if k_active == k_full:
            return q_active
        q_out = torch.zeros((b, k_full), dtype=q_active.dtype, device=q_active.device)
        q_out[:, :k_active] = q_active
        return q_out

    def export_weights(self, f) -> None:
        _write_linear(f, self.v_fc1)
        _write_ln(f, self.v_ln1)
        _write_linear(f, self.v_fc2)
        _write_linear(f, self.act_fc)
        _write_ln(f, self.act_ln)
        _write_linear(f, self.adv_fc1)
        _write_ln(f, self.adv_ln1)
        self.adv_res.export_weights(f)
        _write_linear(f, self.adv_fc2)


class BgCardQGRU(nn.Module):
    """Dueling Q-GRU with 2-block residual trunk + training-only aux ownership head."""

    def __init__(
        self,
        state_dim: int = bgc.STATE_DIM,
        ply_dim: int = bgc.PLY_DIM,
        action_dim: int = bgc.ACTION_DIM,
        state_hidden: int = 384,
        gru_hidden: int = 192,
        act_hidden: int = 128,
        adv_hidden: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.ply_dim = ply_dim
        self.action_dim = action_dim
        self.state_hidden = state_hidden
        self.gru_hidden = gru_hidden
        trunk_dim = state_hidden + gru_hidden

        self.state_fc = nn.Linear(state_dim, state_hidden)
        self.state_ln = nn.LayerNorm(state_hidden)
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk1 = ResBlock(trunk_dim)
        self.trunk2 = ResBlock(trunk_dim)
        self.q_head = DuelingQHead(trunk_dim, action_dim, act_hidden, adv_hidden)
        # Training-only auxiliary card-ownership head (discarded at C++ export -> 0 ms).
        self.aux_own_head = nn.Linear(trunk_dim, 32 * 3)

    def encode_trunk(self, state: torch.Tensor, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        se = F.silu(self.state_ln(self.state_fc(state)))
        he = self.hist_enc(history, n_plies)
        h = torch.cat([se, he], dim=-1)
        h = self.trunk1(h)
        return self.trunk2(h)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        history: torch.Tensor,
        n_plies: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_aux_own: bool = False,
    ):
        trunk = self.encode_trunk(state, history, n_plies)
        q = self.q_head(trunk, actions, valid_mask)
        if return_aux_own:
            own_logits = self.aux_own_head(trunk).view(-1, 32, 3)
            return q, own_logits
        return q

    def export_bin(self, path: str | Path) -> None:
        """Export BGQ1 flat FP32 weights (excluding training-only aux_own_head)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(b"BGQ1")
            f.write(
                struct.pack(
                    "<8i",
                    bgc.ENCODER_VERSION,
                    self.state_dim,
                    self.ply_dim,
                    self.action_dim,
                    self.state_hidden,
                    self.gru_hidden,
                    self.q_head.act_hidden,
                    self.q_head.adv_hidden,
                )
            )
            _write_linear(f, self.state_fc)
            _write_ln(f, self.state_ln)
            self.hist_enc.export_weights(f)
            self.trunk1.export_weights(f)
            self.trunk2.export_weights(f)
            self.q_head.export_weights(f)


class BgBeliefNet(nn.Module):
    """Residual BeliefNet B_phi over exact DealSampler::exactMarginals().

    Predicts logit deltas on top of log(clamp(sampler_marginals, 1e-6)), with a
    zero-initialized output head so at step 0 B_phi == DealSampler exactly.
    """

    def __init__(
        self,
        state_dim: int = bgc.STATE_DIM,
        ply_dim: int = bgc.PLY_DIM,
        state_hidden: int = 256,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.ply_dim = ply_dim
        self.state_hidden = state_hidden
        self.gru_hidden = gru_hidden
        trunk_dim = state_hidden + gru_hidden

        # Feeds state + flattened exact sampler marginals (96) into the state tower.
        self.state_fc = nn.Linear(state_dim + 96, state_hidden)
        self.state_ln = nn.LayerNorm(state_hidden)
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk1 = ResBlock(trunk_dim)
        self.trunk2 = ResBlock(trunk_dim)
        self.delta_head = nn.Linear(trunk_dim, 32 * 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        state: torch.Tensor,
        history: torch.Tensor,
        n_plies: torch.Tensor,
        sampler_marginals: torch.Tensor,
    ) -> torch.Tensor:
        # sampler_marginals: (B, 32, 3) exact analytical uniform probabilities.
        m_flat = sampler_marginals.reshape(state.shape[0], 96).to(state.dtype)
        se = F.silu(self.state_ln(self.state_fc(torch.cat([state, m_flat], dim=-1))))
        he = self.hist_enc(history, n_plies)
        h = self.trunk2(self.trunk1(torch.cat([se, he], dim=-1)))
        delta = self.delta_head(h).view(-1, 32, 3).float()
        base = torch.log(sampler_marginals.float().clamp(min=1e-6))
        logits = base + delta
        return logits.masked_fill(sampler_marginals <= 0.0, -1e9)

    def export_bin(self, path: str | Path) -> None:
        """Export BGB1 flat FP32 weights for zero-dependency C++ Belief-DP inference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(b"BGB1")
            f.write(
                struct.pack(
                    "<5i",
                    bgc.ENCODER_VERSION,
                    self.state_dim,
                    self.ply_dim,
                    self.state_hidden,
                    self.gru_hidden,
                )
            )
            _write_linear(f, self.state_fc)
            _write_ln(f, self.state_ln)
            self.hist_enc.export_weights(f)
            self.trunk1.export_weights(f)
            self.trunk2.export_weights(f)
            _write_linear(f, self.delta_head)


class BgFastPolicy(nn.Module):
    """Compact 256x128 ReLU MLP distilled from converged Q-GRU for C++ rollouts."""

    def __init__(self, state_dim: int = bgc.STATE_DIM, h1: int = 256, h2: int = 128):
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
        """Export BGP1 flat FP32 weights for fast C++ rollout play-outs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(b"BGP1")
            f.write(struct.pack("<5i", bgc.ENCODER_VERSION, self.state_dim, self.h1, self.h2, 32))
            for layer in (self.fc1, self.fc2, self.fc3):
                _write_linear(f, layer)
