"""Bulgarian Belot's networks: cardbot-framework's models (cardbot.models) with
Belot's sizes and file tags, so the C++ side (bg/neural.h) reads their exports
unchanged:

- BgCardQGRU: the dueling action-as-input Q-GRU, plus a training-only head that
  predicts each card's owner from the trunk (left out of the BGQ1 export).
- BgBeliefNet: the residual belief net over DealSampler::exactMarginals() (BGB1).
- BgFastPolicy: the 256x128 MLP distilled from the Q-GRU for play-outs (BGP1).

Parameter names and construction order are those of the earlier Belot models,
so their checkpoints load and fixed-seed runs train identically.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import bg_cpp as bgc
from cardbot.models import QGRU, BeliefNet, FastPolicy
from cardbot.models import DuelingQHead, HistoryEncoder, ResBlock  # noqa: F401  (re-exported)


class BgCardQGRU(QGRU):
    """The Q-GRU with the training-only card-ownership head (32 cards x 3 other seats)."""

    def __init__(self, state_dim: int = bgc.STATE_DIM, ply_dim: int = bgc.PLY_DIM, action_dim: int = bgc.ACTION_DIM,
                 state_hidden: int = 384, gru_hidden: int = 192, act_hidden: int = 128, adv_hidden: int = 256):
        super().__init__(state_dim, ply_dim, action_dim, state_hidden, gru_hidden, act_hidden, adv_hidden)
        self.aux_own_head = nn.Linear(state_hidden + gru_hidden, 32 * 3)

    def encode_trunk(self, state: torch.Tensor, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        return self.trunk(state, history, n_plies)

    def forward(self, state, actions, history, n_plies, valid_mask=None, return_aux_own: bool = False):
        trunk = self.trunk(state, history, n_plies)
        q = self.q_head(trunk, actions, valid_mask)
        if return_aux_own:
            return q, self.aux_own_head(trunk).view(-1, 32, 3)
        return q

    def export_bin(self, path: str | Path) -> None:
        """BGQ1, without the ownership head."""
        super().export_bin(path, tag=b"BGQ1", version=bgc.ENCODER_VERSION)


class BgBeliefNet(BeliefNet):
    def __init__(self, state_dim: int = bgc.STATE_DIM, ply_dim: int = bgc.PLY_DIM, state_hidden: int = 256,
                 gru_hidden: int = 128):
        super().__init__(state_dim, ply_dim, n_cards=32, n_seats=3, state_hidden=state_hidden, gru_hidden=gru_hidden)

    def export_bin(self, path: str | Path) -> None:
        super().export_bin(path, tag=b"BGB1", version=bgc.ENCODER_VERSION)


class BgFastPolicy(FastPolicy):
    def __init__(self, state_dim: int = bgc.STATE_DIM, h1: int = 256, h2: int = 128):
        super().__init__(state_dim, 32, h1, h2)

    def export_bin(self, path: str | Path) -> None:
        super().export_bin(path, tag=b"BGP1", version=bgc.ENCODER_VERSION)
