"""M3 Suit-Equivariant DealOutcomeNet and Residual-Anchored Score-Aware BidNet (docs/PLAN_BELGIAN.md §4.2).

Exact S_4 suit equivariance by construction:
  - Candidate trump suit -> slot 0.
  - In Round 2 (node_k >= 4), blik_suit -> slot 1, and the 2 remaining plain
    suits are sorted descending by their 8-bit hand mask into slots 2, 3.
  - In Round 1 (node_k < 4, where candidate_trump == blik_suit), the 3 plain
    suits are sorted descending by their 8-bit hand mask into slots 1, 2, 3.

Any permutation of the 4 suits produces the bit-for-bit identical feature
vector in a single forward pass (0.0% spread across suits).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belote_cpp as bc
from belote.match_equity import MatchEquityTable

# 32 canonical 5-card bits + 8 blik rank + 1 is_r2 + 8 node_k one-hot + 10 hand summary scalars
AUCTION_FEAT_DIM = 32 + 8 + 1 + 8 + 10  # = 59
# [us/T, them/T, pot/cap, we_deal, v_redeal, ev_take, dv_us, dv_them, v_made - v_nat, best_r2_ev]
MATCH_CTX_DIM = 10
NODE_K7_IDX = 32 + 8 + 1 + 7  # = 48 (one-hot index for node_k == 7)


def encode_auction_take(hand_5: int, blik: int, candidate_trump: int, node_k: int) -> np.ndarray:
    """Exact S_4-equivariant encoding of taking `candidate_trump` at auction node `node_k`."""
    out = np.zeros(AUCTION_FEAT_DIM, dtype=np.float32)
    blik_suit = blik >> 3
    blik_rank = blik & 7
    is_r2 = 1 if node_k >= 4 else 0

    suit_bytes = [(hand_5 >> (s * 8)) & 0xFF for s in range(4)]

    # Canonical suit ordering:
    #  slot 0 = candidate_trump
    #  R2: slot 1 = blik_suit, slots 2..3 = remaining 2 suits sorted descending by mask
    #  R1: slots 1..3 = remaining 3 suits sorted descending by mask
    if is_r2:
        others = [s for s in range(4) if s != candidate_trump and s != blik_suit]
        others.sort(key=lambda s: suit_bytes[s], reverse=True)
        order = [candidate_trump, blik_suit, others[0], others[1]]
    else:
        others = [s for s in range(4) if s != candidate_trump]
        others.sort(key=lambda s: suit_bytes[s], reverse=True)
        order = [candidate_trump, others[0], others[1], others[2]]

    idx = 0
    for slot, s in enumerate(order):
        byte = suit_bytes[s]
        for r in range(8):
            if byte & (1 << r):
                out[idx + slot * 8 + r] = 1.0
    idx += 32

    out[idx + blik_rank] = 1.0
    idx += 8
    out[idx] = float(is_r2)
    idx += 1
    if 0 <= node_k < 8:
        out[idx + node_k] = 1.0
    idx += 8

    # Compact domain summary features (all S_4-equivariant).
    known6 = hand_5 | (1 << blik)
    trumps = (known6 >> (candidate_trump * 8)) & 0xFF
    n_trumps = trumps.bit_count()
    has_j = 1.0 if (trumps & (1 << 4)) else 0.0
    has_9 = 1.0 if (trumps & (1 << 2)) else 0.0
    has_a = 1.0 if (trumps & (1 << 7)) else 0.0
    has_10 = 1.0 if (trumps & (1 << 3)) else 0.0
    has_belot = 1.0 if ((trumps & ((1 << 5) | (1 << 6))) == ((1 << 5) | (1 << 6))) else 0.0
    n_side_aces = sum(1 for s in range(4) if s != candidate_trump and ((known6 >> (s * 8)) & (1 << 7)))
    n_side_voids = sum(1 for s in range(4) if s != candidate_trump and ((known6 >> (s * 8)) & 0xFF) == 0)
    rule_val = float(bc.rule_hand_value(known6, candidate_trump)) / 10.0
    is_dealer_partner = 1.0 if (node_k % 4 == 1) else 0.0

    out[idx + 0] = n_trumps / 6.0
    out[idx + 1] = has_j
    out[idx + 2] = has_9
    out[idx + 3] = has_a
    out[idx + 4] = has_10
    out[idx + 5] = has_belot
    out[idx + 6] = n_side_aces / 3.0
    out[idx + 7] = n_side_voids / 3.0
    out[idx + 8] = rule_val
    out[idx + 9] = is_dealer_partner
    idx += 10
    assert idx == AUCTION_FEAT_DIM
    return out


def encode_match_context(
    table: MatchEquityTable,
    us: int,
    them: int,
    pot: int,
    we_deal: int,
    ev_take: float,
    best_r2_ev: float | None = None,
) -> np.ndarray:
    """Compact match-equity summary + local V gradients + R2 option value for BidNet."""
    v_now = table.lookup(us, them, pot, we_deal)
    v_redeal = table.lookup(us, them, pot, 1 - we_deal)
    dv_us = table.lookup(us + 8, them, 0, 1 - we_deal) - v_now
    dv_them = table.lookup(us, them + 8, 0, 1 - we_deal) - v_now
    v_made = table.lookup(us + 9 + pot, them + 7, 0, 1 - we_deal)
    v_nat = table.lookup(us, them + 16 + pot, 0, 1 - we_deal)
    r2_opt = v_redeal if best_r2_ev is None else best_r2_ev
    return np.array(
        [
            us / float(table.target),
            them / float(table.target),
            pot / float(max(1, table.pot_cap)),
            float(we_deal),
            v_redeal,
            ev_take,
            dv_us,
            dv_them,
            v_made - v_nat,
            r2_opt,
        ],
        dtype=np.float32,
    )


class DealOutcomeNet(nn.Module):
    """Predicts discrete deal outcome probabilities P(bucket | 5c, blik, trump, node_k).

    Each bucket corresponds to a (du_taker, dt_def, held, taker_won) outcome
    observed during card-play rollouts. Exact E[V | Take] at any match state
    (us, them, pot, we_deal) is a single dot product against V.
    """

    def __init__(self, num_buckets: int, feat_dim: int = AUCTION_FEAT_DIM, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.num_buckets = num_buckets
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_buckets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


class BidNet(nn.Module):
    """Residual-anchored score-aware sequential bidder predicting [V(Pass), V(Take)].

    Anchors V(Pass) to exact V_redeal and V(Take) to DealOutcomeNet's exact E[V | Take]
    in logit space, so:
      - Zero-initialized residuals start immediately at calibrated DealOutcomeNet play.
      - Generalizes cleanly across all (us, them, pot, we_deal) match states.
      - At node_k == 7 (last bidder in R2), Pass is clamped to exact V_redeal.
    """

    def __init__(self, feat_dim: int = AUCTION_FEAT_DIM, ctx_dim: int = MATCH_CTX_DIM, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.ctx_dim = ctx_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ctx_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # [delta_logit_pass, delta_logit_take]
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, auction_feat: torch.Tensor, match_ctx: torch.Tensor) -> torch.Tensor:
        deltas = self.net(torch.cat([auction_feat, match_ctx], dim=-1))
        v_redeal = match_ctx[:, 4]
        ev_take = match_ctx[:, 5]
        is_last = auction_feat[:, NODE_K7_IDX] > 0.5

        logit_pass = torch.logit(v_redeal.clamp(1e-4, 1.0 - 1e-4)) + deltas[:, 0]
        logit_take = torch.logit(ev_take.clamp(1e-4, 1.0 - 1e-4)) + deltas[:, 1]
        v_pass = torch.where(is_last, v_redeal, torch.sigmoid(logit_pass))
        v_take = torch.sigmoid(logit_take)
        return torch.stack([v_pass, v_take], dim=-1)


def _write_mlp3(f, seq: nn.Sequential) -> None:
    # Layout: Linear(0), LayerNorm(1), SiLU(2), Linear(3), LayerNorm(4), SiLU(5), Linear(6)
    tensors = [
        seq[0].weight,
        seq[0].bias,
        seq[1].weight,
        seq[1].bias,
        seq[3].weight,
        seq[3].bias,
        seq[4].weight,
        seq[4].bias,
        seq[6].weight,
        seq[6].bias,
    ]
    for t in tensors:
        arr = t.detach().cpu().to(torch.float32).contiguous().numpy()
        f.write(arr.tobytes())


def export_bid_bundle(
    path: str | Path,
    don: DealOutcomeNet,
    bidnet: BidNet,
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
) -> None:
    """Export DealOutcomeNet + BidNet + MatchEquityTable to compact BBM1 binary for C++."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(b"BBM1")
        f.write(
            struct.pack(
                "<8i",
                don.feat_dim,
                bidnet.ctx_dim,
                don.hidden,
                len(buckets),
                bidnet.hidden,
                table.target,
                table.margin,
                table.pot_cap,
            )
        )
        for du, dt, held, won in buckets:
            f.write(struct.pack("<4i", int(du), int(dt), int(held), int(won)))
        f.write(np.ascontiguousarray(table.V, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(table.V_tie, dtype=np.float32).tobytes())
        _write_mlp3(f, don.net)
        _write_mlp3(f, bidnet.net)
