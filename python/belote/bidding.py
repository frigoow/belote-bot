"""M3 Suit-Equivariant DealOutcomeNet and Score-Aware BidNet (PLAN.md §4.2).

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belote_cpp as bc
from belote.match_equity import MatchEquityTable

# 32 canonical 5-card bits + 8 blik rank + 1 is_r2 + 8 node_k one-hot + 10 hand summary scalars
AUCTION_FEAT_DIM = 32 + 8 + 1 + 8 + 10  # = 59
MATCH_CTX_DIM = 9


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
    # Taker's 6 known cards = hand_5 | blik.
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
    is_dealer_partner = 1.0 if (node_k % 4 == 1) else 0.0  # seat after dealer's partner = seat 2 (k=1)

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
) -> np.ndarray:
    """Compact match-equity summary + local V gradients for Score-Aware BidNet."""
    v_now = table.lookup(us, them, pot, we_deal)
    v_redeal = table.lookup(us, them, pot, 1 - we_deal)
    dv_us = table.lookup(us + 8, them, 0, 1 - we_deal) - v_now
    dv_them = table.lookup(us, them + 8, 0, 1 - we_deal) - v_now
    v_made = table.lookup(us + 9 + pot, them + 7, 0, 1 - we_deal)
    v_nat = table.lookup(us, them + 16 + pot, 0, 1 - we_deal)
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
        ],
        dtype=np.float32,
    )


class DealOutcomeNet(nn.Module):
    """Predicts discrete deal outcome probabilities P(bucket | 5c, blik, trump, node_k).

    Each bucket corresponds to a unique (du_taker, dt_def, held, taker_won) outcome
    observed during card-play rollouts. Exact E[V | Take] at any match state
    (us, them, pot, we_deal) is a single dot product against V.
    """

    def __init__(self, num_buckets: int, feat_dim: int = AUCTION_FEAT_DIM, hidden: int = 256):
        super().__init__()
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
    """Score-aware sequential bidder predicting E[V | hand_5, blik, node_k, match_ctx, action].

    Predicts value of Pass vs best candidate Take/Choose(s) given the S_4-equivariant
    auction features and precomputed MatchEquityTable gradients.
    """

    def __init__(self, feat_dim: int = AUCTION_FEAT_DIM, ctx_dim: int = MATCH_CTX_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ctx_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # [V(Pass), V(Take/Choose this suit)]
        )

    def forward(self, auction_feat: torch.Tensor, match_ctx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([auction_feat, match_ctx], dim=-1)))
