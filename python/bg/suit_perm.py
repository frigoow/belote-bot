"""Exact GPU suit-permutation augmentation (S_3 for Suit, S_4 for AT/NT).

Permutes every suit-indexed feature in BgBatchPlayEnv's canonical encoding
(`state[611]`, `history[T, 56]`, `legal_actions[K, 44]`, `belief_targets[32]`,
`belief_marginals[32, 3]`) consistently:
  - mode == "suit": S_3 (6 permutations) over plain suits 1, 2, 3, keeping trump
    pinned at canonical suit 0 so trump card rankings/values never mix with plain.
  - mode in ("at", "nt"): S_4 (24 permutations) over all 4 suits 0, 1, 2, 3.

Zero binary format change (BGQ1/BGB1/BGP1 unchanged).
"""

from __future__ import annotations

import itertools
import torch


def _build_tables(mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "suit":
        perms = [(0,) + p for p in itertools.permutations((1, 2, 3))]
    elif mode in ("at", "nt"):
        perms = list(itertools.permutations((0, 1, 2, 3)))
    else:
        perms = [(0, 1, 2, 3)]

    state_rows = []
    ply_rows = []
    act_rows = []
    card_rows = []

    for p in perms:
        # inv[new_suit] = old_suit so out[..., new_s] = in[..., inv[new_s]]
        inv = [0, 0, 0, 0]
        for old_s, new_s in enumerate(p):
            inv[new_s] = old_s

        # inv_card[new_c] = old_c
        inv_card = [inv[c // 8] * 8 + (c % 8) for c in range(32)]
        card_rows.append(inv_card)

        # state[611]
        st = list(range(611))
        for base in (0, 32, 64, 96, 128, 160, 192, 242, 274, 306, 338, 370, 402, 547, 579):
            for c in range(32):
                st[base + c] = base + inv_card[c]
        # 236..240: led canon suit
        for s in range(4):
            st[236 + s] = 236 + inv[s]
        # 449..489: 4 rel seats x 10 (first 4 slots are canonSuit 0..3 bid flags)
        for r in range(4):
            base = 449 + r * 10
            for s in range(4):
                st[base + s] = base + inv[s]
        # 539..543 unseen per canon suit, 543..547 own per canon suit
        for base in (539, 543):
            for s in range(4):
                st[base + s] = base + inv[s]
        state_rows.append(st)

        # ply[56]
        pl = list(range(56))
        for c in range(32):
            pl[c] = inv_card[c]
        for s in range(4):
            pl[40 + s] = 40 + inv[s]
        ply_rows.append(pl)

        # action[44]
        ac = list(range(44))
        for c in range(32):
            ac[c] = inv_card[c]
        for s in range(4):
            ac[32 + s] = 32 + inv[s]
        act_rows.append(ac)

    return (
        torch.tensor(state_rows, dtype=torch.long),
        torch.tensor(ply_rows, dtype=torch.long),
        torch.tensor(act_rows, dtype=torch.long),
        torch.tensor(card_rows, dtype=torch.long),
    )


class SuitPermuter:
    """Fast GPU gather-based suit permuter for training batches."""

    def __init__(self, mode: str, device: torch.device):
        st, pl, ac, cd = _build_tables(mode)
        self.mode = mode
        self.num_perms = int(st.shape[0])
        self.state_idx = st.to(device)  # [P, 611]
        self.ply_idx = pl.to(device)    # [P, 56]
        self.act_idx = ac.to(device)    # [P, 44]
        self.card_idx = cd.to(device)   # [P, 32]

    def sample_perm_ids(self, bsz: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_perms, (bsz,), device=device)

    def permute_state(self, state: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        # state: [B, 611]
        return torch.gather(state, 1, self.state_idx[pid])

    def permute_history(self, hist: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        # hist: [B, T, 56]
        idx = self.ply_idx[pid].unsqueeze(1).expand(-1, hist.shape[1], -1)
        return torch.gather(hist, 2, idx)

    def permute_actions(self, acts: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        # acts: [B, K, 44]
        idx = self.act_idx[pid].unsqueeze(1).expand(-1, acts.shape[1], -1)
        return torch.gather(acts, 2, idx)

    def permute_cards_1d(self, x: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        # x: [B, 32]
        return torch.gather(x, 1, self.card_idx[pid])

    def permute_cards_2d(self, x: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        # x: [B, 32, C]
        idx = self.card_idx[pid].unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, 1, idx)
