"""Exact 1-pass DAG Match Equity solver V[us, them, pot, we_deal] (docs/PLAN_BELGIAN.md §4.1).

Solves match win probability for any fixed (score-independent) deal outcome
distribution in a single descending sweep over S = us + them with zero value
iteration:
  - Overtime / tiebreak region (max(us, them) >= target, |us - them| < M)
    solved exactly in closed form via a small linear system for pot = 0 and
    2x2 redeal self-loops for pot > 0.
  - Main DAG (us < target, them < target) solved in descending S = us + them
    using vectorized NumPy transitions and exact 2x2 redeal self-loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import belote_cpp as bc


@dataclass(frozen=True)
class DealOutcomeDist:
    """Empirical deal outcome distribution conditioned on we_deal in {0, 1}.

    For each d in (0, 1):
      - pass_prob[d]: probability of Outcome.ALL_PASSED (redeal, dealer rotates).
      - arrays of decided outcomes (parallel 1D arrays):
          probs[d]:   float64, sum(probs[d]) + pass_prob[d] == 1.0
          du[d]:      int32, recorded units added to us
          dt[d]:      int32, recorded units added to them
          held[d]:    int32, > 0 iff Outcome.HANGING (units added to pot)
          us_won[d]:  bool, True if us wins the pot when pot > 0 (non-hanging only)
    """

    pass_prob: tuple[float, float]
    probs: tuple[np.ndarray, np.ndarray]
    du: tuple[np.ndarray, np.ndarray]
    dt: tuple[np.ndarray, np.ndarray]
    held: tuple[np.ndarray, np.ndarray]
    us_won: tuple[np.ndarray, np.ndarray]


def estimate_outcome_dist(
    num_deals: int = 20000,
    bidder: str = "rule",
    player: str = "rule",
    config: bc.Config | None = None,
    seed: int = 1,
) -> DealOutcomeDist:
    """Collect an empirical DealOutcomeDist by simulating deals with C++ agents.

    By team symmetry under identical agents for both teams, we record every deal
    from both team perspectives (team 0 and team 1), doubling effective samples
    and guaranteeing exact symmetry between we_deal=0 and we_deal=1 when seats
    are symmetric.
    """
    cfg = config or bc.Config.belgian()
    bid_agent = bc.make_agent(bidder, seed + 1)
    play_agent = bc.make_agent(player, seed + 2)
    rng = np.random.default_rng(seed)

    # Map (we_deal) -> dict[(du, dt, held, us_won)] -> count
    counts: list[dict[tuple[int, int, int, int], int]] = [{}, {}]
    passes = [0, 0]
    totals = [0, 0]

    for i in range(num_deals):
        deck = rng.permutation(32).tolist()
        dealer = i % 4
        deal = bc.Deal(cfg, dealer, deck)
        while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
            deal.bid(bid_agent.choose_bid(deal))
        if deal.phase == bc.Phase.DONE:
            # All passed: record from both team 0 and team 1 perspectives.
            for team in (0, 1):
                we_deal = 1 if (dealer & 1) == team else 0
                passes[we_deal] += 1
                totals[we_deal] += 1
            continue
        while deal.phase == bc.Phase.PLAY:
            deal.play(play_agent.choose_card(deal))
        res = deal.result
        rec = bc.record_deal(res, cfg)
        deal_winner = res.takers if res.outcome == bc.Outcome.MADE else (1 - res.takers)
        for team in (0, 1):
            we_deal = 1 if (dealer & 1) == team else 0
            du = int(rec.scores[team])
            dt = int(rec.scores[1 - team])
            held = int(rec.held) if res.outcome == bc.Outcome.HANGING else 0
            us_won = 1 if (held == 0 and deal_winner == team) else 0
            key = (du, dt, held, us_won)
            counts[we_deal][key] = counts[we_deal].get(key, 0) + 1
            totals[we_deal] += 1

    pass_prob = (passes[0] / max(1, totals[0]), passes[1] / max(1, totals[1]))
    probs_l, du_l, dt_l, held_l, won_l = [], [], [], [], []
    for d in (0, 1):
        items = list(counts[d].items())
        n = max(1, totals[d])
        probs_l.append(np.array([c / n for _, c in items], dtype=np.float64))
        du_l.append(np.array([k[0] for k, _ in items], dtype=np.int32))
        dt_l.append(np.array([k[1] for k, _ in items], dtype=np.int32))
        held_l.append(np.array([k[2] for k, _ in items], dtype=np.int32))
        won_l.append(np.array([bool(k[3]) for k, _ in items], dtype=bool))

    return DealOutcomeDist(
        pass_prob=pass_prob,
        probs=(probs_l[0], probs_l[1]),
        du=(du_l[0], du_l[1]),
        dt=(dt_l[0], dt_l[1]),
        held=(held_l[0], held_l[1]),
        us_won=(won_l[0], won_l[1]),
    )


@dataclass
class MatchEquityTable:
    """Exact match win probability table for target & margin."""

    target: int
    margin: int
    pot_cap: int
    V: np.ndarray      # shape (target, target, pot_cap + 1, 2)
    V_tie: np.ndarray  # shape (2 * margin - 1, pot_cap + 1, 2), index diff + (margin - 1)

    def lookup(self, us: int, them: int, pot: int, we_deal: int) -> float:
        m = max(1, self.margin)
        if us >= self.target or them >= self.target:
            diff = us - them
            if diff >= m:
                return 1.0
            if diff <= -m:
                return 0.0
            p_idx = min(self.pot_cap, max(0, pot))
            return float(self.V_tie[diff + (m - 1), p_idx, we_deal])
        p_idx = min(self.pot_cap, max(0, pot))
        return float(self.V[us, them, p_idx, we_deal])


def solve_match_equity(
    dist: DealOutcomeDist,
    target: int = 101,
    margin: int = 2,
    pot_cap: int = 64,
) -> MatchEquityTable:
    """Solve V[us, them, pot, we_deal] exactly in one pass over the score DAG."""
    m = max(1, margin)
    n_diff = 2 * m - 1  # diff in [-(m-1) .. +(m-1)]
    p0, p1 = dist.pass_prob

    # ------------------------------------------------------------------
    # 1. Exact closed-form overtime / tiebreak table V_tie[diff_idx, pot, d]
    # ------------------------------------------------------------------
    V_tie = np.zeros((n_diff, pot_cap + 1, 2), dtype=np.float64)

    # 1a. pot = 0: solve 2 * n_diff linear system (A x = b)
    n_states = 2 * n_diff
    A = np.eye(n_states, dtype=np.float64)
    b = np.zeros(n_states, dtype=np.float64)
    for di in range(n_diff):
        diff = di - (m - 1)
        for d in (0, 1):
            row = di * 2 + d
            # Redeal flips dealer, keeps diff and pot=0
            A[row, di * 2 + (1 - d)] -= dist.pass_prob[d]
            probs = dist.probs[d]
            du = dist.du[d]
            dt = dist.dt[d]
            held = dist.held[d]
            for pr, u_add, t_add, h in zip(probs, du, dt, held):
                ndiff = int(diff + u_add - t_add)
                if ndiff >= m:
                    b[row] += pr
                elif ndiff <= -m:
                    pass
                else:
                    # Only possible when h == 0 (a hanging deal changes diff by >= 8 >= m)
                    if h != 0:
                        raise ValueError("Unexpected hanging deal inside overtime margin")
                    ndi = ndiff + (m - 1)
                    A[row, ndi * 2 + (1 - d)] -= pr

    x0 = np.linalg.solve(A, b)
    for di in range(n_diff):
        V_tie[di, 0, 0] = x0[di * 2]
        V_tie[di, 0, 1] = x0[di * 2 + 1]

    # 1b. pot > 0: every decided deal either ends immediately or clears pot -> 0.
    denom = 1.0 - p0 * p1
    for pot in range(1, pot_cap + 1):
        for di in range(n_diff):
            diff = di - (m - 1)
            e = [0.0, 0.0]
            for d in (0, 1):
                probs = dist.probs[d]
                du = dist.du[d]
                dt = dist.dt[d]
                held = dist.held[d]
                us_won = dist.us_won[d]
                is_hang = held > 0
                pot_u = np.where((~is_hang) & us_won, pot, 0)
                pot_t = np.where((~is_hang) & (~us_won), pot, 0)
                ndiff = diff + (du + pot_u) - (dt + pot_t)
                win_now = ndiff >= m
                lose_now = ndiff <= -m
                cont = (~win_now) & (~lose_now)
                val = float(probs[win_now].sum())
                if np.any(cont):
                    # Continues with pot' = 0 into already-solved V_tie[:, 0, 1 - d]
                    ndi = ndiff[cont] + (m - 1)
                    val += float((probs[cont] * V_tie[ndi, 0, 1 - d]).sum())
                e[d] = val
            V_tie[di, pot, 0] = (e[0] + p0 * e[1]) / denom
            V_tie[di, pot, 1] = (e[1] + p1 * e[0]) / denom

    # ------------------------------------------------------------------
    # 2. Main DAG V[us, them, pot, d], single descending sweep on S = us + them
    # ------------------------------------------------------------------
    V = np.zeros((target, target, pot_cap + 1, 2), dtype=np.float64)
    pots = np.arange(pot_cap + 1, dtype=np.int32)[:, None]  # (P, 1)

    # Pre-split outcomes by dealer for vectorized evaluation over all pots at once.
    is_hang_d = [dist.held[d] > 0 for d in (0, 1)]
    non_hang_us_d = [(~is_hang_d[d]) & dist.us_won[d] for d in (0, 1)]
    non_hang_them_d = [(~is_hang_d[d]) & (~dist.us_won[d]) for d in (0, 1)]

    for S in range(2 * (target - 1), -1, -1):
        u_min = max(0, S - (target - 1))
        u_max = min(target - 1, S)
        for us in range(u_min, u_max + 1):
            them = S - us
            e0_vec = np.zeros(pot_cap + 1, dtype=np.float64)
            e1_vec = np.zeros(pot_cap + 1, dtype=np.float64)

            for d, out_vec in ((0, e0_vec), (1, e1_vec)):
                probs = dist.probs[d]  # (K,)
                if len(probs) == 0:
                    continue
                # Shapes: (P, K)
                nu = us + dist.du[d][None, :] + np.where(non_hang_us_d[d][None, :], pots, 0)
                nt = them + dist.dt[d][None, :] + np.where(non_hang_them_d[d][None, :], pots, 0)
                npot = np.where(is_hang_d[d][None, :], np.minimum(pot_cap, pots + dist.held[d][None, :]), 0)

                over = (nu >= target) | (nt >= target)
                diff = nu - nt
                win = over & (diff >= m)
                tie = over & (diff > -m) & (diff < m)
                live = ~over

                vals = np.zeros_like(nu, dtype=np.float64)
                vals[win] = 1.0
                if np.any(tie):
                    vals[tie] = V_tie[diff[tie] + (m - 1), npot[tie], 1 - d]
                if np.any(live):
                    vals[live] = V[nu[live], nt[live], npot[live], 1 - d]
                out_vec[:] = vals @ probs

            V[us, them, :, 0] = (e0_vec + p0 * e1_vec) / denom
            V[us, them, :, 1] = (e1_vec + p1 * e0_vec) / denom

    return MatchEquityTable(target=target, margin=m, pot_cap=pot_cap, V=V, V_tie=V_tie)
