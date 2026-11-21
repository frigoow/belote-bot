"""Verification of exact 1-pass Match Equity V[us, them, pot, we_deal] vs Monte Carlo."""

import numpy as np

import belote_cpp as bc
from belote.match_equity import DealOutcomeDist, estimate_outcome_dist, solve_match_equity


def _simulate_mc(
    dist: DealOutcomeDist,
    target: int,
    margin: int,
    us0: int,
    them0: int,
    pot0: int,
    we_deal0: int,
    n_matches: int,
    seed: int = 1,
) -> tuple[float, int]:
    """Direct Monte Carlo simulation of matches from the exact same DealOutcomeDist."""
    rng = np.random.default_rng(seed)
    m = max(1, margin)
    cdfs = []
    for d in (0, 1):
        p_all = np.concatenate([[dist.pass_prob[d]], dist.probs[d]])
        p_all = p_all / p_all.sum()
        cdfs.append(np.cumsum(p_all))

    wins = 0
    max_pot_seen = pot0
    for _ in range(n_matches):
        us, them, pot, d = us0, them0, pot0, we_deal0
        while True:
            u = rng.random()
            idx = int(np.searchsorted(cdfs[d], u))
            if idx == 0:
                d = 1 - d
                continue
            k = idx - 1
            h = int(dist.held[d][k])
            if h > 0:
                us += int(dist.du[d][k])
                them += int(dist.dt[d][k])
                pot += h
                if pot > max_pot_seen:
                    max_pot_seen = pot
            else:
                us += int(dist.du[d][k]) + (pot if dist.us_won[d][k] else 0)
                them += int(dist.dt[d][k]) + (pot if not dist.us_won[d][k] else 0)
                pot = 0
            d = 1 - d
            if max(us, them) >= target and abs(us - them) >= m:
                if us > them:
                    wins += 1
                break
    return wins / n_matches, max_pot_seen


def test_match_equity_matches_monte_carlo_belgian_and_wikibooks():
    dist = estimate_outcome_dist(num_deals=4000, bidder="rule", player="rule", seed=7)

    # Test both Config::belgian() (margin=2) and Config::wikibooks() (margin=0).
    for margin in (2, 0):
        table = solve_match_equity(dist, target=101, margin=margin, pot_cap=64)
        # Symmetry at (0, 0, 0): V(0,0,0,0) + V(0,0,0,1) == 1.0 exactly.
        assert abs(table.lookup(0, 0, 0, 0) + table.lookup(0, 0, 0, 1) - 1.0) < 1e-12
        # Monotonicity in score.
        assert table.lookup(90, 50, 0, 0) > table.lookup(50, 50, 0, 0) > table.lookup(50, 90, 0, 0)

        # Verify against Monte Carlo across diverse score regimes (including high pot & overtime!).
        probe_states = [
            (0, 0, 0, 0),
            (85, 85, 8, 1),
            (95, 90, 16, 0),
            (100, 99, 8, 1),
            (101, 101, 0, 0),
        ]
        n_mc = 15000
        for s_idx, (us, them, pot, wd) in enumerate(probe_states):
            dp_val = table.lookup(us, them, pot, wd)
            mc_val, max_pot = _simulate_mc(dist, 101, margin, us, them, pot, wd, n_mc, seed=100 + s_idx)
            se = np.sqrt(max(1e-6, dp_val * (1.0 - dp_val)) / n_mc)
            assert abs(dp_val - mc_val) < 4.5 * se + 1e-3, (
                f"margin={margin} state={(us, them, pot, wd)}: dp={dp_val:.5f} vs mc={mc_val:.5f} (se={se:.5f})"
            )
            assert max_pot < 64
