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


def test_match_equity_matches_real_cpp_match_engine():
    """Verify solve_match_equity directly against real C++ bc.Match + bc.Deal simulations."""
    cfg = bc.Config.belgian()
    dist = estimate_outcome_dist(num_deals=8000, bidder="rule", player="rule", config=cfg, seed=7)
    table = solve_match_equity(dist, target=cfg.match_target, margin=cfg.match_win_margin, pot_cap=64)

    # Run real C++ bc.Match matches with RuleAgent and compare win rates at (0,0,0) and mid-match.
    rng = np.random.default_rng(77)
    n_matches = 600
    wins_wd = [0, 0]
    tot_wd = [0, 0]
    for i in range(n_matches):
        first_dealer = i % 4
        m = bc.Match(cfg, first_dealer)
        wd0 = 1 if (first_dealer & 1) == 0 else 0
        steps = 0
        while not m.finished and steps < 400:
            deck = rng.permutation(32).tolist()
            d = bc.Deal(cfg, m.dealer, deck)
            while d.phase in (bc.Phase.BID1, bc.Phase.BID2):
                d.bid(bc.rule_bid(d))
            while d.phase == bc.Phase.PLAY:
                d.play(bc.rule_choose_card(d))
            m.record(d.result)
            steps += 1
        assert m.finished
        tot_wd[wd0] += 1
        if m.winner == 0:
            wins_wd[wd0] += 1

    for wd in (0, 1):
        emp = wins_wd[wd] / tot_wd[wd]
        dp_v = table.lookup(0, 0, 0, wd)
        se = np.sqrt(dp_v * (1.0 - dp_v) / tot_wd[wd])
        assert abs(emp - dp_v) < 4.0 * se, f"we_deal={wd}: real bc.Match={emp:.4f} vs DP={dp_v:.4f} (se={se:.4f})"
