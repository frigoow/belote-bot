"""M3 Stage 2 trainer: Suit-Equivariant DealOutcomeNet + Residual-Anchored BidNet (PLAN.md §4.2).

Key upgrades over v1:
  - Fast multithreaded C++ auction outcome collection & deal play-outs (50x faster).
  - Rare tail outcome bucket collapsing (keeps compact ~48 buckets covering >99% mass).
  - Residual-anchored BidNet with best-R2-suit option value feature and exact node_k==7 redeal.
  - Fixes league_rule_team suit-encoding bug (always encodes the suit actually chosen).
  - Vectorized lockstep GPU/CPU auction rollouts + MatchEquityTable V^(t+1) updates across sweeps.
  - Exports BBM1 binary bundle and runs exact C++ paired 101-point match gates (pure + mixed).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import belote_cpp as bc
from belote.bidding import (
    AUCTION_FEAT_DIM,
    MATCH_CTX_DIM,
    BidNet,
    DealOutcomeNet,
    encode_auction_take,
    encode_match_context,
    export_bid_bundle,
)
from belote.match_equity import DealOutcomeDist, MatchEquityTable, estimate_outcome_dist, solve_match_equity


def collapse_buckets(
    raw_outcomes: np.ndarray,
    max_buckets: int = 48,
    min_count: int = 30,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Collapse rare high-toon tail buckets onto nearest kept bucket of same (hanging, won) class."""
    tuples = [tuple(int(v) for v in row) for row in raw_outcomes]
    counts = Counter(tuples)
    kept = [b for b, c in counts.most_common(max_buckets) if c >= min_count]
    if not kept:
        kept = [b for b, _ in counts.most_common(min(max_buckets, len(counts)))]
    kept.sort()
    b2i = {b: idx for idx, b in enumerate(kept)}

    def nearest_id(o: tuple[int, int, int, int]) -> int:
        if o in b2i:
            return b2i[o]
        is_hang = o[2] > 0
        won = o[3]
        same_cls = [idx for idx, kb in enumerate(kept) if (kb[2] > 0) == is_hang and kb[3] == won]
        pool = same_cls if same_cls else list(range(len(kept)))
        return min(
            pool,
            key=lambda idx: abs(kept[idx][0] - o[0]) + abs(kept[idx][1] - o[1]) + abs(kept[idx][2] - o[2]),
        )

    y = np.array([nearest_id(o) for o in tuples], dtype=np.int64)
    return y, kept


def collect_outcome_dataset(
    num_deals: int,
    config: bc.Config,
    fast_policy: bc.FastPolicyModel | None = None,
    seed: int = 1,
    threads: int = 8,
    max_buckets: int = 48,
    min_count: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Collect (auction_feat, bucket_idx) pairs using multithreaded C++ rollouts."""
    raw = bc.collect_auction_outcomes(
        num_deals=num_deals,
        config=config,
        fast_policy=fast_policy,
        seed=seed,
        threads=threads,
        explore_prob=0.18,
    )
    X = raw["features"]
    Y, buckets = collapse_buckets(raw["outcomes"], max_buckets=max_buckets, min_count=min_count)
    return X, Y, buckets


def build_bucket_v_matrix(
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
    us: int,
    them: int,
    pot: int,
    we_deal: int,
) -> np.ndarray:
    """Returns float32 vector of length K with post-deal V for each outcome bucket."""
    next_d = 1 - (we_deal & 1)
    out = np.empty(len(buckets), dtype=np.float32)
    for i, (du, dt, held, taker_won) in enumerate(buckets):
        if held > 0:
            out[i] = table.lookup(us + du, them + dt, min(table.pot_cap, pot + held), next_d)
        else:
            nu = us + du + (pot if taker_won else 0)
            nt = them + dt + (pot if not taker_won else 0)
            out[i] = table.lookup(nu, nt, 0, next_d)
    return out


def expected_v_from_buckets(
    probs: np.ndarray,
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
    us: int,
    them: int,
    pot: int,
    we_deal: int,
) -> float:
    """Exact E[V | Take] from DealOutcomeNet bucket probabilities."""
    v_vec = build_bucket_v_matrix(buckets, table, us, them, pot, we_deal)
    return float(np.dot(probs, v_vec))


@torch.no_grad()
def choose_bid_with_bidnet(
    deal: bc.Deal,
    node_k: int,
    don: DealOutcomeNet,
    bidnet: BidNet,
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
    us: int,
    them: int,
    pot: int,
    we_deal: int,
    device: torch.device,
) -> bc.Bid:
    """Selects highest-E[V] legal bid (Pass vs Take/Choose(s)) at the given match state."""
    me = deal.current_player
    hand_5 = deal.hand(me)
    blik = deal.blik
    blik_suit = blik >> 3
    legal = deal.legal_bids()
    candidates = [b for b in legal if b.kind != bc.Bid.Kind.PASS]

    v_vec = build_bucket_v_matrix(buckets, table, us, them, pot, we_deal)
    v_redeal = table.lookup(us, them, pot, 1 - we_deal)

    # Best Round-2 backup suit EV when deciding in Round 1 (node_k < 4).
    best_r2_ev = v_redeal
    if node_k < 4:
        r2_suits = [s for s in range(4) if deal.config.round2_allow_blik_suit or s != blik_suit]
        r2_feats = np.stack([encode_auction_take(hand_5, blik, s, node_k + 4) for s in r2_suits], axis=0)
        r2_probs = don.predict_probs(torch.from_numpy(r2_feats).to(device)).cpu().numpy()
        best_r2_ev = float(np.max(r2_probs @ v_vec))

    feats = [
        encode_auction_take(hand_5, blik, blik_suit if b.kind == bc.Bid.Kind.TAKE else b.suit, node_k)
        for b in candidates
    ]
    f_np = np.stack(feats, axis=0)
    f_t = torch.from_numpy(f_np).to(device)
    probs = don.predict_probs(f_t).cpu().numpy()
    evs = probs @ v_vec

    ctxs = np.stack(
        [encode_match_context(table, us, them, pot, we_deal, float(evs[i]), best_r2_ev) for i in range(len(candidates))],
        axis=0,
    )
    c_t = torch.from_numpy(ctxs).to(device)
    preds = bidnet(f_t, c_t).cpu().numpy()
    best_i = int(np.argmax(preds[:, 1]))
    return candidates[best_i] if float(preds[best_i, 1]) > float(preds[best_i, 0]) else bc.Bid.pass_()


def train_don(
    don: DealOutcomeNet,
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 10,
    lr: float = 1.2e-3,
    batch_size: int = 1024,
) -> float:
    opt = torch.optim.AdamW(don.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    n = len(X)
    last_ce = 0.0
    for ep in range(epochs):
        perm = rng.permutation(n)
        don.train()
        losses = []
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(Y[idx]).to(device)
            loss = F.cross_entropy(don(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        last_ce = float(np.mean(losses))
        if ep in (0, epochs // 2, epochs - 1):
            print(f"   DealOutcomeNet epoch {ep + 1}/{epochs}: ce={last_ce:.4f}", flush=True)
    return last_ce


def collect_sweep_batched(
    sweep: int,
    num_deals: int,
    score_grid: list[tuple[int, int, int]],
    don: DealOutcomeNet,
    bidnet: BidNet,
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
    config: bc.Config,
    fast_policy: bc.FastPolicyModel | None,
    device: torch.device,
    rng: np.random.Generator,
    chunk_size: int = 512,
    threads: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DealOutcomeDist]:
    """Batched GPU auction + C++ multithreaded play-out sweep.

    Also collects (taken_feats, taken_outcomes) for DealOutcomeNet refitting and
    an empirical DealOutcomeDist (on non-exploratory deals) to update V^(t+1).
    """
    don.eval()
    bidnet.eval()

    # Precompute bucket-V vectors for every (grid_idx, perspective_team, we_deal).
    v_vecs: dict[tuple[int, int, int, int], np.ndarray] = {}
    for us0, them0, pot0 in score_grid:
        for us, them in ((us0, them0), (them0, us0)):
            for wd in (0, 1):
                key = (us, them, pot0, wd)
                if key not in v_vecs:
                    v_vecs[key] = build_bucket_v_matrix(buckets, table, us, them, pot0, wd)

    af_list, mc_list, act_list, tgt_list = [], [], [], []
    taken_feats_list, taken_out_list = [], []
    dist_counts: list[dict[tuple[int, int, int, int], int]] = [{}, {}]
    dist_passes = [0, 0]
    dist_totals = [0, 0]

    eps = 0.18 if sweep == 1 else max(0.06, 0.14 - 0.03 * (sweep - 2))
    r2_allow_blik = bool(config.round2_allow_blik_suit)

    for base_i in range(0, num_deals, chunk_size):
        cur_n = min(chunk_size, num_deals - base_i)
        deals: list[bc.Deal] = []
        grid_states: list[tuple[int, int, int]] = []
        dealers: list[int] = []
        league_rule_teams: list[int] = []
        had_explore: list[bool] = [False] * cur_n
        # Single-deviation exploration (zero continuation bias!):
        # On 40% of deals, pick ONE random node_k_exp in 0..7 to explore; every
        # subsequent seat (k > node_k_exp) plays 100% greedily so V(Pass) targets
        # are NEVER inflated by downstream opponents making random suicide takes.
        exp_node: list[int] = [int(rng.integers(0, 8)) if rng.random() < 0.40 else -1 for _ in range(cur_n)]
        # Per-deal decision records: (node_k, team, af, mc, is_take)
        deal_records: list[list[tuple[int, int, np.ndarray, np.ndarray, int]]] = [[] for _ in range(cur_n)]
        # Record taker's (af, team) when taken for DealOutcomeNet refitting.
        taker_af: list[np.ndarray | None] = [None] * cur_n

        for j in range(cur_n):
            idx_g = base_i + j
            us0, them0, pot0 = score_grid[idx_g % len(score_grid)]
            deck = rng.permutation(32).tolist()
            dealer = idx_g % 4
            deals.append(bc.Deal(config, dealer, deck))
            grid_states.append((us0, them0, pot0))
            dealers.append(dealer)
            # 20% conventional RuleBidder team prevents private relays.
            league_rule_teams.append(int(rng.integers(0, 2)) if rng.random() < 0.20 else -1)

        # Lockstep auction across up to 8 plies.
        for node_k in range(8):
            active = [j for j in range(cur_n) if deals[j].phase in (bc.Phase.BID1, bc.Phase.BID2)]
            if not active:
                break

            # Build batched DealOutcomeNet features for all legal candidates (+ R2 backup suits if node_k < 4).
            cand_feats_batch = []
            cand_meta = []  # (j, legal_cands, r2_slice_start, r2_slice_end, c_slice_start, c_slice_end)
            cursor = 0
            for j in active:
                d = deals[j]
                me = d.current_player
                hand_5 = d.hand(me)
                blik = d.blik
                blik_suit = blik >> 3
                legal = d.legal_bids()
                cands = [b for b in legal if b.kind != bc.Bid.Kind.PASS]

                c_start = cursor
                for b in cands:
                    s = blik_suit if b.kind == bc.Bid.Kind.TAKE else b.suit
                    cand_feats_batch.append(encode_auction_take(hand_5, blik, s, node_k))
                cursor += len(cands)
                c_end = cursor

                r2_start = cursor
                if node_k < 4:
                    for s in range(4):
                        if r2_allow_blik or s != blik_suit:
                            cand_feats_batch.append(encode_auction_take(hand_5, blik, s, node_k + 4))
                            cursor += 1
                r2_end = cursor
                cand_meta.append((j, cands, c_start, c_end, r2_start, r2_end))

            all_f_np = np.stack(cand_feats_batch, axis=0)
            with torch.no_grad():
                all_probs = don.predict_probs(torch.from_numpy(all_f_np).to(device)).cpu().numpy()

            # Compute EVs & build BidNet inputs for all active deals.
            bn_af_list = []
            bn_mc_list = []
            bn_slices = []  # (j, cands, c_start, bn_start, bn_end, evs)
            bn_cur = 0
            for j, cands, c_start, c_end, r2_start, r2_end in cand_meta:
                d = deals[j]
                p = d.current_player
                t_p = p & 1
                us0, them0, pot0 = grid_states[j]
                us = us0 if t_p == 0 else them0
                them = them0 if t_p == 0 else us0
                we_deal = 1 if (dealers[j] & 1) == t_p else 0
                v_vec = v_vecs[(us, them, pot0, we_deal)]
                v_redeal = table.lookup(us, them, pot0, 1 - we_deal)

                evs = all_probs[c_start:c_end] @ v_vec
                if r2_end > r2_start:
                    best_r2_ev = float(np.max(all_probs[r2_start:r2_end] @ v_vec))
                else:
                    best_r2_ev = v_redeal

                b_start = bn_cur
                for ci in range(len(cands)):
                    bn_af_list.append(all_f_np[c_start + ci])
                    bn_mc_list.append(
                        encode_match_context(table, us, them, pot0, we_deal, float(evs[ci]), best_r2_ev)
                    )
                bn_cur += len(cands)
                bn_slices.append((j, t_p, cands, b_start, bn_cur, evs))

            bn_af_np = np.stack(bn_af_list, axis=0)
            bn_mc_np = np.stack(bn_mc_list, axis=0)
            with torch.no_grad():
                bn_preds = bidnet(
                    torch.from_numpy(bn_af_np).to(device),
                    torch.from_numpy(bn_mc_np).to(device),
                ).cpu().numpy()

            for j, t_p, cands, b_start, b_end, evs in bn_slices:
                d = deals[j]
                preds_j = bn_preds[b_start:b_end]  # (C, 2)
                best_ci = int(np.argmax(preds_j[:, 1]))

                if league_rule_teams[j] == t_p:
                    chosen = bc.rule_bid(d)
                    if chosen.kind != bc.Bid.Kind.PASS:
                        chosen_ci = next(ci for ci, cb in enumerate(cands) if cb == chosen)
                    else:
                        chosen_ci = best_ci
                elif node_k == exp_node[j]:
                    # Single-seat exploratory deviation: try the opposite of greedy
                    # (or best candidate take) while all downstream seats play greedily.
                    had_explore[j] = True
                    greedy_take = float(preds_j[best_ci, 1]) > float(preds_j[best_ci, 0])
                    if greedy_take:
                        chosen = bc.Bid.pass_()
                        chosen_ci = best_ci
                    else:
                        chosen_ci = best_ci
                        chosen = cands[chosen_ci]
                else:
                    chosen_ci = best_ci
                    chosen = cands[best_ci] if float(preds_j[best_ci, 1]) > float(preds_j[best_ci, 0]) else bc.Bid.pass_()

                is_take = 1 if chosen.kind != bc.Bid.Kind.PASS else 0
                af_rec = bn_af_np[b_start + chosen_ci]
                mc_rec = bn_mc_np[b_start + chosen_ci]
                deal_records[j].append((node_k, t_p, af_rec, mc_rec, is_take))
                if is_take:
                    taker_af[j] = af_rec
                d.bid(chosen)

        # Play out all non-passed deals in parallel in C++!
        play_indices = [j for j in range(cur_n) if deals[j].phase == bc.Phase.PLAY]
        if play_indices:
            play_res = bc.play_out_deals_batch([deals[j] for j in play_indices], fast_policy, threads)
            p_scores = play_res["scores"]
            p_held = play_res["held"]
            p_outcome = play_res["outcome"]
            p_takers = play_res["takers"]
            idx_to_pos = {j: pos for pos, j in enumerate(play_indices)}
        else:
            idx_to_pos = {}

        for j in range(cur_n):
            us0, them0, pot0 = grid_states[j]
            d0_we_deal = 1 if (dealers[j] & 1) == 0 else 0
            next_d0 = 1 - d0_we_deal
            if j in idx_to_pos:
                pos = idx_to_pos[j]
                sc0, sc1 = int(p_scores[pos, 0]), int(p_scores[pos, 1])
                h = int(p_held[pos])
                out_code = int(p_outcome[pos])
                tak = int(p_takers[pos])
                is_hang = out_code == int(bc.Outcome.HANGING)
                is_made = out_code == int(bc.Outcome.MADE)
                if is_hang:
                    v_t0 = table.lookup(us0 + sc0, them0 + sc1, min(table.pot_cap, pot0 + h), next_d0)
                else:
                    winner = tak if is_made else (1 - tak)
                    nu0 = us0 + sc0 + (pot0 if winner == 0 else 0)
                    nt0 = them0 + sc1 + (pot0 if winner == 1 else 0)
                    v_t0 = table.lookup(nu0, nt0, 0, next_d0)

                if taker_af[j] is not None:
                    taken_feats_list.append(taker_af[j])
                    du_t = sc0 if tak == 0 else sc1
                    dt_d = sc1 if tak == 0 else sc0
                    tw = 1 if ((not is_hang) and is_made) else 0
                    taken_out_list.append((du_t, dt_d, h if is_hang else 0, tw))

                # Collect clean self-play outcome distribution on non-exploratory, non-league deals.
                if not had_explore[j] and league_rule_teams[j] == -1:
                    winner = tak if is_made else (1 - tak)
                    for team in (0, 1):
                        wd = 1 if (dealers[j] & 1) == team else 0
                        du = sc0 if team == 0 else sc1
                        dt = sc1 if team == 0 else sc0
                        key = (du, dt, h if is_hang else 0, 1 if ((not is_hang) and winner == team) else 0)
                        dist_counts[wd][key] = dist_counts[wd].get(key, 0) + 1
                        dist_totals[wd] += 1
            else:
                v_t0 = table.lookup(us0, them0, pot0, next_d0)
                if not had_explore[j] and league_rule_teams[j] == -1:
                    for team in (0, 1):
                        wd = 1 if (dealers[j] & 1) == team else 0
                        dist_passes[wd] += 1
                        dist_totals[wd] += 1

            # Only record decisions at k >= exp_node[j] (or all if no deviation happened)
            # so every V(Pass) label sees a 100% greedy, zero-suicide-take continuation!
            min_rec_k = exp_node[j] if had_explore[j] else 0
            for rec_k, t_p, af, mc, is_take in deal_records[j]:
                if rec_k < min_rec_k:
                    continue
                af_list.append(af)
                mc_list.append(mc)
                act_list.append(is_take)
                tgt_list.append(v_t0 if t_p == 0 else (1.0 - v_t0))

    pass_prob = (dist_passes[0] / max(1, dist_totals[0]), dist_passes[1] / max(1, dist_totals[1]))
    probs_l, du_l, dt_l, held_l, won_l = [], [], [], [], []
    for d_idx in (0, 1):
        items = list(dist_counts[d_idx].items())
        n_tot = max(1, dist_totals[d_idx])
        probs_l.append(np.array([c / n_tot for _, c in items], dtype=np.float64))
        du_l.append(np.array([k[0] for k, _ in items], dtype=np.int32))
        dt_l.append(np.array([k[1] for k, _ in items], dtype=np.int32))
        held_l.append(np.array([k[2] for k, _ in items], dtype=np.int32))
        won_l.append(np.array([bool(k[3]) for k, _ in items], dtype=bool))
    emp_dist = DealOutcomeDist(
        pass_prob=pass_prob,
        probs=(probs_l[0], probs_l[1]),
        du=(du_l[0], du_l[1]),
        dt=(dt_l[0], dt_l[1]),
        held=(held_l[0], held_l[1]),
        us_won=(won_l[0], won_l[1]),
    )

    return (
        np.stack(af_list, axis=0),
        np.stack(mc_list, axis=0),
        np.array(act_list, dtype=np.int64),
        np.array(tgt_list, dtype=np.float32),
        np.stack(taken_feats_list, axis=0) if taken_feats_list else np.zeros((0, AUCTION_FEAT_DIM), dtype=np.float32),
        np.array(taken_out_list, dtype=np.int32) if taken_out_list else np.zeros((0, 4), dtype=np.int32),
        emp_dist,
    )


def eval_bid_bundle_cpp(
    bundle_path: Path,
    play_spec: str,
    config: bc.Config,
    pairs: int = 300,
    mixed_partner: bool = False,
    s3_worlds: int = 0,
    seed: int = 555,
    threads: int = 8,
) -> bc.MatchDuelResult:
    """Fast, exact C++ paired 101-point match eval using (decided_idx, redeal_attempt) sync."""
    opt = bc.MatchDuelOptions()
    bid_spec = f"bid:{bundle_path}" + (f":s3={s3_worlds}" if s3_worlds > 0 else "") + f"+{play_spec}"
    opt.a = bid_spec
    opt.b = play_spec
    opt.bidder = "self"
    opt.player = play_spec
    opt.partner = play_spec if mixed_partner else ""
    opt.matches = pairs
    opt.seed = seed
    opt.threads = threads
    opt.config = config
    return bc.run_match_duel(opt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome-deals", type=int, default=100000)
    ap.add_argument("--sweep-deals", type=int, default=60000)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--eval-pairs", type=int, default=300)
    ap.add_argument("--max-buckets", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fast-policy", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/bidding_v2.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = bc.Config.belgian()
    fast_policy = bc.FastPolicyModel.load(str(args.fast_policy)) if args.fast_policy else None
    play_spec = f"fast:{args.fast_policy}" if args.fast_policy else "rule"
    bundle_path = args.save.with_suffix(".bin")

    print(f"1. Solving baseline 1-pass MatchEquityTable V[us, them, pot, we_deal] ({play_spec})...", flush=True)
    dist = estimate_outcome_dist(num_deals=30000, bidder="rule", player=play_spec, config=config, seed=args.seed)
    table = solve_match_equity(dist, target=config.match_target, margin=config.match_win_margin, pot_cap=64)
    print(f"   V(0,0,0,we_deal=0)={table.lookup(0,0,0,0):.4f}, V(0,0,0,we_deal=1)={table.lookup(0,0,0,1):.4f}", flush=True)

    print(f"2. Collecting {args.outcome_deals:,} deals in C++ for DealOutcomeNet...", flush=True)
    t0 = time.perf_counter()
    X_init, Y_init, buckets = collect_outcome_dataset(
        args.outcome_deals,
        config,
        fast_policy,
        seed=args.seed + 10,
        threads=args.threads,
        max_buckets=args.max_buckets,
    )
    print(
        f"   Collected {len(X_init):,} taken deals collapsed to {len(buckets)} buckets in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    don = DealOutcomeNet(num_buckets=len(buckets)).to(device)
    train_don(don, X_init, Y_init, device, seed=args.seed, epochs=12)

    # Evaluate Sweep 0 (pure DealOutcomeNet + zero-residual BidNet) before any sweeps!
    bidnet = BidNet().to(device)
    export_bid_bundle(bundle_path, don, bidnet, buckets, table)
    ev0_pure = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=args.eval_pairs, mixed_partner=False, seed=700, threads=args.threads)
    ev0_mix = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=args.eval_pairs, mixed_partner=True, seed=701, threads=args.threads)
    print(
        f"   Sweep 0 (DealOutcomeNet anchor): pure={100*ev0_pure.win_rate:.1f}%±{100*1.96*ev0_pure.win_rate_se:.1f}pp | "
        f"mixed={100*ev0_mix.win_rate:.1f}%±{100*1.96*ev0_mix.win_rate_se:.1f}pp ({ev0_pure.seconds:.1f}s)",
        flush=True,
    )

    # 3. Fixed-point score-aware BidNet sweeps over stratified match states.
    bn_opt = torch.optim.AdamW(bidnet.parameters(), lr=5e-4, weight_decay=2e-4)
    bn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bn_opt, T_max=max(1, args.sweeps), eta_min=5e-5)
    score_grid = [
        (0, 0, 0),
        (25, 25, 0),
        (50, 50, 0),
        (65, 35, 0),
        (35, 65, 0),
        (75, 75, 0),
        (88, 68, 0),
        (68, 88, 0),
        (92, 92, 0),
        (80, 80, 8),
        (90, 85, 16),
        (96, 94, 0),
        (99, 98, 0),
        (101, 101, 0),
    ]

    rng = np.random.default_rng(args.seed + 100)
    best_score = ev0_pure.win_rate + 0.5 * ev0_mix.win_rate
    best_state = {
        "don": {k: v.cpu().clone() for k, v in don.state_dict().items()},
        "bidnet": {k: v.cpu().clone() for k, v in bidnet.state_dict().items()},
        "table": table,
    }

    # Accumulated replay buffer across sweeps (capped at 600k decisions) to prevent drift.
    buf_af: list[np.ndarray] = []
    buf_mc: list[np.ndarray] = []
    buf_act: list[np.ndarray] = []
    buf_tgt: list[np.ndarray] = []

    for sweep in range(1, args.sweeps + 1):
        t_s = time.perf_counter()
        AF, MC, ACT, TGT, X_sp, O_sp, emp_dist = collect_sweep_batched(
            sweep=sweep,
            num_deals=args.sweep_deals,
            score_grid=score_grid,
            don=don,
            bidnet=bidnet,
            buckets=buckets,
            table=table,
            config=config,
            fast_policy=fast_policy,
            device=device,
            rng=rng,
            threads=args.threads,
        )

        buf_af.append(AF)
        buf_mc.append(MC)
        buf_act.append(ACT)
        buf_tgt.append(TGT)
        all_af = np.concatenate(buf_af, axis=0)[-600000:]
        all_mc = np.concatenate(buf_mc, axis=0)[-600000:]
        all_act = np.concatenate(buf_act, axis=0)[-600000:]
        all_tgt = np.concatenate(buf_tgt, axis=0)[-600000:]

        bidnet.train()
        bn_losses = []
        n_ep = 4 if sweep > 1 else 5
        for _ep in range(n_ep):
            perm = rng.permutation(len(all_tgt))
            for s in range(0, len(all_tgt), 1024):
                idx = perm[s : s + 1024]
                afb = torch.from_numpy(all_af[idx]).to(device)
                mcb = torch.from_numpy(all_mc[idx]).to(device)
                ab = torch.from_numpy(all_act[idx]).to(device)
                tb = torch.from_numpy(all_tgt[idx]).to(device)
                deltas = bidnet.net(torch.cat([afb, mcb], dim=-1))
                preds = bidnet(afb, mcb)
                chosen_v = preds[torch.arange(len(idx), device=device), ab]
                loss = F.mse_loss(chosen_v, tb) + 5e-4 * (deltas ** 2).mean()
                bn_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bidnet.parameters(), 2.0)
                bn_opt.step()
                bn_losses.append(float(loss.item()))
        bn_sched.step()

        # Update MatchEquityTable V^(t+1) from non-exploratory BidNet self-play deals.
        if emp_dist.probs[0].size > 5 and emp_dist.probs[1].size > 5:
            table = solve_match_equity(emp_dist, target=config.match_target, margin=config.match_win_margin, pot_cap=64)

        # Refit DealOutcomeNet on 50/50 mix of initial exploratory deals + on-policy BidNet deals after sweep 1.
        if sweep == 1 and len(X_sp) >= 10000:
            b2i = {b: idx for idx, b in enumerate(buckets)}
            def map_o(row):
                o = tuple(int(v) for v in row)
                if o in b2i:
                    return b2i[o]
                is_hang = o[2] > 0
                won = o[3]
                same_cls = [idx for idx, kb in enumerate(buckets) if (kb[2] > 0) == is_hang and kb[3] == won]
                pool = same_cls if same_cls else list(range(len(buckets)))
                return min(pool, key=lambda idx: abs(buckets[idx][0] - o[0]) + abs(buckets[idx][1] - o[1]) + abs(buckets[idx][2] - o[2]))
            Y_sp = np.array([map_o(r) for r in O_sp], dtype=np.int64)
            X_mix = np.concatenate([X_init, X_sp], axis=0)
            Y_mix = np.concatenate([Y_init, Y_sp], axis=0)
            print(f"   Refitting DealOutcomeNet on {len(X_mix):,} mixed exploratory+on-policy deals...", flush=True)
            train_don(don, X_mix, Y_mix, device, seed=args.seed + sweep, epochs=6, lr=6e-4)

        export_bid_bundle(bundle_path, don, bidnet, buckets, table)
        ev_p = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=args.eval_pairs, mixed_partner=False, seed=700, threads=args.threads)
        ev_m = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=args.eval_pairs, mixed_partner=True, seed=701, threads=args.threads)
        print(
            f"   Sweep {sweep}/{args.sweeps} ({time.perf_counter() - t_s:.1f}s | {len(TGT):,} dec | mse={np.mean(bn_losses):.4f} | "
            f"pass={100*0.5*(emp_dist.pass_prob[0]+emp_dist.pass_prob[1]):.1f}%): "
            f"pure={100*ev_p.win_rate:.1f}%±{100*1.96*ev_p.win_rate_se:.1f}pp | "
            f"mixed={100*ev_m.win_rate:.1f}%±{100*1.96*ev_m.win_rate_se:.1f}pp",
            flush=True,
        )
        sc = ev_p.win_rate + 0.5 * ev_m.win_rate
        if sc > best_score:
            best_score = sc
            best_state = {
                "don": {k: v.cpu().clone() for k, v in don.state_dict().items()},
                "bidnet": {k: v.cpu().clone() for k, v in bidnet.state_dict().items()},
                "table": table,
            }

    # Restore best sweep & save final PyTorch + C++ BBM1 bundle.
    don.load_state_dict(best_state["don"])
    bidnet.load_state_dict(best_state["bidnet"])
    table = best_state["table"]
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "buckets": buckets,
            "don_state_dict": don.state_dict(),
            "bidnet_state_dict": bidnet.state_dict(),
            "V": table.V,
            "V_tie": table.V_tie,
            "target": table.target,
            "margin": table.margin,
            "pot_cap": table.pot_cap,
        },
        args.save,
    )
    export_bid_bundle(bundle_path, don, bidnet, buckets, table)

    # Final high-precision held-out 101-point match gate (500 pairs = 1000 matches).
    ev_final_p = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=500, mixed_partner=False, seed=9001, threads=args.threads)
    ev_final_m = eval_bid_bundle_cpp(bundle_path, play_spec, config, pairs=500, mixed_partner=True, seed=9002, threads=args.threads)
    print(
        f"\nFINAL HELD-OUT GATE (500 paired 101-pt matches, {play_spec}):\n"
        f"  Pure BidNet vs RuleBidder:   {100*ev_final_p.win_rate:.2f}% ± {100*1.96*ev_final_p.win_rate_se:.2f}pp "
        f"(margin {ev_final_p.score_margin:+.2f} ± {1.96*ev_final_p.score_margin_se:.2f} units/match)\n"
        f"  Mixed (BidNet+Rule) vs Rule: {100*ev_final_m.win_rate:.2f}% ± {100*1.96*ev_final_m.win_rate_se:.2f}pp "
        f"(margin {ev_final_m.score_margin:+.2f} ± {1.96*ev_final_m.score_margin_se:.2f} units/match)\n"
        f"Saved PyTorch checkpoint to {args.save} and C++ BBM1 bundle to {bundle_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
