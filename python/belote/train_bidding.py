"""M3 Stage 2 trainer: Suit-Equivariant DealOutcomeNet + Score-Aware BidNet (PLAN.md §4.2).

Trains:
  1. DealOutcomeNet(hand_5, blik, trump, node_k) -> P(outcome bucket),
     enabling exact E[V | Take/Choose(s)] at any match state via one dot product.
  2. Score-Aware BidNet(auction_feat, match_ctx) -> [E[V | Pass], E[V | Take]],
     trained via fixed-point self-play sweeps over a stratified grid of match
     states with a 20% conventional RuleBidder partner/opponent league to prevent
     private relays, gated on both pure and mixed-partner 101-point matches.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belote_cpp as bc
from belote.bidding import (
    AUCTION_FEAT_DIM,
    MATCH_CTX_DIM,
    BidNet,
    DealOutcomeNet,
    encode_auction_take,
    encode_match_context,
)
from belote.match_equity import MatchEquityTable, estimate_outcome_dist, solve_match_equity


def _play_out_deal(deal: bc.Deal, fast_policy: bc.FastPolicyModel | None) -> bc.DealResult:
    if fast_policy is not None:
        while deal.phase == bc.Phase.PLAY:
            deal.play(fast_policy.choose_card(deal))
    else:
        while deal.phase == bc.Phase.PLAY:
            deal.play(bc.rule_choose_card(deal))
    return deal.result


def collect_outcome_dataset(
    num_deals: int,
    config: bc.Config,
    fast_policy: bc.FastPolicyModel | None = None,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Collect (auction_feat, bucket_idx) pairs for DealOutcomeNet training."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    outcomes: list[tuple[int, int, int, int]] = []

    for i in range(num_deals):
        deck = rng.permutation(32).tolist()
        deal = bc.Deal(config, i % 4, deck)
        node_k = 0
        # Mix rule bidder (with strong jitter) and random exploratory takes so
        # DealOutcomeNet calibrates across strong, borderline, and weak takes.
        while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
            me = deal.current_player
            hand_5 = deal.hand(me)
            blik = deal.blik
            legal = deal.legal_bids()
            if rng.random() < 0.15:
                chosen = legal[int(rng.integers(1, len(legal)))]
            else:
                thr = float(rng.uniform(3.5, 6.2))
                chosen = bc.rule_bid(deal, thr)
            if chosen.kind != bc.Bid.Kind.PASS:
                cand_trump = (blik >> 3) if chosen.kind == bc.Bid.Kind.TAKE else chosen.suit
                feats.append(encode_auction_take(hand_5, blik, cand_trump, node_k))
                deal.bid(chosen)
                break
            deal.bid(chosen)
            node_k += 1

        if deal.phase != bc.Phase.PLAY:
            continue
        res = _play_out_deal(deal, fast_policy)
        rec = bc.record_deal(res, config)
        takers = res.takers
        du_taker = int(rec.scores[takers])
        dt_def = int(rec.scores[1 - takers])
        held = int(rec.held) if res.outcome == bc.Outcome.HANGING else 0
        taker_won = 1 if (held == 0 and res.outcome == bc.Outcome.MADE) else 0
        outcomes.append((du_taker, dt_def, held, taker_won))

    buckets = sorted(set(outcomes))
    bucket_to_id = {b: idx for idx, b in enumerate(buckets)}
    X = np.stack(feats, axis=0)
    Y = np.array([bucket_to_id[o] for o in outcomes], dtype=np.int64)
    return X, Y, buckets


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
    val = 0.0
    for pr, (du, dt, held, taker_won) in zip(probs, buckets):
        if pr <= 0.0:
            continue
        if held > 0:
            nu, nt, npot = us + du, them + dt, min(table.pot_cap, pot + held)
        else:
            nu = us + du + (pot if taker_won else 0)
            nt = them + dt + (pot if not taker_won else 0)
            npot = 0
        val += float(pr) * table.lookup(nu, nt, npot, 1 - we_deal)
    return val


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
    legal = deal.legal_bids()
    candidates = [b for b in legal if b.kind != bc.Bid.Kind.PASS]

    feats = []
    for b in candidates:
        s = (blik >> 3) if b.kind == bc.Bid.Kind.TAKE else b.suit
        feats.append(encode_auction_take(hand_5, blik, s, node_k))
    f_np = np.stack(feats, axis=0)
    f_t = torch.from_numpy(f_np).to(device)
    probs = don.predict_probs(f_t).cpu().numpy()

    ctxs = []
    for i in range(len(candidates)):
        ev_take = expected_v_from_buckets(probs[i], buckets, table, us, them, pot, we_deal)
        ctxs.append(encode_match_context(table, us, them, pot, we_deal, ev_take))
    c_t = torch.from_numpy(np.stack(ctxs, axis=0)).to(device)

    preds = bidnet(f_t, c_t).cpu().numpy()  # (C, 2): [V(Pass), V(Take)]
    # Best candidate suit by V(Take); compare against its predicted V(Pass).
    best_i = int(np.argmax(preds[:, 1]))
    v_pass = float(preds[best_i, 0])
    v_take = float(preds[best_i, 1])
    return candidates[best_i] if v_take > v_pass else bc.Bid.pass_()


def eval_bidnet_matches(
    don: DealOutcomeNet,
    bidnet: BidNet,
    buckets: list[tuple[int, int, int, int]],
    table: MatchEquityTable,
    device: torch.device,
    config: bc.Config,
    fast_policy: bc.FastPolicyModel | None = None,
    pairs: int = 80,
    mixed_partner: bool = False,
    seed: int = 555,
) -> tuple[float, float]:
    """Paired 101-point matches: BidNet vs RuleBidder (pure or mixed-partner)."""
    don.eval()
    bidnet.eval()
    rng = np.random.default_rng(seed)
    pair_scores = []

    for i in range(pairs):
        pair_seed = seed * 100003 + i * 1009
        first_dealer = i % 4
        partner_slot = (i // 4) & 1
        win_sum = 0.0

        for a_team in (0, 1):
            m = bc.Match(config, first_dealer)
            deal_idx = 0
            while not m.finished and deal_idx < 400:
                d_rng = np.random.default_rng(pair_seed + deal_idx)
                deck = d_rng.permutation(32).tolist()
                deal = bc.Deal(config, m.dealer, deck)
                node_k = 0
                while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
                    p = deal.current_player
                    is_partner = mixed_partner and ((p // 2) == partner_slot)
                    use_bn = (not is_partner) and ((p & 1) == a_team)
                    if use_bn:
                        t_p = p & 1
                        us = m.scores[t_p]
                        them = m.scores[1 - t_p]
                        we_deal = 1 if (m.dealer & 1) == t_p else 0
                        bid = choose_bid_with_bidnet(
                            deal, node_k, don, bidnet, buckets, table, us, them, m.pot, we_deal, device
                        )
                    else:
                        bid = bc.rule_bid(deal)
                    deal.bid(bid)
                    node_k += 1

                if deal.phase == bc.Phase.PLAY:
                    _play_out_deal(deal, fast_policy)
                m.record(deal.result)
                deal_idx += 1

            w = 0.5 if not m.finished else (1.0 if m.winner == a_team else 0.0)
            win_sum += 0.5 * w
        pair_scores.append(win_sum)

    arr = np.asarray(pair_scores, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, 1.96 * se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome-deals", type=int, default=40000)
    ap.add_argument("--sweep-deals", type=int, default=20000)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--eval-pairs", type=int, default=80)
    ap.add_argument("--fast-policy", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=Path, default=Path("checkpoints/bidding.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = bc.Config.belgian()
    fast_policy = bc.FastPolicyModel.load(str(args.fast_policy)) if args.fast_policy else None

    print("1. Solving baseline 1-pass MatchEquityTable V[us, them, pot, we_deal]...", flush=True)
    dist = estimate_outcome_dist(num_deals=20000, bidder="rule", player="rule", config=config, seed=args.seed)
    table = solve_match_equity(dist, target=config.match_target, margin=config.match_win_margin, pot_cap=64)
    print(f"   V(0,0,0,we_deal=0)={table.lookup(0,0,0,0):.4f}, V(0,0,0,we_deal=1)={table.lookup(0,0,0,1):.4f}", flush=True)

    print(f"2. Collecting {args.outcome_deals:,} deals for DealOutcomeNet...", flush=True)
    t0 = time.perf_counter()
    X_out, Y_out, buckets = collect_outcome_dataset(args.outcome_deals, config, fast_policy, seed=args.seed + 10)
    print(f"   Collected {len(X_out):,} taken deals across {len(buckets)} outcome buckets in {time.perf_counter() - t0:.1f}s", flush=True)

    don = DealOutcomeNet(num_buckets=len(buckets)).to(device)
    don_opt = torch.optim.AdamW(don.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    for ep in range(8):
        perm = rng.permutation(len(X_out))
        losses = []
        don.train()
        for s in range(0, len(X_out), 512):
            idx = perm[s : s + 512]
            xb = torch.from_numpy(X_out[idx]).to(device)
            yb = torch.from_numpy(Y_out[idx]).to(device)
            loss = F.cross_entropy(don(xb), yb)
            don_opt.zero_grad(set_to_none=True)
            loss.backward()
            don_opt.step()
            losses.append(float(loss.item()))
        if ep in (0, 3, 7):
            print(f"   DealOutcomeNet epoch {ep + 1}/8: ce={np.mean(losses):.4f}", flush=True)

    # 3. Fixed-point score-aware BidNet sweeps over stratified match states.
    bidnet = BidNet().to(device)
    bn_opt = torch.optim.AdamW(bidnet.parameters(), lr=1e-3, weight_decay=1e-4)
    score_grid = [
        (0, 0, 0),
        (30, 30, 0),
        (60, 40, 0),
        (40, 60, 0),
        (75, 75, 0),
        (88, 70, 0),
        (70, 88, 0),
        (92, 92, 0),
        (80, 80, 8),
        (90, 85, 16),
        (98, 95, 0),
        (101, 101, 0),
    ]

    for sweep in range(1, args.sweeps + 1):
        t_s = time.perf_counter()
        af_list, mc_list, act_list, tgt_list = [], [], [], []
        don.eval()
        bidnet.eval()

        for i in range(args.sweep_deals):
            us0, them0, pot0 = score_grid[i % len(score_grid)]
            deck = rng.permutation(32).tolist()
            dealer = i % 4
            deal = bc.Deal(config, dealer, deck)
            # 20% league mixture with conventional RuleBidder to prevent private relays.
            league_rule_team = int(rng.integers(0, 2)) if rng.random() < 0.20 else -1

            # Record auction decisions so each seat gets labeled with realized match equity.
            records: list[tuple[int, np.ndarray, np.ndarray, int]] = []  # (team, af, mc, is_take)
            node_k = 0
            while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
                p = deal.current_player
                t_p = p & 1
                us = us0 if t_p == 0 else them0
                them = them0 if t_p == 0 else us0
                we_deal = 1 if (dealer & 1) == t_p else 0
                hand_5 = deal.hand(p)
                blik = deal.blik
                legal = deal.legal_bids()
                cands = [b for b in legal if b.kind != bc.Bid.Kind.PASS]

                # Pick candidate suit (highest DealOutcomeNet E[V | Take]).
                cand_feats = np.stack(
                    [encode_auction_take(hand_5, blik, (blik >> 3) if b.kind == bc.Bid.Kind.TAKE else b.suit, node_k) for b in cands],
                    axis=0,
                )
                with torch.no_grad():
                    c_probs = don.predict_probs(torch.from_numpy(cand_feats).to(device)).cpu().numpy()
                evs = [expected_v_from_buckets(c_probs[j], buckets, table, us, them, pot0, we_deal) for j in range(len(cands))]
                best_j = int(np.argmax(evs))
                best_cand = cands[best_j]
                af = cand_feats[best_j]
                mc = encode_match_context(table, us, them, pot0, we_deal, evs[best_j])

                if league_rule_team == t_p:
                    chosen = bc.rule_bid(deal)
                elif sweep == 1:
                    # First sweep: explore around DealOutcomeNet vs V_redeal threshold.
                    v_pass_est = table.lookup(us, them, pot0, 1 - we_deal)
                    if rng.random() < 0.20:
                        chosen = best_cand if rng.random() < 0.5 else bc.Bid.pass_()
                    else:
                        chosen = best_cand if evs[best_j] > v_pass_est else bc.Bid.pass_()
                else:
                    if rng.random() < 0.15:
                        chosen = best_cand if rng.random() < 0.5 else bc.Bid.pass_()
                    else:
                        with torch.no_grad():
                            pred = bidnet(
                                torch.from_numpy(af).unsqueeze(0).to(device),
                                torch.from_numpy(mc).unsqueeze(0).to(device),
                            ).squeeze(0).cpu().numpy()
                        chosen = best_cand if pred[1] > pred[0] else bc.Bid.pass_()

                is_take = 1 if chosen.kind != bc.Bid.Kind.PASS else 0
                records.append((t_p, af, mc, is_take))
                deal.bid(chosen)
                node_k += 1

            if deal.phase == bc.Phase.PLAY:
                res = _play_out_deal(deal, fast_policy)
                rec = bc.record_deal(res, config)
                if res.outcome == bc.Outcome.HANGING:
                    nu0, nt0 = us0 + rec.scores[0], them0 + rec.scores[1]
                    npot = min(table.pot_cap, pot0 + rec.held)
                else:
                    winner = res.takers if res.outcome == bc.Outcome.MADE else (1 - res.takers)
                    nu0 = us0 + rec.scores[0] + (pot0 if winner == 0 else 0)
                    nt0 = them0 + rec.scores[1] + (pot0 if winner == 1 else 0)
                    npot = 0
                v_t0 = table.lookup(nu0, nt0, npot, 1 - (1 if (dealer & 1) == 0 else 0))
            else:
                v_t0 = table.lookup(us0, them0, pot0, 1 - (1 if (dealer & 1) == 0 else 0))

            for t_p, af, mc, is_take in records:
                af_list.append(af)
                mc_list.append(mc)
                act_list.append(is_take)
                tgt_list.append(v_t0 if t_p == 0 else (1.0 - v_t0))

        AF = np.stack(af_list, axis=0)
        MC = np.stack(mc_list, axis=0)
        ACT = np.array(act_list, dtype=np.int64)
        TGT = np.array(tgt_list, dtype=np.float32)

        bidnet.train()
        for _ep in range(4):
            perm = rng.permutation(len(TGT))
            for s in range(0, len(TGT), 512):
                idx = perm[s : s + 512]
                afb = torch.from_numpy(AF[idx]).to(device)
                mcb = torch.from_numpy(MC[idx]).to(device)
                ab = torch.from_numpy(ACT[idx]).to(device)
                tb = torch.from_numpy(TGT[idx]).to(device)
                preds = bidnet(afb, mcb)
                chosen_v = preds[torch.arange(len(idx), device=device), ab]
                loss = F.mse_loss(chosen_v, tb)
                bn_opt.zero_grad(set_to_none=True)
                loss.backward()
                bn_opt.step()

        wr_pure, ci_pure = eval_bidnet_matches(
            don, bidnet, buckets, table, device, config, fast_policy, pairs=args.eval_pairs, mixed_partner=False
        )
        wr_mix, ci_mix = eval_bidnet_matches(
            don, bidnet, buckets, table, device, config, fast_policy, pairs=args.eval_pairs, mixed_partner=True
        )
        print(
            f"[Sweep {sweep}/{args.sweeps} | {len(TGT):,} decisions | {time.perf_counter() - t_s:.1f}s] "
            f"101-match vs rule: pure={100 * wr_pure:.1f}%±{100 * ci_pure:.1f}pp | "
            f"mixed-partner={100 * wr_mix:.1f}%±{100 * ci_mix:.1f}pp",
            flush=True,
        )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "buckets": buckets,
            "don_state_dict": don.state_dict(),
            "bidnet_state_dict": bidnet.state_dict(),
        },
        args.save,
    )
    print(f"Saved bidding checkpoint to {args.save}", flush=True)


if __name__ == "__main__":
    main()
