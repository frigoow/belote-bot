"""Exact S_4 suit-equivariance, residual anchoring, and C++ BBM1 parity tests for M3 bidding."""

import itertools
import random

import numpy as np
import torch

import belote_cpp as bc
from belote.bidding import (
    AUCTION_FEAT_DIM,
    MATCH_CTX_DIM,
    NODE_K7_IDX,
    BidNet,
    DealOutcomeNet,
    encode_auction_take,
    encode_match_context,
    export_bid_bundle,
)
from belote.match_equity import estimate_outcome_dist, solve_match_equity
from belote.train_bidding import (
    choose_bid_with_bidnet,
    collect_outcome_dataset,
    expected_v_from_buckets,
)


def _permute_mask(mask: int, perm: tuple[int, int, int, int]) -> int:
    out = 0
    for c in range(32):
        if mask & (1 << c):
            s = perm[c >> 3]
            r = c & 7
            out |= 1 << (s * 8 + r)
    return out


def test_exact_s4_suit_equivariance_all_24_permutations():
    rng = random.Random(42)
    model = DealOutcomeNet(num_buckets=48)
    model.eval()
    perms = list(itertools.permutations(range(4)))

    for _ in range(25):
        deck = list(range(32))
        rng.shuffle(deck)
        hand_5 = 0
        for c in deck[:5]:
            hand_5 |= 1 << c
        blik = deck[5]
        blik_suit = blik >> 3

        for node_k, cand_trump in [(rng.randrange(0, 4), blik_suit), (rng.randrange(4, 8), (blik_suit + 1) % 4)]:
            base_feat = encode_auction_take(hand_5, blik, cand_trump, node_k)
            cpp_feat = bc.encode_auction_take(hand_5, blik, cand_trump, node_k)
            np.testing.assert_allclose(cpp_feat, base_feat, atol=1e-6)
            assert base_feat.shape == (AUCTION_FEAT_DIM,)
            with torch.no_grad():
                base_probs = model.predict_probs(torch.from_numpy(base_feat).unsqueeze(0)).numpy()

            for perm in perms:
                p_hand = _permute_mask(hand_5, perm)
                p_blik = perm[blik >> 3] * 8 + (blik & 7)
                p_trump = perm[cand_trump]
                feat = encode_auction_take(p_hand, p_blik, p_trump, node_k)
                np.testing.assert_array_equal(feat, base_feat)
                with torch.no_grad():
                    probs = model.predict_probs(torch.from_numpy(feat).unsqueeze(0)).numpy()
                np.testing.assert_array_equal(probs, base_probs)


def test_bidnet_residual_anchor_and_node_k7():
    dist = estimate_outcome_dist(num_deals=500, seed=1)
    table = solve_match_equity(dist, target=101, margin=2, pot_cap=32)
    ctx = encode_match_context(table, us=80, them=75, pot=8, we_deal=1, ev_take=0.62, best_r2_ev=0.58)
    assert ctx.shape == (MATCH_CTX_DIM,)
    assert np.all(np.isfinite(ctx))

    bidnet = BidNet()
    # Zero-initialized residuals must exactly reproduce [v_redeal, ev_take].
    af = torch.zeros((2, AUCTION_FEAT_DIM))
    af[1, NODE_K7_IDX] = 1.0
    mc = torch.from_numpy(ctx).unsqueeze(0).expand(2, -1)
    vals = bidnet(af, mc)
    assert vals.shape == (2, 2)
    np.testing.assert_allclose(vals[0, 0].item(), ctx[4], atol=1e-5)
    np.testing.assert_allclose(vals[0, 1].item(), ctx[5], atol=1e-5)

    # Even with non-zero final weights, node_k == 7 must clamp V(Pass) to exact v_redeal.
    with torch.no_grad():
        bidnet.net[-1].bias.fill_(1.5)
    vals2 = bidnet(af, mc)
    np.testing.assert_allclose(vals2[1, 0].item(), ctx[4], atol=1e-6)
    assert vals2[0, 0].item() > ctx[4] + 0.05


def test_bidding_pipeline_and_cpp_bbm1_parity(tmp_path):
    cfg = bc.Config.belgian()
    X, Y, buckets = collect_outcome_dataset(num_deals=120, config=cfg, seed=9, threads=2, max_buckets=32, min_count=1)
    assert len(X) > 60
    assert X.shape[1] == AUCTION_FEAT_DIM
    assert len(buckets) >= 5

    dist = estimate_outcome_dist(num_deals=400, seed=2)
    table = solve_match_equity(dist, target=101, margin=2, pot_cap=32)
    torch.manual_seed(7)
    don = DealOutcomeNet(num_buckets=len(buckets))
    bidnet = BidNet()
    # Non-zero residuals to test full C++ BidNet parity.
    with torch.no_grad():
        bidnet.net[-1].weight.normal_(0.0, 0.05)
        bidnet.net[-1].bias.copy_(torch.tensor([0.12, -0.08]))
    don.eval()
    bidnet.eval()

    bundle = tmp_path / "bid.bin"
    export_bid_bundle(bundle, don, bidnet, buckets, table)
    cpp_bm = bc.BidModel.load(str(bundle))

    # Verify exact V lookup parity including overtime margin=2.
    for us, them, pot, wd in [(0, 0, 0, 0), (88, 72, 8, 1), (101, 101, 0, 0), (101, 100, 16, 1), (102, 100, 0, 0)]:
        np.testing.assert_allclose(cpp_bm.lookup_v(us, them, pot, wd), table.lookup(us, them, pot, wd), atol=1e-6)

    # Verify C++ BidModel matches Python choose_bid_with_bidnet & expected_v_from_buckets across R1 and R2.
    rng = random.Random(123)
    for trial in range(12):
        deck = list(range(32))
        rng.shuffle(deck)
        deal = bc.Deal(cfg, trial % 4, deck)
        for nk in range(8):
            if deal.phase not in (bc.Phase.BID1, bc.Phase.BID2):
                break
            us, them, pot = 60 + trial * 3, 55 + trial * 2, (trial % 3) * 8
            me = deal.current_player
            wd = 1 if (deal.dealer & 1) == (me & 1) else 0
            cands = [b for b in deal.legal_bids() if b.kind != bc.Bid.Kind.PASS]
            bs = deal.blik >> 3
            v_redeal = table.lookup(us, them, pot, 1 - wd)
            r2_ev = v_redeal
            if nk < 4:
                r2_f = np.stack([encode_auction_take(deal.hand(me), deal.blik, s, nk + 4) for s in range(4) if s != bs])
                r2_p = don.predict_probs(torch.from_numpy(r2_f)).numpy()
                r2_ev = max(expected_v_from_buckets(r2_p[j], buckets, table, us, them, pot, wd) for j in range(len(r2_p)))
            fs = np.stack(
                [encode_auction_take(deal.hand(me), deal.blik, bs if b.kind == bc.Bid.Kind.TAKE else b.suit, nk) for b in cands]
            )
            ps = don.predict_probs(torch.from_numpy(fs)).numpy()
            evs = [expected_v_from_buckets(ps[j], buckets, table, us, them, pot, wd) for j in range(len(cands))]
            cs = np.stack([encode_match_context(table, us, them, pot, wd, evs[j], r2_ev) for j in range(len(cands))])
            with torch.no_grad():
                preds = bidnet(torch.from_numpy(fs), torch.from_numpy(cs)).numpy()
            best_i = int(np.argmax(preds[:, 1]))
            cpp_ev = cpp_bm.eval_bid(deal, us, them, pot)
            np.testing.assert_allclose(cpp_ev.v_pass, float(preds[best_i, 0]), atol=1e-5)
            np.testing.assert_allclose(cpp_ev.v_take, float(preds[best_i, 1]), atol=1e-5)
            np.testing.assert_allclose(cpp_ev.ev_take, float(evs[best_i]), atol=1e-5)
            assert cpp_ev.best_bid in deal.legal_bids()
            # Exact bid choice match whenever there isn't a <1e-5 symmetric-suit tie.
            sorted_vt = np.sort(preds[:, 1])
            has_tie = len(sorted_vt) > 1 and (sorted_vt[-1] - sorted_vt[-2] < 1e-5)
            if not has_tie and abs(float(preds[best_i, 1] - preds[best_i, 0])) > 1e-5:
                py_bid = choose_bid_with_bidnet(deal, nk, don, bidnet, buckets, table, us, them, pot, wd, torch.device("cpu"))
                assert cpp_ev.best_bid == py_bid
            lp = cpp_bm.bid_log_prob(deal.hand(me), deal.blik, nk, cfg, cpp_ev.best_bid, us, them, pot, wd)
            assert np.isfinite(lp) and lp <= 1e-9
            deal.bid(bc.Bid.pass_())

    # Smoke-test Stage 3 ordered-deck auction search and M4 bid+hybrid:me:auct match duel.
    from belote.models import FastPolicy

    fp = FastPolicy()
    fp.eval()
    fp_bin = tmp_path / "fast.bin"
    fp.export_bin(fp_bin)
    cpp_fp = bc.FastPolicyModel.load(str(fp_bin))
    d_s3 = bc.Deal(cfg, 0, list(range(32)))
    b_s3 = cpp_bm.choose_bid_search(d_s3, cpp_fp, 80, 75, 8, worlds=8, search_margin=1.0, seed=5)
    assert b_s3 in d_s3.legal_bids()

    opt = bc.MatchDuelOptions()
    opt.a = f"bid:{bundle}:s3=4:m=0.05+hybrid:{fp_bin}:8:ab5:me:auct"
    opt.b = "rule"
    opt.bidder = "self"
    opt.matches = 2
    opt.seed = 1
    opt.threads = 1
    opt.config = cfg
    res = bc.run_match_duel(opt)
    assert res.pairs == 2
    assert 0.0 <= res.win_rate <= 1.0
