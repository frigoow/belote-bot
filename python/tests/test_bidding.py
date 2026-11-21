"""Exact S_4 suit-equivariance and shape tests for M3 bidding models."""

import itertools
import random

import numpy as np
import torch

import belote_cpp as bc
from belote.bidding import (
    AUCTION_FEAT_DIM,
    MATCH_CTX_DIM,
    BidNet,
    DealOutcomeNet,
    encode_auction_take,
    encode_match_context,
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
    model = DealOutcomeNet(num_buckets=64)
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

        # Check both Round 1 (candidate_trump == blik_suit) and Round 2 (candidate_trump != blik_suit).
        for node_k, cand_trump in [(rng.randrange(0, 4), blik_suit), (rng.randrange(4, 8), (blik_suit + 1) % 4)]:
            base_feat = encode_auction_take(hand_5, blik, cand_trump, node_k)
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


def test_bidnet_and_match_context():
    dist = estimate_outcome_dist(num_deals=500, seed=1)
    table = solve_match_equity(dist, target=101, margin=2, pot_cap=32)
    ctx = encode_match_context(table, us=80, them=75, pot=8, we_deal=1, ev_take=0.62)
    assert ctx.shape == (MATCH_CTX_DIM,)
    assert np.all(np.isfinite(ctx))

    bidnet = BidNet()
    af = torch.zeros((4, AUCTION_FEAT_DIM))
    mc = torch.from_numpy(ctx).unsqueeze(0).expand(4, -1)
    vals = bidnet(af, mc)
    assert vals.shape == (4, 2)
    assert torch.all((vals >= 0.0) & (vals <= 1.0))


def test_bidding_pipeline_smoke():
    cfg = bc.Config.belgian()
    X, Y, buckets = collect_outcome_dataset(num_deals=60, config=cfg, seed=9)
    assert len(X) > 30
    assert X.shape[1] == AUCTION_FEAT_DIM
    assert len(buckets) >= 5

    dist = estimate_outcome_dist(num_deals=300, seed=2)
    table = solve_match_equity(dist, target=101, margin=2, pot_cap=32)
    don = DealOutcomeNet(num_buckets=len(buckets))
    bidnet = BidNet()
    don.eval()
    bidnet.eval()

    probs = np.full(len(buckets), 1.0 / len(buckets), dtype=np.float64)
    ev = expected_v_from_buckets(probs, buckets, table, 50, 50, 0, 0)
    assert 0.0 <= ev <= 1.0

    deal = bc.Deal(cfg, 0, list(range(32)))
    bid = choose_bid_with_bidnet(deal, 0, don, bidnet, buckets, table, 50, 50, 0, 1, torch.device("cpu"))
    assert bid in deal.legal_bids()
