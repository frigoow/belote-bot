"""Golden parity & zero-leakage tests for the C++ <-> Python observation encoder."""

import random

import numpy as np

import belote_cpp as bc
from belote.encoder import (
    ACTION_DIM,
    ENCODER_VERSION,
    MAX_PLIES,
    PLY_DIM,
    STATE_DIM,
    canon_card,
    encode_action_py,
    encode_history_py,
    encode_state_py,
    raw_card,
)


def _configs():
    losers = bc.Config.belgian()
    losers.toon_losers_show = True
    return [bc.Config.belgian(), bc.Config.wikibooks(), bc.Config.tanguy_compat(), losers]


def test_constants_and_bijection():
    assert bc.ENCODER_VERSION == ENCODER_VERSION
    assert bc.STATE_DIM == STATE_DIM
    assert bc.PLY_DIM == PLY_DIM
    assert bc.MAX_PLIES == MAX_PLIES
    assert bc.ACTION_DIM == ACTION_DIM
    for trump in range(4):
        seen = set()
        for c in range(32):
            cc = bc.canon_card(c, trump)
            assert cc == canon_card(c, trump)
            assert bc.raw_card(cc, trump) == c
            assert raw_card(cc, trump) == c
            seen.add(cc)
        assert len(seen) == 32
        # Trump cards always map to canonical suit 0 (indices 0..7).
        for r in range(8):
            assert bc.canon_card(bc.card_of(trump, r), trump) == r


def test_golden_parity_across_configs():
    rng = random.Random(101)
    checked_obs = 0
    checked_actions = 0
    for cfg in _configs():
        for _ in range(40):
            deck = list(range(32))
            rng.shuffle(deck)
            deal = bc.Deal(cfg, rng.randrange(4), deck)
            while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
                bids = deal.legal_bids()
                deal.bid(rng.choice(bids[1:]) if rng.random() < 0.35 else bc.Bid.pass_())
            if deal.phase != bc.Phase.PLAY:
                continue
            for p in range(4):
                if deal.can_set_declare(p):
                    deal.set_declare(p, rng.random() < 0.8)

            # Check several plies throughout the entire deal (tricks 0..7).
            sample_plies = set(rng.sample(range(32), 6))
            for ply in range(32):
                if deal.phase != bc.Phase.PLAY:
                    break
                if ply in sample_plies:
                    me = deal.current_player
                    obs = bc.observe(deal, me)
                    cpp_state, cpp_hist, cpp_n = bc.encode_observation(obs)
                    py_state = encode_state_py(obs)
                    py_hist, py_n = encode_history_py(obs)

                    assert cpp_n == py_n == ply
                    np.testing.assert_array_equal(cpp_state, py_state)
                    np.testing.assert_array_equal(cpp_hist, py_hist)
                    np.testing.assert_array_equal(cpp_state, bc.encode_state(obs))
                    checked_obs += 1

                    for card in deal.legal_cards():
                        cpp_act = bc.encode_action(obs, card)
                        py_act = encode_action_py(obs, card)
                        np.testing.assert_array_equal(cpp_act, py_act)
                        checked_actions += 1

                deal.play(rng.choice(deal.legal_cards()))

    assert checked_obs >= 500
    assert checked_actions >= 1000


def test_trump_suit_equivariance():
    """Rotating all 4 suits by +1 (including trump) leaves the encoded state identical."""
    cfg = bc.Config.belgian()
    rng = random.Random(202)
    tested = 0
    for _ in range(30):
        deck = list(range(32))
        rng.shuffle(deck)
        dealer = rng.randrange(4)
        deal0 = bc.Deal(cfg, dealer, deck)
        # Rotate every card in the deck by +1 suit (mod 4).
        rot_deck = [((c >> 3) + 1) % 4 * 8 + (c & 7) for c in deck]
        deal1 = bc.Deal(cfg, dealer, rot_deck)
        while deal0.phase in (bc.Phase.BID1, bc.Phase.BID2):
            b0 = bc.rule_bid(deal0)
            b1 = bc.Bid.choose((b0.suit + 1) % 4) if b0.kind == bc.Bid.Kind.CHOOSE else b0
            deal0.bid(b0)
            deal1.bid(b1)
        if deal0.phase != bc.Phase.PLAY:
            continue
        for _ in range(14):
            if deal0.phase != bc.Phase.PLAY:
                break
            c0 = rng.choice(deal0.legal_cards())
            c1 = ((c0 >> 3) + 1) % 4 * 8 + (c0 & 7)
            deal0.play(c0)
            deal1.play(c1)
        if deal0.phase != bc.Phase.PLAY:
            continue
        me = deal0.current_player
        s0, h0, n0 = bc.encode_observation(bc.observe(deal0, me))
        s1, h1, n1 = bc.encode_observation(bc.observe(deal1, me))
        assert n0 == n1
        np.testing.assert_array_equal(s0, s1)
        np.testing.assert_array_equal(h0, h1)
        tested += 1
    assert tested >= 15


def test_zero_hidden_information_leakage():
    """Replaying any consistent sampled world must produce the exact same encoding."""
    cfg = bc.Config.belgian()
    rng = random.Random(303)
    tested = 0
    for _ in range(40):
        deck = list(range(32))
        rng.shuffle(deck)
        deal = bc.Deal(cfg, rng.randrange(4), deck)
        while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
            deal.bid(bc.rule_bid(deal))
        if deal.phase != bc.Phase.PLAY:
            continue
        for _ in range(12):
            deal.play(rng.choice(deal.legal_cards()))
        me = deal.current_player
        obs_real = bc.observe(deal, me)
        # Note: sampler.replay uses Deal::fromPlay (no blik / bids_made=-1), so
        # compare across two distinct sampled worlds to verify hidden cards never leak.
        sampler = bc.WorldSampler(obs_real)
        worlds = sampler.sample(seed=99, n=4)
        if len(worlds) < 2:
            continue
        d0 = sampler.replay(worlds[0])
        d1 = sampler.replay(worlds[1])
        s0, h0, n0 = bc.encode_observation(bc.observe(d0, me))
        s1, h1, n1 = bc.encode_observation(bc.observe(d1, me))
        assert n0 == n1
        np.testing.assert_array_equal(s0, s1)
        np.testing.assert_array_equal(h0, h1)
        tested += 1
    assert tested >= 20
