"""The belote_cpp bindings: the engine, sampler, search and harness from Python."""

import random

import belote_cpp as bc


def random_deal(rng, config=None, cards=32):
    """A deal bid by the rule bidder and played randomly for `cards` cards."""
    config = config or bc.Config.belgian()
    while True:
        deck = list(range(bc.CARDS))
        rng.shuffle(deck)
        deal = bc.Deal(config, rng.randrange(4), deck)
        while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
            deal.bid(bc.rule_bid(deal))
        if deal.phase != bc.Phase.PLAY:
            continue
        for _ in range(cards):
            if deal.phase != bc.Phase.PLAY:
                break
            deal.play(rng.choice(deal.legal_cards()))
        return deal


def test_cards():
    assert bc.parse_card("10H") == bc.card_of(2, 3)
    assert bc.card_name(bc.parse_card("JS")) == "JS"
    mask = bc.parse_cards("AD 7C JH")
    assert bc.mask_string(mask) == "7C AD JH"
    assert bc.cards_mask(bc.mask_cards(mask)) == mask
    assert sum(bc.card_points(c, 0) for c in range(32)) == 152


def test_full_deal_scores():
    rng = random.Random(1)
    for _ in range(200):
        deal = random_deal(rng)
        assert deal.phase == bc.Phase.DONE
        r = deal.result
        assert sum(r.card_points) in (162, 252)
        rec = bc.record_deal(r, deal.config)
        # Complement recording: the deal's units are split between the teams.
        assert sum(rec.scores) + rec.held == (sum(r.credited) + r.held) // 10


def test_bidding_rules():
    config = bc.Config.belgian()
    deal = bc.Deal(config, 3, list(range(32)))
    assert deal.blik == 20
    for _ in range(4):
        deal.bid(bc.Bid.pass_())
    assert deal.phase == bc.Phase.BID2
    assert bc.Bid.choose(bc.suit_of(deal.blik)) not in deal.legal_bids()


def test_sampler_matches_truth():
    rng = random.Random(2)
    for _ in range(50):
        deal = random_deal(rng, cards=rng.randrange(18, 28))
        if deal.phase != bc.Phase.PLAY:
            continue
        obs = bc.observe(deal, deal.current_player)
        sampler = bc.WorldSampler(obs)
        truth = bc.World([deal.hand(p) for p in range(4)])
        assert sampler.admits(truth) and sampler.consistent(truth)
        worlds = sampler.enumerate(100000)
        assert any(list(w.hands) == list(truth.hands) for w in worlds)
        for w in sampler.sample(seed=3, n=20):
            assert sampler.consistent(w)
            assert w.hands[obs.observer] == obs.hand


def test_ismcts_plays_legal_cards():
    rng = random.Random(4)
    for _ in range(10):
        deal = random_deal(rng, cards=rng.randrange(0, 28))
        if deal.phase != bc.Phase.PLAY:
            continue
        result = bc.ismcts_search(deal, bc.IsmctsOptions(iterations=200))
        assert result.best_move in deal.legal_cards()


def test_endgame_and_rollout():
    rng = random.Random(7)
    for _ in range(10):
        deal = random_deal(rng, cards=22)
        if deal.phase != bc.Phase.PLAY:
            continue
        er = bc.solve_endgame(deal)
        assert er.best_move in deal.legal_cards()
        assert len(er.move_values) == len(deal.legal_cards())
        rr = bc.rollout_search(deal, bc.RolloutOptions(worlds=16, ab_handoff_trick=5, seed=1))
        assert rr.best_move in deal.legal_cards()


def test_agents_and_duel():
    rule = bc.make_agent("rule")
    rollout = bc.make_agent("rollout:16")
    rng = random.Random(5)
    deal = random_deal(rng, cards=3)
    assert rule.choose_card(deal) in deal.legal_cards()
    assert rollout.choose_card(deal) in deal.legal_cards()
    assert "margin=2" in bc.config_summary(bc.Config.belgian())

    options = bc.DuelOptions()
    options.a, options.b, options.deals, options.threads = "rule", "random", 200, 4
    result = bc.run_duel(options)
    assert result.decks > 150
    assert result.units - 3 * result.units_se > 0
    assert "score=complement" in result.config_summary

    m_opts = bc.MatchDuelOptions()
    m_opts.a, m_opts.b, m_opts.matches, m_opts.threads = "rule", "random", 20, 4
    m_res = bc.run_match_duel(m_opts)
    assert m_res.pairs == 20
    assert m_res.win_rate > 0.70


def test_match():
    config = bc.Config.belgian()
    match = bc.Match(config, 0)
    rng = random.Random(6)
    while not match.finished:
        deck = list(range(32))
        rng.shuffle(deck)
        deal = bc.Deal(config, match.dealer, deck)
        while deal.phase in (bc.Phase.BID1, bc.Phase.BID2):
            deal.bid(bc.rule_bid(deal))
        while deal.phase == bc.Phase.PLAY:
            deal.play(rng.choice(deal.legal_cards()))
        match.record(deal.result)
    lead = max(match.scores) - min(match.scores)
    assert max(match.scores) >= 101 and lead >= 2
