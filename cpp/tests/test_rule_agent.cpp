// Rule agent v2 (agents/rule_agent.h):
//  - tableView() reads exactly the public facts inferConstraints() does;
//  - the rule policy only ever plays legal cards, under every rule profile;
//  - the play-out search is legal and reproducible with a seed;
//  - a quick strength check against the old rule player.

#include "agents/agent.h"
#include "agents/duel.h"
#include "agents/rule_agent.h"
#include "belote/match.h"
#include "belote/observation.h"
#include "belote/sampler.h"
#include "search/endgame.h"
#include "check.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <sstream>

using namespace belote;

namespace {

Deal randomPlayDeal(std::mt19937_64& rng, int cards, const Config& cfg) {
    for (;;) {
        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        Deal d(cfg, int(rng() % 4), deck);
        RandomAgent bidder(rng(), 0.4);
        while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) d.bid(bidder.chooseBid(d));
        if (d.phase() != Phase::kPlay) continue;
        RandomAgent player(rng());
        for (int i = 0; i < cards && d.phase() == Phase::kPlay; ++i) d.play(player.chooseCard(d));
        if (d.phase() == Phase::kPlay) return d;
    }
}

const std::vector<Config>& profiles() {
    static const std::vector<Config> cfgs = [] {
        Config strict = Config::belgian();
        strict.trump_lead_partner_must_overtrump = true;
        strict.must_undertrump = false;
        return std::vector<Config>{Config::belgian(), Config::wikibooks(), Config::tanguyCompat(), strict};
    }();
    return cfgs;
}

void testTableView() {
    std::mt19937_64 rng(21);
    int n = 0;
    for (const Config& cfg : profiles()) {
        for (int i = 0; i < 600; ++i) {
            const Deal d = randomPlayDeal(rng, int(rng() % 31), cfg);
            for (int me = 0; me < kPlayers; ++me) {
                const TableView v = tableView(d, me);
                const Constraints c = inferConstraints(observe(d, me));
                for (int p = 0; p < kPlayers; ++p) {
                    CHECK_EQ(v.excluded[p], c.excluded[p]);
                    CHECK_EQ(v.pinned[p], c.pinned[p]);
                }
                // mayHold never denies a card the real holder has.
                for (int p = 0; p < kPlayers; ++p) {
                    if (p == me) continue;
                    for (Mask m = d.hand(p); m; m &= m - 1) CHECK(v.mayHold(p, lowestCard(m)));
                }
                ++n;
            }
        }
    }
    std::printf("table view: %d views match inferConstraints\n", n);
}

void testLegality() {
    std::mt19937_64 rng(22);
    long decisions = 0;
    std::array<long, int(RuleReason::kCount)> reasons{};
    for (const Config& cfg : profiles()) {
        for (int g = 0; g < 400; ++g) {
            Deal d = randomPlayDeal(rng, 0, cfg);
            while (d.phase() == Phase::kPlay) {
                RuleReason why;
                const int c = rule2ChooseCard(d, &why);
                CHECK(c != kNoCard && (d.legalMoves() & bit(c)));
                ++reasons[std::size_t(why)];
                ++decisions;
                d.play(c);
            }
            CHECK(d.phase() == Phase::kDone);
        }
    }
    std::printf("rule2 legality: %ld decisions\n", decisions);
    for (int r = 0; r < int(RuleReason::kCount); ++r)
        std::printf("  %-34s %6.2f%%\n", ruleReasonName(RuleReason(r)), 100.0 * double(reasons[std::size_t(r)]) / double(decisions));
}

void testRulePlay() {
    std::mt19937_64 rng(23);
    int overruled = 0, n = 0;
    double ms = 0;
    for (int i = 0; i < 120; ++i) {
        const Deal d = randomPlayDeal(rng, int(rng() % 28), profiles()[i % 2]);
        RulePlayOptions o;
        o.worlds = 24;
        o.seed = 1000 + std::uint64_t(i);
        o.lik_eps = (i % 3 == 0) ? 0.5 : 1.0;
        const auto t0 = std::chrono::steady_clock::now();
        const RulePlayResult a = rulePlaySearch(d, o);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const RulePlayResult b = rulePlaySearch(d, o);
        CHECK(d.legalMoves() & bit(a.best_move));
        CHECK_EQ(a.best_move, b.best_move);
        CHECK_EQ(a.rule_move, rule2ChooseCard(d));
        overruled += a.overruled;
        ++n;
    }
    std::printf("ruleplay: %d positions legal & reproducible, %d overruled, %.2f ms/search (24 worlds)\n", n,
                overruled, ms / n);
}

void testAuctionLikelihood() {
    // Deals the taker took in round 1: weighting worlds by the auction must
    // raise the taker's share of the trump jack and fit the real hands better.
    std::mt19937_64 rng(24);
    double uni_j = 0, wt_j = 0, uni_ll = 0, wt_ll = 0;
    int n = 0;
    while (n < 150) {
        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        Deal d(Config::belgian(), int(rng() % 4), deck);
        while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) d.bid(ruleBid(d));
        if (d.phase() != Phase::kPlay || d.bidsMade() >= 4) continue;
        const int obs_seat = nextPlayer(d.taker());
        const Mask jack = bit(cardOf(d.trump(), kJ));
        if (d.hand(obs_seat) & jack) continue;
        const Observation o = observe(d, obs_seat);
        const WorldSampler s(o);
        std::vector<double> ll;
        std::vector<int> has_j;
        for (int k = 0; k < 200; ++k) {
            auto w = s.sample(rng, 500);
            if (!w) continue;
            ll.push_back(auctionLogLik(o, *w, 5.0, 1.0));
            has_j.push_back((w->hands[d.taker()] & jack) != 0);
        }
        const double top = *std::max_element(ll.begin(), ll.end());
        double sw = 0, sj = 0;
        for (std::size_t k = 0; k < ll.size(); ++k) {
            const double x = std::exp(ll[k] - top);
            sw += x;
            sj += x * has_j[k];
        }
        uni_j += double(std::accumulate(has_j.begin(), has_j.end(), 0)) / double(has_j.size());
        wt_j += sj / sw;
        const bool real = (d.hand(d.taker()) & jack) != 0;
        auto logp = [&](double p) { return std::log(std::clamp(real ? p : 1 - p, 1e-3, 1.0)); };
        uni_ll += logp(double(std::accumulate(has_j.begin(), has_j.end(), 0)) / double(has_j.size()));
        wt_ll += logp(sj / sw);
        ++n;
    }
    std::printf("auction weighting: P(taker holds trump J) %.3f uniform -> %.3f weighted; "
                "log-loss %.3f -> %.3f over %d deals\n",
                uni_j / n, wt_j / n, -uni_ll / n, -wt_ll / n, n);
    CHECK(wt_j > uni_j);
    CHECK(wt_ll > uni_ll);
}

void testSimBid() {
    std::mt19937_64 rng(25);
    int n = 0, takes = 0;
    for (int i = 0; i < 60; ++i) {
        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);
        std::shuffle(deck.begin(), deck.end(), rng);
        Deal d(Config::belgian(), int(rng() % 4), deck);
        const int passes = int(rng() % 8);
        for (int k = 0; k < passes && (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2); ++k) d.bid(Bid::pass());
        if (d.phase() != Phase::kBid1 && d.phase() != Phase::kBid2) continue;
        RuleBidOptions o;
        o.worlds = 32;
        o.seed = 77 + std::uint64_t(i);
        const RuleBidResult a = ruleBidSearch(d, o);
        const RuleBidResult b = ruleBidSearch(d, o);
        const auto legal = d.legalBids();
        CHECK(std::find(legal.begin(), legal.end(), a.bid) != legal.end());
        CHECK(a.bid == b.bid);
        CHECK_EQ(a.values.size(), legal.size());
        takes += a.bid.kind != Bid::kPass;
        ++n;
    }
    std::printf("simbid: %d bidding positions legal & reproducible, %d takes\n", n, takes);
}

void testStrength() {
    DuelOptions o;
    o.a = "rule2";
    o.b = "rule";
    o.deals = 1000;
    o.seed = 5;
    const DuelResult r = runDuel(o);
    std::printf("rule2 vs rule: %+.2f +- %.2f units/deal over %d decks\n", r.units, 1.96 * r.units_se, r.decks);
    CHECK(r.units > 0);
}

void playHistory(Deal& d, const char* history) {
    std::istringstream input(history);
    for (std::string card; input >> card;) d.play(parseCard(card));
}

void testLoggedDecisions() {
    const Config cfg = Config::belgian();
    // Match 502, deal 1. Partner has shown JC-QC-KC and owns the diamond
    // trick. Clear the singleton club, then ruff the return with the ten:
    // the only two higher outstanding trumps beat the queen as well.
    Deal returning = Deal::fromPlay(cfg, 0, 1, kSpades, {
        parseCards("8S 9H 10D JD AC AS 9S 7C"), parseCards("JS 8H 7D 8C 10S QS JH 10H"),
        parseCards("7S AH AD 9D 10C 7H KH QD"), parseCards("KS QH 8D KD 9C JC QC KC")
    }, parseCard("JS"), 0);
    playHistory(returning, "JS 7S KS 8S 8H AH QH 9H AD 8D 10D 7D 9D KD JD");
    CHECK_EQ(rule2ChooseCard(returning), parseCard("8C"));
    playHistory(returning, "8C 9C AC");
    CHECK_EQ(rule2ChooseCard(returning), parseCard("10S"));

    // Match 502, deal 12. Partner owns the heart trick and neither opponent
    // has trumps. The spade void covers more outstanding honors.
    Deal exit = Deal::fromPlay(cfg, 3, 0, kDiamonds, {
        parseCards("7D KH KC KD 8C AD 7S 9D"), parseCards("8H 10C 7C 9S JS QS AS QC"),
        parseCards("8D 7H AC JD 10H 9H 10S QD"), parseCards("10D AH 9C 8S JH QH KS JC")
    }, parseCard("8C"), 4);
    playHistory(exit, "7D 8H 8D 10D AH KH 10C 7H 9C KC 7C AC JD 8S KD 9S 10H JH");
    RuleParams quick;
    quick.count_layouts = 0;
    CHECK_EQ(rule2ChooseCard(exit, nullptr, quick), parseCard("7S"));
    auto agent = makeAgent("rule2", 1);
    agent->setMatchState({{91, 98}, 0, true});
    CHECK_EQ(agent->chooseCard(exit), parseCard("7S"));

    DealResult result;
    result.takers = 0;
    result.outcome = Outcome::kMade;
    result.credited = {102, 60};
    CHECK_EQ(matchFinishValue(result, cfg, {91, 98}, 0), -1.0);
    result.credited = {127, 35};
    const double ongoing = matchFinishValue(result, cfg, {91, 98}, 0);
    CHECK(ongoing > 0 && ongoing < 1);
    CHECK_EQ(matchFinishValue(result, cfg, {91, 98}, 1), 1.0);

    // Match 503, deal 3. Cashing the last master trump improves on the
    // heuristic lead across the enumerated layouts. Swapping unseen cards
    // consistently with the history must give exactly the same decision.
    Deal finish = Deal::fromPlay(cfg, 2, 3, kClubs, {
        parseCards("7C 8C 10C AH AS AC 7H 8H"), parseCards("KC 10S 7D 9H 8S JD QD KD"),
        parseCards("JH JS 10D KH KS 8D QH QS"), parseCards("JC 9C AD 10H 7S QC 9D 9S")
    }, parseCard("JC"), 0);
    playHistory(finish, "JC 7C KC JH 9C 8C 10S JS AD 10C 7D 10D AH 9H KH 10H AS 8S KS 7S");
    RuleReason why;
    const int card = rule2ChooseCard(finish, &why);
    CHECK_EQ(card, parseCard("AC"));
    CHECK_EQ(why, RuleReason::kCountedEndgame);
    RulePlayOptions no_search;
    no_search.worlds = 0;
    no_search.ab_handoff_trick = -1;
    CHECK_EQ(rulePlaySearch(finish, no_search).rule_move, rule2ChooseCard(finish, nullptr, quick));
    WorldSampler sampler(observe(finish, finish.currentPlayer()));
    const auto layouts = sampler.enumerate(RuleParams{}.count_layouts);
    CHECK(!layouts.empty());
    const int initial = rule2ChooseCard(finish, nullptr, quick);
    RuleParams too_many;
    too_many.count_layouts = int(sampler.count()) - 1;
    CHECK_EQ(rule2ChooseCard(finish, nullptr, too_many), initial);
    bool improves = false;
    for (const World& w : layouts) {
        const Deal replay = sampler.replay(w);
        CHECK_EQ(replay.blik(), finish.blik());
        CHECK_EQ(replay.bidsMade(), finish.bidsMade());
        CHECK_EQ(rule2ChooseCard(replay), card);
        std::array<int, kCards> value{};
        for (const auto& [move, score] : solveEndgame(replay).move_values) value[move] = score;
        CHECK(value[card] >= value[initial]);
        improves |= value[card] > value[initial];
    }
    CHECK(improves);
}

void testDeclarationBid() {
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    auto place = [&](int pos, const char* card) {
        std::iter_swap(deck.begin() + pos, std::find(deck.begin(), deck.end(), parseCard(card)));
    };
    place(0, "7C"); place(1, "8C"); place(2, "9C");
    place(12, "AD"); place(13, "JH"); place(20, "8H");
    Config cfg = Config::belgian();
    auto agent = makeAgent("rule2", 1);
    CHECK(agent->chooseBid(Deal(cfg, 3, deck)) == Bid::take());
    cfg.toon_enabled = false;
    CHECK(agent->chooseBid(Deal(cfg, 3, deck)) == Bid::pass());

    // Passing in the last seat redeals at the same score and retains the pot.
    Deal last(cfg, 3, deck);
    for (int i = 0; i < 7; ++i) last.bid(Bid::pass());
    RuleBidOptions bids;
    bids.worlds = 4;
    bids.seed = 7;
    bids.match = {{96, 99}, 8, true};
    const RuleBidResult choices = ruleBidSearch(last, bids);
    const auto passed = std::find_if(choices.values.begin(), choices.values.end(),
        [](const auto& entry) { return entry.first == Bid::pass(); });
    CHECK(passed != choices.values.end());
    last.bid(Bid::pass());
    if (passed != choices.values.end()) {
        const double expected = -recordedPoints(kDealPoints, cfg) * matchFinishValue(last.result(), cfg, {96, 99}, 8);
        CHECK(std::abs(passed->second - expected) < 1e-9);
    }
}

}  // namespace

int main() {
    testLoggedDecisions();
    testDeclarationBid();
    testTableView();
    testLegality();
    testRulePlay();
    testAuctionLikelihood();
    testSimBid();
    testStrength();
    return checkResult("test_rule_agent");
}
