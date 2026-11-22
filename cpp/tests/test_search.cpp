// ISMCTS adapter, exact AB endgame solver, shared-world rollouts, and harness:
//  - every determinization the search sees is consistent with the observer's
//    knowledge (and keeps the observer's own hand);
//  - the exact AB endgame solver matches unpruned brute-force minimax on every
//    legal move across tricks 6-8 under belgian, wikibooks, and raw rules;
//  - shared-world rollouts and all harness modes (play, bid, mixed, match)
//    behave as expected.

#include "agents/agent.h"
#include "agents/duel.h"
#include "belote/match.h"
#include "belote/sampler.h"
#include "check.h"
#include "search/belote_game.h"
#include "search/endgame.h"
#include "search/ismcts.h"
#include "search/rollout.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>

using namespace belote;

namespace {

Deal randomPlayDeal(std::mt19937_64& rng, int cards, const Config& cfg = Config::belgian()) {
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

void testDeterminizations() {
    std::mt19937_64 rng(5);
    long clones = 0;
    for (int i = 0; i < 300; ++i) {
        const Deal d = randomPlayDeal(rng, int(rng() % 28));
        const int obs = d.currentPlayer();
        const WorldSampler s(observe(d, obs));
        const BeloteGame game(d);
        for (int k = 0; k < 10; ++k) {
            auto clone = game.cloneAndRandomise(unsigned(obs));
            const Deal& c = static_cast<const BeloteGame&>(*clone).deal();
            World w;
            for (int p = 0; p < kPlayers; ++p) w.hands[p] = c.hand(p);
            CHECK_EQ(w.hands[obs], d.hand(obs));
            CHECK(s.admits(w) && s.consistent(w));
            CHECK_EQ(c.currentPlayer(), d.currentPlayer());
            CHECK_EQ(c.legalMoves(), d.legalMoves());  // the observer's own options
            ++clones;
        }
    }
    CHECK_EQ(BeloteGame::fallbacks(), 0);
    std::printf("determinizations: %ld checked\n", clones);
}

void testSearchMoves() {
    std::mt19937_64 rng(6);
    for (int i = 0; i < 40; ++i) {
        Deal d = randomPlayDeal(rng, int(rng() % 28));
        const SearchResult r = ismctsSearch(d, {200, 1});
        CHECK(d.legalMoves() & bit(r.best_move));
        if (popcount(d.legalMoves()) > 1) CHECK(r.total_visits > 0);
        const SearchResult p = ismctsSearch(d, {400, 4});
        CHECK(d.legalMoves() & bit(p.best_move));
        const RolloutResult rr = rolloutSearch(d, {16, -1, std::uint64_t(i + 1)});
        CHECK(d.legalMoves() & bit(rr.best_move));
    }
}

// Reference unpruned minimax directly on Deal::play() + recordDeal()/credited.
int bruteMinimax(const Deal& d, int root_team, EndgameObjective obj) {
    if (d.phase() == Phase::kDone) {
        if (obj == EndgameObjective::kRaw)
            return d.result().credited[root_team] - d.result().credited[1 - root_team];
        const RecordedDeal rec = recordDeal(d.result(), d.config());
        return rec.scores[root_team] - rec.scores[1 - root_team];
    }
    const bool maxing = teamOf(d.currentPlayer()) == root_team;
    int best = maxing ? -1000000 : 1000000;
    for (Mask m = d.legalMoves(); m; m &= m - 1) {
        Deal next = d;
        next.play(lowestCard(m));
        const int v = bruteMinimax(next, root_team, obj);
        best = maxing ? std::max(best, v) : std::min(best, v);
    }
    return best;
}

void testEndgameExact() {
    std::mt19937_64 rng(77);
    const std::vector<Config> cfgs = {Config::belgian(), Config::wikibooks(), Config::tanguyCompat()};
    int checked = 0;
    for (const Config& cfg : cfgs) {
        const EndgameObjective obj =
            cfg.score_mode == ScoreMode::kRaw ? EndgameObjective::kRaw : EndgameObjective::kUnits;
        for (int i = 0; i < 40; ++i) {
            // Tricks 6-8: 20 to 30 cards already played (2 to 12 cards left).
            const int cards = 20 + int(rng() % 11);
            const Deal d = randomPlayDeal(rng, cards, cfg);
            const int root_team = teamOf(d.currentPlayer());
            const EndgameResult er = solveEndgame(d, {obj, 16});
            CHECK(d.legalMoves() & bit(er.best_move));
            CHECK_EQ(int(er.move_values.size()), popcount(d.legalMoves()));
            int best_truth = -1000000;
            for (const auto& [card, val] : er.move_values) {
                Deal next = d;
                next.play(card);
                const int truth = bruteMinimax(next, root_team, obj);
                CHECK_EQ(val, truth);
                best_truth = std::max(best_truth, truth);
            }
            CHECK_EQ(er.value, best_truth);
            ++checked;
        }
    }
    // Benchmark trick 5 (16 cards played, 4 tricks = 16 cards left) speed.
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t total_nodes = 0;
    constexpr int kBench = 20;
    for (int i = 0; i < kBench; ++i) {
        const Deal d = randomPlayDeal(rng, 16, Config::belgian());
        const EndgameResult er = solveEndgame(d);
        CHECK(d.legalMoves() & bit(er.best_move));
        total_nodes += er.nodes;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("endgame exact: %d positions matched brute-force; trick-5 (16 cards left): %.2f ms/solve (%.0f nodes)\n",
                checked, ms / kBench, double(total_nodes) / kBench);
}

void testEndgameEarlyScoring() {
    // Verifies early-deal endgame root setup (tricks 0-3): mandatory toon
    // announcing (toon_optional=false), optional announcing, pending show
    // (toon_show_trick=1/3, toon_losers_show=true/false), and belot_forfeited
    // at root after trump Q played before K under wikibooks().
    std::mt19937_64 rng(88);
    Config mand = Config::belgian();
    mand.toon_optional = false;
    Config losers = Config::belgian();
    losers.toon_losers_show = true;
    Config early_show = Config::belgian();
    early_show.toon_show_trick = 1;
    early_show.toon_optional = false;
    early_show.carre_low_points = 50;
    const std::vector<Config> cfgs = {Config::belgian(), Config::wikibooks(), Config::tanguyCompat(),
                                      mand, losers, early_show};

    int checked = 0;
    for (const Config& cfg : cfgs) {
        const EndgameObjective obj =
            cfg.score_mode == ScoreMode::kRaw ? EndgameObjective::kRaw : EndgameObjective::kUnits;
        for (int i = 0; i < 80; ++i) {
            // Stop at tricks 0..3 (0 to 13 cards already played).
            const int stop_cards = int(rng() % 14);
            Deal d = randomPlayDeal(rng, stop_cards, cfg);
            if (d.phase() != Phase::kPlay) continue;

            // Play a random legal continuation to completion through Deal.
            Deal sim = d;
            std::vector<int> line;
            RandomAgent ply(rng());
            while (sim.phase() == Phase::kPlay) {
                const int c = ply.chooseCard(sim);
                line.push_back(c);
                sim.play(c);
            }
            const int truth = (obj == EndgameObjective::kRaw)
                                  ? (sim.result().credited[0] - sim.result().credited[1])
                                  : (recordDeal(sim.result(), cfg).scores[0] - recordDeal(sim.result(), cfg).scores[1]);
            const int got = evaluateEndgameLine(d, line, obj);
            CHECK_EQ(got, truth);
            ++checked;
        }
    }

    // Explicit regression: trick 0 with toon_optional=false (seat 1 has a vierde
    // and hasn't played yet) + wikibooks belot_forfeited at root (Q played first).
    {
        const std::array<Mask, kPlayers> hands = {
            parseCards("7H 8H 9H 10H JH QH KH AH"), parseCards("7C 8C 9C 10C JC QC KC AC"),
            parseCards("7D 8D 9D 10D JD QD KD AD"), parseCards("7S 8S 9S 10S JS QS KS AS")};
        // 1) Mandatory declaring at trick 0 (after only seat 0 played):
        Deal d_mand = Deal::fromPlay(mand, 3, 0, kHearts, hands);
        d_mand.play(parseCard("7H"));  // seat 0 played; seats 1,2,3 haven't played yet
        Deal sim = d_mand;
        std::vector<int> line;
        while (sim.phase() == Phase::kPlay) {
            const int c = lowestCard(sim.legalMoves());
            line.push_back(c);
            sim.play(c);
        }
        const RecordedDeal rec = recordDeal(sim.result(), mand);
        CHECK_EQ(evaluateEndgameLine(d_mand, line, EndgameObjective::kUnits), rec.scores[0] - rec.scores[1]);

        // 2) Wikibooks belot forfeited BEFORE endgame root (Q played first):
        Deal d_wb = Deal::fromPlay(Config::wikibooks(), 3, 0, kHearts, hands);
        d_wb.play(parseCard("QH"));  // forfeits belot!
        Deal sim_wb = d_wb;
        std::vector<int> line_wb;
        while (sim_wb.phase() == Phase::kPlay) {
            const int c = lowestCard(sim_wb.legalMoves());
            line_wb.push_back(c);
            sim_wb.play(c);
        }
        CHECK_EQ(sim_wb.result().belot_team, -1);
        const RecordedDeal rec_wb = recordDeal(sim_wb.result(), Config::wikibooks());
        CHECK_EQ(evaluateEndgameLine(d_wb, line_wb, EndgameObjective::kUnits), rec_wb.scores[0] - rec_wb.scores[1]);
    }
    std::printf("endgame early-deal scoring: %d lines matched Deal::play()\n", checked);
}

void testConfigSummaryComplete() {
    const Config base = Config::belgian();
    const std::string base_s = configSummary(base);
    auto checkDiff = [&](auto mutate) {
        Config c = base;
        mutate(c);
        CHECK(configSummary(c) != base_s);
    };
    checkDiff([](Config& c) { c.redeal_next_dealer = !c.redeal_next_dealer; });
    checkDiff([](Config& c) { c.round2_allow_blik_suit = !c.round2_allow_blik_suit; });
    checkDiff([](Config& c) { c.trump_lead_partner_must_overtrump = !c.trump_lead_partner_must_overtrump; });
    checkDiff([](Config& c) { c.must_undertrump = !c.must_undertrump; });
    checkDiff([](Config& c) { c.toon_enabled = !c.toon_enabled; });
    checkDiff([](Config& c) { c.carre_low_points = 50; });
    checkDiff([](Config& c) { c.toon_optional = !c.toon_optional; });
    checkDiff([](Config& c) { c.toon_show_trick = 1; });
    checkDiff([](Config& c) { c.toon_losers_show = !c.toon_losers_show; });
    checkDiff([](Config& c) { c.belot_queen_first_forfeits = !c.belot_queen_first_forfeits; });
    checkDiff([](Config& c) { c.tie_rule = TieRule::kTakersMake; });
    checkDiff([](Config& c) { c.score_mode = ScoreMode::kUnits; });
    checkDiff([](Config& c) { c.ronde_points = 250; });
    checkDiff([](Config& c) { c.match_target = 1001; });
    checkDiff([](Config& c) { c.match_win_margin = 0; });
}

void testSearchReproducibility() {
    for (const char* spec : {"rollout:16", "ismcts:60"}) {
        DuelOptions o1;
        o1.a = spec;
        o1.b = "rule";
        o1.deals = 24;
        o1.seed = 42;
        o1.threads = 1;
        DuelOptions o4 = o1;
        o4.threads = 4;
        const DuelResult r1 = runDuel(o1);
        const DuelResult r4 = runDuel(o4);
        CHECK_EQ(r1.decks, r4.decks);
        CHECK_EQ(r1.units, r4.units);
        CHECK_EQ(r1.raw, r4.raw);
    }
    // Also verify MatchDuel reproducibility across thread counts.
    MatchDuelOptions m1;
    m1.a = "rollout:8";
    m1.b = "rule";
    m1.matches = 8;
    m1.seed = 77;
    m1.threads = 1;
    MatchDuelOptions m4 = m1;
    m4.threads = 4;
    const MatchDuelResult mr1 = runMatchDuel(m1);
    const MatchDuelResult mr4 = runMatchDuel(m4);
    CHECK_EQ(mr1.win_rate, mr4.win_rate);
    CHECK_EQ(mr1.score_margin, mr4.score_margin);
    std::printf("reproducibility: rollout & ismcts identical across 1 vs 4 threads\n");
}

void testHarnessModes() {
    // 1. Card-play ladder sanity.
    DuelOptions o;
    o.deals = 120;
    o.threads = 8;
    o.a = "ismcts:300";
    o.b = "random";
    const DuelResult r = runDuel(o);
    std::printf("ismcts:300 vs random: %+.2f +- %.2f units/deal over %d decks\n", r.units, 1.96 * r.units_se,
                r.decks);
    CHECK(r.units - 3 * r.units_se > 0);
    CHECK(!r.config_summary.empty());

    // 2. Symmetry sanity.
    o.a = "random";
    o.b = "random";
    const DuelResult same = runDuel(o);
    CHECK(std::abs(same.units) < 4 * same.units_se + 0.5);

    // 3. Bidding-only mode: rule bidder vs random bidder (fixed rule play).
    DuelOptions bid_o;
    bid_o.deals = 200;
    bid_o.threads = 8;
    bid_o.a = "rule";
    bid_o.b = "random";
    bid_o.bidder = "self";
    bid_o.player = "rule";
    const DuelResult br = runDuel(bid_o);
    std::printf("bid-only rule vs random (fixed rule play): %+.2f +- %.2f units/deal over %d decks\n", br.units,
                1.96 * br.units_se, br.decks);
    CHECK(br.units - 3 * br.units_se > 0);

    // 4. Mixed-partner mode: (rule + rule) vs (random + rule).
    DuelOptions mix_o;
    mix_o.deals = 120;
    mix_o.threads = 8;
    mix_o.a = "rule";
    mix_o.b = "random";
    mix_o.partner = "rule";
    const DuelResult mr = runDuel(mix_o);
    CHECK(mr.units - 3 * mr.units_se > 0);

    // 5. Full 101-point match mode: rule vs random.
    MatchDuelOptions mo;
    mo.matches = 60;
    mo.threads = 8;
    mo.a = "rule";
    mo.b = "random";
    const MatchDuelResult mat = runMatchDuel(mo);
    std::printf("match-101 rule vs random: %.1f%% +- %.1f pp (%+.1f units/match, %.1f deals/match)\n",
                100.0 * mat.win_rate, 196.0 * mat.win_rate_se, mat.score_margin, mat.mean_deals);
    CHECK(mat.win_rate > 0.75);
    CHECK(!mat.config_summary.empty());
}

}  // namespace

int main() {
    testDeterminizations();
    testSearchMoves();
    testEndgameExact();
    testEndgameEarlyScoring();
    testConfigSummaryComplete();
    testSearchReproducibility();
    testHarnessModes();
    return checkResult("test_search");
}
