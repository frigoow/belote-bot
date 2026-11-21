// Our Bulgarian Belot card player (bg/view.h, bg/agent.h):
//  - the facts read from the public play never exclude a card the player holds;
//  - the real deal replays from the public record, sampled worlds replay too;
//  - the exact endgame solver matches brute force through bg::Round itself;
//  - the rule policy only plays legal cards;
//  - the search player beats the bare policy in mirrored deals.

#include "bg/agent.h"
#include "bg/view.h"
#include "check.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

using namespace bg;

namespace {

// A simple bidder for self-play tests only (our real bids come from elsewhere).
std::uint8_t testBid(const Round& r) {
    const Mask h = r.hand(r.toMove());
    const std::uint8_t avail = r.availableBids();
    double best_v = 0;
    std::uint8_t best = kPass;
    for (int s = 0; s < 4; ++s) {
        const std::uint8_t flag = std::uint8_t(1u << s);
        if (!(avail & flag)) continue;
        const Mask t = h & suitMask(s);
        double v = popcount(t) * 0.8;
        if (t & bit(cardOf(s, belote::kJ))) v += 1.6;
        if (t & bit(cardOf(s, belote::k9))) v += 1.0;
        for (int o = 0; o < 4; ++o)
            if (o != s && (h & bit(cardOf(o, belote::kA)))) v += 0.6;
        if (v >= 4.2 && v > best_v) best_v = v, best = flag;
    }
    if (avail & kAllTrumps) {
        double v = 0;
        for (int s = 0; s < 4; ++s) {
            if (h & bit(cardOf(s, belote::kJ))) v += 1.5;
            if (h & bit(cardOf(s, belote::k9))) v += 0.8;
        }
        if (v >= 4.0 && v + 0.5 > best_v) best_v = v + 0.5, best = kAllTrumps;
    }
    if (avail & kNoTrumps) {
        double v = 0;
        for (int s = 0; s < 4; ++s) {
            if (h & bit(cardOf(s, belote::kA))) v += 1.4;
            if (h & bit(cardOf(s, belote::k10))) v += 0.5;
        }
        if (v >= 4.4 && v > best_v) best_v = v, best = kNoTrumps;
    }
    return best;
}

std::array<int, kCards> randomDeal(std::mt19937_64& rng) {
    std::array<int, kCards> d;
    std::iota(d.begin(), d.end(), 0);
    std::shuffle(d.begin(), d.end(), rng);
    return d;
}

// A deal advanced by `cards` random plays (random declarations and belotes).
std::optional<Round> randomPosition(std::mt19937_64& rng, int cards) {
    Round r(randomDeal(rng), int(rng() % 4));
    while (r.decision() == Round::Decision::kBid) {
        const std::uint8_t avail = r.availableBids();
        std::vector<std::uint8_t> opts{kPass};
        for (int b = 0; b < 8; ++b)
            if (avail & (1u << b)) opts.push_back(std::uint8_t(1u << b));
        r.bid(rng() % 3 ? std::uint8_t(kPass) : opts[rng() % opts.size()]);
    }
    if (r.finished()) return std::nullopt;
    for (int i = 0; !r.finished();) {
        if (r.decision() == Round::Decision::kAnnounce) {
            std::vector<Announce> chosen;
            for (const Announce& a : r.offeredAnnounces())
                if (rng() % 5) chosen.push_back(a);
            r.declare(chosen);
            continue;
        }
        if (i++ >= cards) break;
        Mask legal = r.legalMoves();
        int k = int(rng() % std::uint64_t(popcount(legal)));
        while (k--) legal &= legal - 1;
        r.play(lowestCard(legal), rng() % 4 != 0);
    }
    if (r.finished()) return std::nullopt;
    return r;
}

void testFactsAndReplay() {
    std::mt19937_64 rng(31);
    int positions = 0, accepted = 0, tried = 0;
    for (int i = 0; i < 1500; ++i) {
        auto r = randomPosition(rng, int(rng() % 30));
        if (!r || r->decision() != Round::Decision::kPlay) continue;
        const int me = r->toMove();
        const PublicState ps = publicState(*r, me);
        const Facts f = inferFacts(ps);
        Hands real{};
        for (int p = 0; p < kPlayers; ++p) {
            real[std::size_t(p)] = r->hand(p);
            CHECK(!(f.excluded[std::size_t(p)] & r->hand(p)));
            CHECK((f.pinned[std::size_t(p)] & r->hand(p)) == f.pinned[std::size_t(p)]);
        }
        CHECK_EQ(ps.toMove(), me);
        const DealSampler s(ps);
        CHECK(s.admits(real));
        const auto rep = s.replay(real);
        CHECK(rep.has_value());
        if (rep) {
            CHECK_EQ(rep->legalMoves(), r->legalMoves());
            CHECK_EQ(rep->toMove(), me);
        }
        for (int k = 0; k < 5; ++k) {
            const auto h = s.sampleUnchecked(rng);
            CHECK(h.has_value());
            if (!h) continue;
            CHECK(s.admits(*h));
            ++tried;
            accepted += s.replay(*h).has_value();
        }
        ++positions;
    }
    std::printf("facts/replay: %d positions, sampled-world replay acceptance %.3f\n", positions,
                double(accepted) / std::max(1, tried));
}

// Brute force through bg::Round (belote claimed whenever the engine allows it).
int bruteForce(const Round& r, int team) {
    if (r.finished()) {
        const RoundResult& res = r.result();
        return team == 0 ? res.sn - res.ew : res.ew - res.sn;
    }
    const bool mine = teamOf(r.toMove()) == team;
    int best = mine ? -1000000 : 1000000;
    for (Mask m = r.legalMoves(); m; m &= m - 1) {
        Round n = r;
        n.play(lowestCard(m), true);
        const int v = bruteForce(n, team);
        best = mine ? std::max(best, v) : std::min(best, v);
    }
    return best;
}

void testEndgame() {
    std::mt19937_64 rng(32);
    int checked = 0;
    long nodes = 0;
    double ms = 0;
    for (int i = 0; checked < 300 && i < 20000; ++i) {
        auto r = randomPosition(rng, 20 + int(rng() % 8));  // tricks 6-7
        if (!r || r->decision() != Round::Decision::kPlay || r->trickNumber() < 2) continue;
        const int team = int(rng() % 2);
        const auto t0 = std::chrono::steady_clock::now();
        const EndgameResult e = solveEndgame(*r, team);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        nodes += e.nodes;
        for (const auto& [card, value] : e.move_values) {
            Round n = *r;
            n.play(card, true);
            CHECK_EQ(value, bruteForce(n, team));
        }
        ++checked;
    }
    // Timing at trick 5 (16 cards left).
    double ms5 = 0;
    int n5 = 0;
    for (int i = 0; n5 < 100 && i < 20000; ++i) {
        auto r = randomPosition(rng, 16);
        if (!r || r->decision() != Round::Decision::kPlay || r->trickNumber() != 5) continue;
        const auto t0 = std::chrono::steady_clock::now();
        solveEndgame(*r, 0);
        ms5 += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++n5;
    }
    std::printf("endgame: %d positions match brute force (%.3f ms, %ld nodes avg); trick 5: %.2f ms/solve\n", checked,
                ms / checked, nodes / std::max(1, checked), ms5 / std::max(1, n5));
}

void testPolicyLegal() {
    std::mt19937_64 rng(33);
    long decisions = 0;
    for (int g = 0; g < 2000; ++g) {
        Round r(randomDeal(rng), g % 4);
        while (r.decision() == Round::Decision::kBid) r.bid(testBid(r));
        while (!r.finished()) {
            if (r.decision() == Round::Decision::kAnnounce) {
                r.declare(chooseDeclarations(r.offeredAnnounces()));
                continue;
            }
            const int c = rulePolicy(r);
            CHECK(r.legalMoves() & bit(c));
            r.play(c, true);
            ++decisions;
        }
    }
    std::printf("policy: %ld decisions legal\n", decisions);
}

// Mirrored deals: X plays for one team, Y for the other, then swapped.
double playDeal(const std::array<int, kCards>& deck, int first, bool x_is_team0, const SearchOptions& xo,
                std::uint64_t seed, double* ms_x, long* cards_x) {
    Round r(deck, first);
    while (r.decision() == Round::Decision::kBid) r.bid(testBid(r));
    if (r.finished()) return 0;
    std::mt19937_64 rng(seed);
    while (!r.finished()) {
        if (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
            continue;
        }
        const int seat = r.toMove();
        const bool x = (teamOf(seat) == 0) == x_is_team0;
        int card;
        if (x) {
            SearchOptions o = xo;
            o.seed = rng() | 1;
            const auto t0 = std::chrono::steady_clock::now();
            card = searchCard(publicState(r, seat), o).card;
            *ms_x += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++*cards_x;
        } else {
            card = rulePolicy(r);
        }
        r.play(card, true);
    }
    const RoundResult& res = r.result();
    const int m = res.sn - res.ew;
    return x_is_team0 ? m : -m;
}

void testSearchStrength() {
    std::mt19937_64 rng(34);
    SearchOptions o;
    o.worlds = 24;
    std::vector<double> pairs;
    double ms = 0;
    long cards = 0;
    for (int i = 0; i < 300; ++i) {
        const auto deck = randomDeal(rng);
        const int first = i % 4;
        const std::uint64_t seed = rng();
        const double a = playDeal(deck, first, true, o, seed, &ms, &cards);
        const double b = playDeal(deck, first, false, o, seed, &ms, &cards);
        pairs.push_back(0.5 * (a + b));
    }
    double mean = 0, var = 0;
    for (double x : pairs) mean += x / double(pairs.size());
    for (double x : pairs) var += (x - mean) * (x - mean) / double(pairs.size() - 1);
    const double ci = 1.96 * std::sqrt(var / double(pairs.size()));
    std::printf("search (24 worlds) vs policy: %+.2f +- %.2f game points/deal over %zu mirrored deals, %.2f ms/card\n",
                mean, ci, pairs.size(), ms / std::max(1L, cards));
    CHECK(mean > 0);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    testFactsAndReplay();
    testEndgame();
    testPolicyLegal();
    testSearchStrength();
    return checkResult("test_bg_agent");
}
