// The world sampler against brute force.
//
// Ground truth: a world (hidden cards dealt to the other three players) is
// consistent if replaying the deal from it through the engine reproduces
// every public event. For random positions:
//  1. the real deal is always admitted by the per-card constraints;
//  2. the sampler's worlds are exactly the brute-force consistent worlds
//     (so the per-card inference never wrongly excludes a world);
//  3. with no announced/shown toon to check, every world the per-card table
//     counts is consistent (the inference is complete);
//  4. sampling is uniform over the consistent worlds.

#include "belote/deal.h"
#include "belote/observation.h"
#include "belote/sampler.h"
#include "check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace belote;

namespace {

std::array<int, kCards> shuffled(std::mt19937_64& rng) {
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    return deck;
}

int randomCard(Mask m, std::mt19937_64& rng) {
    int k = std::uniform_int_distribution<int>(0, popcount(m) - 1)(rng);
    while (k--) m &= m - 1;
    return lowestCard(m);
}

// A random deal stopped after `cards` plays, with random bids and declare choices.
bool randomPosition(const Config& cfg, std::mt19937_64& rng, int cards, Deal& out) {
    Deal d(cfg, int(rng() % 4), shuffled(rng));
    std::bernoulli_distribution take(0.35), declare(0.75);
    while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) {
        const auto bids = d.legalBids();
        d.bid(take(rng) ? bids[1 + rng() % (bids.size() - 1)] : Bid::pass());
    }
    if (d.phase() != Phase::kPlay) return false;
    for (int p = 0; p < kPlayers; ++p)
        if (d.canSetDeclare(p)) d.setDeclare(p, declare(rng));
    for (int i = 0; i < cards && d.phase() == Phase::kPlay; ++i) d.play(randomCard(d.legalMoves(), rng));
    if (d.phase() != Phase::kPlay) return false;
    out = d;
    return true;
}

using Key = std::array<Mask, kPlayers>;

void bruteForce(const WorldSampler& s, int i, const std::vector<int>& cards, const std::array<int, 3>& others,
                std::array<int, 3>& left, World& w, std::set<Key>& out) {
    if (i == int(cards.size())) {
        if (s.consistent(w)) out.insert(w.hands);
        return;
    }
    for (int k = 0; k < 3; ++k) {
        if (!left[k]) continue;
        --left[k];
        w.hands[others[k]] |= bit(cards[i]);
        bruteForce(s, i + 1, cards, others, left, w, out);
        w.hands[others[k]] &= ~bit(cards[i]);
        ++left[k];
    }
}

std::set<Key> bruteForceWorlds(const WorldSampler& s) {
    const Observation& o = s.observation();
    std::vector<int> cards;
    for (Mask m = ~(o.hand | o.played()); m; m &= m - 1) cards.push_back(lowestCard(m));
    std::array<int, 3> others, left;
    for (int k = 0; k < 3; ++k) {
        others[k] = (o.observer + 1 + k) % kPlayers;
        left[k] = o.handSize(others[k]);
    }
    World w;
    w.hands[o.observer] = o.hand;
    std::set<Key> out;
    bruteForce(s, 0, cards, others, left, w, out);
    return out;
}

World realWorld(const Deal& d) {
    World w;
    for (int p = 0; p < kPlayers; ++p) w.hands[p] = d.hand(p);
    return w;
}

std::vector<Config> configs() {
    Config belgian = Config::belgian();
    Config wikibooks = Config::wikibooks();
    Config loose = Config::belgian();
    loose.must_undertrump = false;
    loose.trump_lead_partner_must_overtrump = true;
    loose.toon_losers_show = true;
    Config early = Config::belgian();
    early.toon_show_trick = 1;
    early.toon_optional = false;
    early.carre_low_points = 50;
    return {belgian, wikibooks, loose, early};
}

void testRealWorldAdmitted(int deals) {
    std::mt19937_64 rng(11);
    long positions = 0;
    for (const Config& cfg : configs()) {
        for (int i = 0; i < deals; ++i) {
            Deal g(cfg, int(rng() % 4), shuffled(rng));
            std::bernoulli_distribution take(0.35), declare(0.75);
            while (g.phase() == Phase::kBid1 || g.phase() == Phase::kBid2) {
                const auto bids = g.legalBids();
                g.bid(take(rng) ? bids[1 + rng() % (bids.size() - 1)] : Bid::pass());
            }
            if (g.phase() != Phase::kPlay) continue;
            for (int p = 0; p < kPlayers; ++p)
                if (g.canSetDeclare(p)) g.setDeclare(p, declare(rng));
            while (g.phase() == Phase::kPlay) {
                for (int obs = 0; obs < kPlayers; ++obs) {
                    const WorldSampler s(observe(g, obs));
                    const World real = realWorld(g);
                    CHECK(s.admits(real));
                    CHECK(s.consistent(real));
                    ++positions;
                }
                g.play(randomCard(g.legalMoves(), rng));
            }
        }
    }
    std::printf("real world admitted: %ld positions\n", positions);
}

void testAgainstBruteForce(int per_config) {
    std::mt19937_64 rng(22);
    long positions = 0, checked = 0, worlds = 0;
    double min_accept = 1.0, sum_accept = 0;
    for (const Config& cfg : configs()) {
        for (int i = 0; i < per_config;) {
            Deal d(cfg, 0, shuffled(rng));
            const int cards = 16 + int(rng() % 14);  // trick 5 to trick 8
            if (!randomPosition(cfg, rng, cards, d)) continue;
            ++i;
            const int obs = int(rng() % 4);
            const WorldSampler s(observe(d, obs));
            const std::set<Key> truth = bruteForceWorlds(s);
            const auto listed = s.enumerate(1u << 20);
            std::set<Key> mine;
            for (const World& w : listed) mine.insert(w.hands);
            CHECK(truth == mine);
            CHECK(truth.count(realWorld(d).hands) == 1);
            if (!s.needsCheck()) {
                CHECK_EQ(s.count(), truth.size());
            } else {
                ++checked;
                const double accept = double(truth.size()) / double(s.count());
                min_accept = std::min(min_accept, accept);
                sum_accept += accept;
            }
            ++positions;
            worlds += long(truth.size());
        }
    }
    std::printf("brute force: %ld positions (%ld with toon checks), %ld consistent worlds; "
                "replay acceptance mean %.3f, min %.3f\n",
                positions, checked, worlds, sum_accept / std::max(1L, checked), min_accept);
}

void testUniform() {
    std::mt19937_64 rng(33);
    int tested = 0;
    for (int attempt = 0; attempt < 5000 && tested < 12; ++attempt) {
        const Config cfg = Config::belgian();
        Deal d(cfg, 0, shuffled(rng));
        if (!randomPosition(cfg, rng, 18 + int(rng() % 8), d)) continue;
        const int obs = int(rng() % 4);
        const WorldSampler s(observe(d, obs));
        const auto all = s.enumerate(400);
        if (all.size() < 5 || all.size() > 400) continue;
        // Alternate between positions with and without the toon check.
        if (s.needsCheck() != (tested % 2 == 0)) continue;
        std::map<Key, int> freq;
        for (const World& w : all) freq[w.hands] = 0;
        const int n = 300 * int(all.size());
        for (int k = 0; k < n; ++k) {
            auto w = s.sample(rng);
            CHECK(w.has_value());
            if (!w) break;
            auto it = freq.find(w->hands);
            CHECK(it != freq.end());
            if (it != freq.end()) ++it->second;
        }
        const double expect = n / double(all.size());
        double chi2 = 0;
        for (const auto& [k, f] : freq) chi2 += (f - expect) * (f - expect) / expect;
        const double df = double(all.size()) - 1;
        CHECK(chi2 < df + 6 * std::sqrt(2 * df) + 10);
        ++tested;
    }
    CHECK(tested >= 10);
    std::printf("uniformity: %d positions\n", tested);
}

void testEarlyAcceptance() {
    // Rejection cost at trick 2, where announcements are checked but nothing
    // is shown yet.
    std::mt19937_64 rng(44);
    long samples = 0, failures = 0, positions = 0;
    for (int i = 0; i < 400; ++i) {
        const Config cfg = Config::belgian();
        Deal d(cfg, 0, shuffled(rng));
        if (!randomPosition(cfg, rng, 5, d)) continue;
        const WorldSampler s(observe(d, d.currentPlayer()));
        if (!s.needsCheck()) continue;
        ++positions;
        for (int k = 0; k < 20; ++k) {
            ++samples;
            if (!s.sample(rng)) ++failures;
        }
    }
    std::printf("trick-2 samples with announcements: %ld positions, %ld samples, %ld gave up\n", positions,
                samples, failures);
    CHECK(failures * 100 <= samples);
}

}  // namespace

int main() {
    testRealWorldAdmitted(150);
    testAgainstBruteForce(150);
    testUniform();
    testEarlyAcceptance();
    return checkResult("test_sampler");
}
