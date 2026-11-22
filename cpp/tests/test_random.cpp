// Randomised tests:
//  1. Invariants over many random matches under the Belgian rules.
//  2. A cross-check against Tanguy's engine (tanguy_oracle.h): with
//     Config::tanguyCompat(), every legal-move set and every final score
//     must agree on random deals.

#include "belote/deal.h"
#include "belote/match.h"
#include "check.h"
#include "tanguy_oracle.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace belote;

namespace {

std::array<int, kCards> shuffled(std::mt19937& rng) {
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    return deck;
}

int randomCard(Mask m, std::mt19937& rng) {
    int k = std::uniform_int_distribution<int>(0, popcount(m) - 1)(rng);
    while (k--) m &= m - 1;
    return lowestCard(m);
}

// Bids: take/choose with probability p_take, otherwise pass.
void randomBids(Deal& d, std::mt19937& rng, double p_take = 0.3) {
    std::bernoulli_distribution take(p_take);
    while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) {
        const auto bids = d.legalBids();
        if (take(rng)) d.bid(bids[std::uniform_int_distribution<int>(1, int(bids.size()) - 1)(rng)]);
        else d.bid(Bid::pass());
    }
}

void checkDealInvariants(const Deal& d) {
    const DealResult& r = d.result();
    if (r.outcome == Outcome::kAllPassed) return;
    const int card_total = r.card_points[0] + r.card_points[1];
    CHECK(card_total == kDealPoints || (r.ronde_team >= 0 && card_total == 252));
    for (int p = 0; p < kPlayers; ++p) CHECK_EQ(d.hand(p), 0u);
    CHECK(r.toon_points[0] == 0 || r.toon_points[1] == 0);
    // Only one team shows (toon_losers_show = false), and only real toons are announced.
    for (int p = 0; p < kPlayers; ++p) {
        if (d.shown(p)) CHECK(d.announced(p) && !(d.shown((p + 1) % kPlayers) || d.shown((p + 3) % kPlayers)));
        if (d.announced(p)) CHECK(announcementOf(d.toons(p), r.trump).kind == d.announcement(p).kind);
    }
    // Complement recording: the two teams' units (plus any held) are the deal's units.
    const RecordedDeal rec = recordDeal(r, d.config());
    CHECK_EQ(rec.scores[0] + rec.scores[1] + rec.held, (r.credited[0] + r.credited[1] + r.held) / 10);
    switch (r.outcome) {
        case Outcome::kMade:
            CHECK(r.totals[r.takers] > r.totals[1 - r.takers] || r.ronde_team == r.takers);
            CHECK(r.credited == r.totals);
            break;
        case Outcome::kNat:
            CHECK_EQ(r.credited[r.takers], r.belot_points[r.takers]);
            CHECK_EQ(r.credited[1 - r.takers],
                     card_total + r.toon_points[0] + r.toon_points[1] + r.belot_points[1 - r.takers]);
            break;
        case Outcome::kHanging:
            CHECK_EQ(r.totals[0], r.totals[1]);
            CHECK_EQ(r.held, r.totals[r.takers]);
            break;
        default:
            CHECK(false);
    }
}

void testRandomMatches(int matches) {
    std::mt19937 rng(12345);
    const Config cfg = Config::belgian();
    std::bernoulli_distribution declare(0.8);
    long deals = 0, outcomes[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < matches; ++i) {
        Match match(cfg, i % kPlayers);
        while (!match.finished()) {
            Deal d(cfg, match.dealer(), shuffled(rng));
            randomBids(d, rng);
            if (d.phase() == Phase::kPlay) {
                Mask all = 0;
                for (int p = 0; p < kPlayers; ++p) {
                    CHECK_EQ(popcount(d.hand(p)), 8);
                    all |= d.hand(p);
                    if (d.canSetDeclare(p)) d.setDeclare(p, declare(rng));
                }
                CHECK_EQ(all, 0xFFFFFFFFu);
            }
            while (d.phase() == Phase::kPlay) {
                const Mask legal = d.legalMoves();
                CHECK(legal != 0);
                CHECK((legal & ~d.hand(d.currentPlayer())) == 0);
                d.play(randomCard(legal, rng));
            }
            checkDealInvariants(d);
            ++outcomes[int(d.result().outcome)];
            ++deals;
            match.record(d.result());
            CHECK(match.dealsRecorded() < 500);
            if (match.dealsRecorded() >= 500) break;
        }
    }
    std::printf("random matches: %d matches, %ld deals (made %ld, nat %ld, hanging %ld, all passed %ld)\n",
                matches, deals, outcomes[int(Outcome::kMade)], outcomes[int(Outcome::kNat)],
                outcomes[int(Outcome::kHanging)], outcomes[int(Outcome::kAllPassed)]);
    CHECK(outcomes[int(Outcome::kHanging)] > 0);
}

std::vector<int> cardsOf(Mask m) {
    std::vector<int> v;
    for (; m; m &= m - 1) v.push_back(lowestCard(m));
    return v;
}

void testTanguyCrossCheck(int deals) {
    std::mt19937 rng(777);
    const Config cfg = Config::tanguyCompat();
    long compared_moves = 0, played = 0;
    for (int i = 0; i < deals; ++i) {
        Deal d(cfg, i % kPlayers, shuffled(rng));
        randomBids(d, rng, 0.5);
        if (d.phase() != Phase::kPlay) continue;
        ++played;

        // Belote as Tanguy's engine sees it: one player holding K+Q of trump.
        int belote_team = -1;
        const Mask kq = bit(cardOf(d.trump(), kK)) | bit(cardOf(d.trump(), kQ));
        for (int p = 0; p < kPlayers; ++p)
            if ((d.hand(p) & kq) == kq) belote_team = p % 2;

        std::vector<tanguy::TrickCards> tricks;
        tanguy::TrickCards current;
        while (d.phase() == Phase::kPlay) {
            const int p = d.currentPlayer();
            const std::vector<int> mine = cardsOf(d.legalMoves());
            const std::vector<int> theirs =
                tanguy::LegalCardPlays(cardsOf(d.hand(p)), current, d.trump(), p);
            CHECK(mine == theirs);
            ++compared_moves;
            const int card = mine[std::uniform_int_distribution<int>(0, int(mine.size()) - 1)(rng)];
            d.play(card);
            current.emplace_back(p, card);
            if (current.size() == 4) {
                tricks.push_back(current);
                current.clear();
            }
        }
        const auto [decl, other] = tanguy::FinalScores(tricks, d.trump(), d.takers(), belote_team);
        CHECK_EQ(d.result().credited[d.takers()], decl);
        CHECK_EQ(d.result().credited[1 - d.takers()], other);
    }
    std::printf("tanguy cross-check: %ld deals, %ld legal-move sets compared\n", played, compared_moves);
}

void benchmark(int deals) {
    std::mt19937 rng(1);
    const Config cfg = Config::belgian();
    const auto start = std::chrono::steady_clock::now();
    long cards = 0;
    for (int i = 0; i < deals; ++i) {
        Deal d(cfg, 0, shuffled(rng));
        d.bid(Bid::take());
        while (d.phase() == Phase::kPlay) {
            d.play(randomCard(d.legalMoves(), rng));
            ++cards;
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("benchmark: %d random deals in %.3f s (%.2f us/deal, %.0f ns/card)\n", deals, s,
                1e6 * s / deals, 1e9 * s / cards);
}

}  // namespace

int main() {
    testRandomMatches(2000);
    testTanguyCrossCheck(50000);
    benchmark(200000);
    return checkResult("test_random");
}
