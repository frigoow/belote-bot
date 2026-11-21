#include "bg/rule_policy.h"
#include "check.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

using namespace bg;

namespace {

using Deck = std::array<int, kCards>;

// Deal exactly these five cards to one seat. Fill the rest in deck order.
Deck withFive(int seat, Mask five, bool reverse = false) {
    Deck deck;
    deck.fill(kNoCard);
    const Mask chosen = five;
    int next = reverse ? 31 : 0;
    const int step = reverse ? -1 : 1;
    for (int i = 0; five; five &= five - 1, ++i) deck[4 * i + seat] = lowestCard(five);
    for (int& c : deck) {
        if (c != kNoCard) continue;
        while (chosen & bit(next)) next += step;
        c = next;
        next += step;
    }
    return deck;
}

void checkCall(const char* cards, std::uint8_t expected, std::initializer_list<std::uint8_t> history = {}) {
    const int p = int(history.size()) % 4;
    for (bool reverse : {false, true}) {
        Round r(withFive(p, belote::parseCards(cards), reverse), 0);
        for (auto b : history) r.bid(b);
        CHECK_EQ(r.toMove(), p);
        CHECK_EQ(naturalBid(r), expected);
    }
}

// Independent reference: enumerate all submasks (including sizes other
// than five), and replay the actual auction through Round's legal-bid API.
double reference(const PublicState& ps, const Hands& hands) {
    double ll = 0;
    for (int p = 0; p < kPlayers; ++p) {
        if (p == ps.me) continue;
        const Mask eight = hands[p] | ps.playedBy(p);
        double total = 0;
        int subsets = 0;
        for (Mask five = eight; five; five = (five - 1) & eight) {
            if (popcount(five) != 5) continue;
            Round r(withFive(p, five), ps.first);
            double pr = 1;
            while (r.bids().size() < ps.bids.size()) {
                const auto [seat, bid] = ps.bids[r.bids().size()];
                CHECK_EQ(seat, r.toMove());
                if (p == seat) {
                    const int legal = 1 + popcount(r.availableBids());
                    pr *= .25 / legal + (naturalBid(r) == bid ? .75 : 0);
                }
                r.bid(bid); // includes automatic passes
            }
            total += pr;
            ++subsets;
        }
        CHECK_EQ(subsets, 56);
        ll += std::log(total / subsets);
    }
    return ll;
}

void testAuction() {
    checkCall("7C 8D 7H 8S QS", kPass);
    checkCall("JH 9H 7H 8D 7S", kHearts);
    checkCall("AC 10C AD KS 7H", kNoTrumps);
    checkCall("JC JD JH 7S 8S", kAllTrumps);
    checkCall("JH 9H AH 7C 8D", kDouble, {kHearts});
    checkCall("JH 9H AH 7C 8D", kReDouble, {kHearts, kDouble});

    PublicState ps;
    ps.bids = {{0,kClubs},{1,kNoTrumps},{2,kDouble},{3,kAllTrumps},
               {0,kDouble},{1,kReDouble},{2,kPass},{3,kPass},{0,kPass}};
    Hands h{};
    // Deal eight disjoint cards to each seat and move one from each into
    // public history so treating the current hand as the original fails.
    for (int p = 0; p < 4; ++p) {
        h[p] = Mask(0xff) << (8 * p);
        const int played = lowestCard(h[p]);
        h[p] &= ~bit(played);
        ps.plays.push_back({p, played, false});
    }
    for (ps.me = 0; ps.me < 4; ++ps.me) {
        ps.hand = h[ps.me];
        CHECK(std::abs(auctionLogLik(ps, h) - reference(ps, h)) < 1e-10);
        Hands masked = h;
        masked[ps.me] = 0;
        CHECK_EQ(auctionLogLik(ps, h), auctionLogLik(ps, masked));
    }
    ps.me = 1;
    ps.plays.clear();
    ps.bids = {{0,kHearts}};
    for (int p = 0; p < 4; ++p) h[p] |= bit(8 * p);
    const double weak = auctionLogLik(ps, h);
    std::swap(h[0], h[2]);
    CHECK(auctionLogLik(ps, h) > weak);
    ps.bids.clear();
    CHECK_EQ(auctionLogLik(ps, h), 0);
}

// A blind pair of worlds with equal prior probability. Generate auctions
// with the bidder and its stated lapse rate, then score the posterior on
// the true world against the one with two unseen seats exchanged.
void testCalibration() {
    std::mt19937_64 rng(121);
    Deck deck;
    std::iota(deck.begin(), deck.end(), 0);
    int measured = 0;
    double brier = 0, confidence = 0, correct = 0;
    for (int i = 0; i < 1600; ++i) {
        std::shuffle(deck.begin(), deck.end(), rng);
        Round r(deck, i % 4);
        while (r.decision() == Round::Decision::kBid) {
            std::uint8_t bid = naturalBid(r);
            if (rng() % 4 == 0) {
                Mask legal = r.availableBids();
                int k = int(rng() % (1 + popcount(legal)));
                bid = kPass;
                if (k > 0) {
                    while (--k) legal &= legal - 1;
                    bid = std::uint8_t(bit(lowestCard(legal)));
                }
            }
            r.bid(bid);
        }
        if (r.finished()) continue;
        const int me = (i / 4) % 4;
        PublicState ps = publicState(r, me);
        Hands h{};
        for (int p = 0; p < 4; ++p) h[p] = r.hand(p);
        const double truth = auctionLogLik(ps, h);
        std::swap(h[nextSeat(me)], h[partnerOf(me)]);
        const double prob = 1 / (1 + std::exp(auctionLogLik(ps, h) - truth));
        brier += (1 - prob) * (1 - prob);
        confidence += std::max(prob, 1 - prob);
        correct += prob == .5 ? .5 : prob > .5;
        ++measured;
    }
    CHECK(measured > 1200);
    CHECK(brier / measured < .23); // uniform predicts .5, Brier score .25
    CHECK(std::abs(confidence - correct) / measured < .035);
    std::printf("auction calibration: Brier %.3f, confidence %.3f, accuracy %.3f (%d pairs)\n",
                brier / measured, confidence / measured, correct / measured, measured);
}

void testLastTwo() {
    std::mt19937_64 rng(89);
    Deck deck;
    std::iota(deck.begin(), deck.end(), 0);
    int checked = 0;
    for (int d = 0; checked < 200 && d < 4000; ++d) {
        std::shuffle(deck.begin(), deck.end(), rng);
        Round r(deck, d % 4, d % 5);
        r.bid(std::uint8_t(1u << (d % 6)));
        if (d % 3) r.bid(kDouble);
        if (d % 3 == 2) r.bid(kReDouble);
        while (r.decision() == Round::Decision::kBid) r.bid(kPass);
        while (!r.finished()) {
            if (r.decision() == Round::Decision::kAnnounce) {
                r.declare({});
                continue;
            }
            if (r.trickNumber() == 7 && r.trick().n == 3) break;
            Mask m = r.legalMoves();
            int k = int(rng() % popcount(m));
            while (k--) m &= m - 1;
            r.play(lowestCard(m), true);
        }
        if (r.finished()) continue;
        const int me = r.toMove(), move = rulePolicy(r);
        const int other = lowestCard(r.legalMoves() & ~bit(move));
        PublicState ps = publicState(r, me);
        const DealSampler sampler(ps);
        std::array<int, 3> cards{};
        Mask pool = ~(ps.hand | ps.played());
        for (int& card : cards) {
            card = lowestCard(pool);
            pool &= pool - 1;
        }
        int gain = 0, layouts = 0;
        do {
            Hands h{};
            h[me] = ps.hand;
            for (int i = 0; i < 3; ++i) h[(me + 1 + i) % 4] = bit(cards[i]);
            auto replicate = sampler.replay(h);
            if (!replicate) continue;
            for (int c : {move, other}) {
                Round next = *replicate;
                next.play(c, true); // Round auto-plays the whole last trick
                CHECK(next.finished());
                const int margin = next.result().sn - next.result().ew;
                gain += (c == move ? 1 : -1) * (teamOf(me) == 0 ? margin : -margin);
            }
            ++layouts;
        } while (std::next_permutation(cards.begin(), cards.end()));
        CHECK(layouts > 0);
        CHECK(gain >= 0);
        ++checked;
    }
    CHECK_EQ(checked, 200);
}

void testPolicy() {
    std::mt19937_64 rng(207);
    Deck deck;
    std::iota(deck.begin(), deck.end(), 0);
    int alternatives = 0;
    for (int d = 0; d < 400; ++d) {
        std::shuffle(deck.begin(), deck.end(), rng);
        Round r(deck, d % 4);
        while (r.decision() == Round::Decision::kBid) {
            const auto action = naturalBid(r);
            CHECK(action == kPass || (r.availableBids() & action));
            r.bid(action);
        }
        while (!r.finished()) {
            if (r.decision() == Round::Decision::kAnnounce) {
                r.declare(r.offeredAnnounces());
                continue;
            }
            const int c = rulePolicy(r);
            CHECK(r.legalMoves() & bit(c));
            if (rng() % 9 == 0) {
                const auto ps = publicState(r, r.toMove());
                DealSampler sampler(ps);
                auto h = sampler.sample(rng);
                if (h) {
                    auto other = sampler.replay(*h);
                    CHECK(other.has_value());
                    if (other && h->at(nextSeat(ps.me)) != r.hand(nextSeat(ps.me))) {
                        CHECK_EQ(rulePolicy(*other), c);
                        ++alternatives;
                    }
                }
            }
            r.play(c, true);
        }
    }
    CHECK(alternatives > 200);
}

}  // namespace

int main() {
    testAuction();
    testCalibration();
    testLastTwo();
    testPolicy();
    return checkResult("test_bg_rule_policy");
}
