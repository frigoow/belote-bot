#include "bg/rule_policy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bg {
namespace {

// Reuse the sampler's inference exactly. Only plays and the two identifiable
// carré types are inputs to inferFacts. The bounded scratch vectors allocate
// once per thread, never per simulated deal or card.
Facts tableFacts(const Round& r) {
    thread_local PublicState ps = [] {
        PublicState state;
        state.plays.reserve(kCards);
        state.declared.reserve(kPlayers);
        return state;
    }();
    ps.me = r.toMove();
    ps.contract = r.contract();
    ps.plays.clear();
    ps.declared.clear();
    for (const auto& pl : r.plays()) ps.plays.push_back({pl.first, pl.second.first, pl.second.second});
    for (const auto& a : r.announces())
        if (a.type == kFourNines || a.type == kFourJacks) ps.declared.push_back({a.player, a.type, kNoCard});
    return inferFacts(ps);
}

struct Strength {
    Mask above[2][kCards]{};
    Strength() {
        for (int c = 0; c < kCards; ++c) {
            for (int r = 0; r < 8; ++r) {
                const int other = cardOf(suitOf(c), r);
                if (plainOrder(other) > plainOrder(c)) above[0][c] |= bit(other);
                if (trumpOrder(other) > trumpOrder(c)) above[1][c] |= bit(other);
            }
        }
    }
};
const Strength strength;

struct Table {
    const Round& r;
    const Trick& t;
    int me, partner, trump;
    bool all, doubled, sweep;
    Mask hand, unseen;
    Facts facts;
    Hands possible{};
    Mask opponents;
    int led, winner, pot = 0;

    explicit Table(const Round& round)
        : r(round), t(r.trick()), me(r.toMove()), partner(partnerOf(me)), trump(trumpSuit(r.contract())),
          all(r.contract() & kAllTrumps), doubled(r.contract() & (kDouble | kReDouble)),
          sweep(r.trickNumber() >= 3 && (!r.won(0) || !r.won(1))), hand(r.hand(me)),
          unseen(~(hand | r.won(0) | r.won(1))), facts(tableFacts(r)),
          led(t.n ? suitOf(t.cards[0]) : -1), winner(t.n ? trickWinnerIndex(r.contract(), t) : -1) {
        for (int i = 0; i < t.n; ++i) {
            unseen &= ~bit(t.cards[i]);
            pot += points(t.cards[i]);
        }
        const Mask pins = facts.pinned[0] | facts.pinned[1] | facts.pinned[2] | facts.pinned[3];
        for (int p = 0; p < kPlayers; ++p)
            possible[p] = (unseen & ~pins & ~facts.excluded[p]) | facts.pinned[p];
        opponents = possible[nextSeat(me)] | possible[nextSeat(partner)];
    }

    bool trumpRank(int c) const { return all || suitOf(c) == trump; }
    Mask above(int c) const { return strength.above[trumpRank(c)][c]; }
    int points(int c) const {
        return trumpRank(c) ? belote::kTrumpPoints[rankOf(c)] : belote::kPlainPoints[rankOf(c)];
    }

    // Likelihood proxy for a ruff. Distinguish a possible short suit from
    // a proven void, and a pinned follower from either. Overruff threats
    // must be both above the winner and within the seat's trump ceiling.
    double ruffRisk(int p, int suit, Mask trumps) const {
        if (!(possible[p] & trumps) || (facts.pinned[p] & suitMask(suit))) return 0;
        static constexpr double voidRisk[] = {1, .65, .4, .24, .14, .08, .04, .02, .01};
        return voidRisk[popcount(possible[p] & suitMask(suit))];
    }

    double risk(int c, int suit, int p) const {
        const bool ruffed = suitOf(c) != suit;
        const Mask higher = above(c) & possible[p];
        if (ruffed) return ruffRisk(p, suit, higher);
        double lose = (higher & facts.pinned[p]) ? 1.0 : std::min(.85, .36 * popcount(higher));
        if (trump >= 0 && suit != trump)
            lose += (1.0 - lose) * ruffRisk(p, suit, suitMask(trump));
        return lose;
    }

    // Losing the last guard exposes a nine (AT/trump), ten or king (plain).
    double keep(int c) const {
        const Mask suit = suitMask(suitOf(c)), others = hand & suit & ~bit(c);
        const int over = popcount(above(c) & unseen);
        double value = over == 0 ? 12.0 + .45 * points(c) : over == 1 ? 5.0 : 0.0;
        if (suitOf(c) == trump) value += 4.0;
        if (others && !(others & (others - 1))) {
            const int h = lowestCard(others);
            if ((above(c) & others) && popcount(above(h) & unseen) == 1) value += 6.0;
        }
        return value;
    }

    // Claim a legal belot before a forced overtake or an AT discard can
    // break it. An off-suit AT king/queen actually destroys that option.
    double belot(int c) const {
        if (!trumpRank(c) || (rankOf(c) != belote::kQ && rankOf(c) != belote::kK)) return 0;
        if (!(hand & bit(cardOf(suitOf(c), rankOf(c) == belote::kQ ? belote::kK : belote::kQ)))) return 0;
        return beloteAllowed(hand, r.contract(), t, c) ? 8.0 : -16.0;
    }

    double lead(int c) const {
        const int s = suitOf(c);
        const Mask suit = suitMask(s);
        const int length = popcount(hand & suit);
        const bool master = !(above(c) & unseen);
        const bool attacking = teamOf(me) == teamOf(r.contractPlayer());
        const bool draw = trump >= 0 && (opponents & suitMask(trump));
        const double loss = std::max(risk(c, s, nextSeat(me)), risk(c, s, nextSeat(partner)));
        // With no cashing lead, play low from length to establish a suit
        // while keeping outside entries and guards.
        double score = -1.35 * points(c) + .7 * length;
        if (master) score = 14 + .7 * points(c) - 30 * loss;
        else if (points(c) >= 10) score -= 6;
        if (s == trump) {
            if (!draw) score -= (unseen & suit) ? 100 : 21;
            else if (attacking && master) score += 20;
            else if (me == r.contractPlayer() && length >= 3) score += 9;
            else score -= 13;
        } else if (trump >= 0) {
            if (!(possible[partner] & suit) && (possible[partner] & suitMask(trump)))
                score += 13 + .9 * points(c);
            if (length == 1 && (hand & suitMask(trump))) score += 4;
        } else if (all) {
            score += length;
        }
        if (!master) score -= .7 * keep(c);
        return score + belot(c);
    }

    bool publicScore(std::array<int, 2>& bank) const {
        int bestType[2][2]{}, meld[2][2]{};
        for (int tm = 0; tm < 2; ++tm)
            for (Mask m = r.won(tm); m; m &= m - 1) bank[tm] += points(lowestCard(m));
        for (const auto& a : r.announces()) {
            const int team = teamOf(a.player);
            if (a.type == kBelot) bank[team] += 20;
            else {
                const int group = a.type >= kFourOfAKind;
                bestType[group][team] = std::max(bestType[group][team], a.type);
                meld[group][team] += a.value();
            }
        }
        bool knownScore = true;
        for (int group = 0; group < 2; ++group) {
            const int* best = bestType[group];
            if (best[0] == best[1]) knownScore &= best[0] == 0;
            else {
                const int team = best[1] > best[0];
                bank[team] += meld[group][team];
            }
        }
        return knownScore;
    }

    RoundResult lastTwo(int c, std::array<int, 4> end, const std::array<int, 2>& bank) const {
        Trick current = t;
        current.cards[3] = c;
        current.players[3] = me;
        current.n = 4;
        const int leader = current.players[trickWinnerIndex(r.contract(), current)];
        end[me] = lowestCard(hand & ~bit(c));
        Trick final;
        int lastPoints = 10;
        for (int i = 0; i < 4; ++i) {
            const int seat = (leader + i) & 3;
            final.players[i] = seat;
            final.cards[i] = end[seat];
            lastPoints += points(end[seat]);
        }
        final.n = 4;
        const int win7 = teamOf(leader), win8 = teamOf(final.players[trickWinnerIndex(r.contract(), final)]);
        auto raw = bank;
        raw[win7] += pot + points(c);
        raw[win8] += lastPoints;
        if (beloteAllowed(hand, r.contract(), t, c)) raw[teamOf(me)] += 20;
        return finalizeScore(r.contract(), r.contractPlayer(), raw[0], raw[1],
            !r.won(0) && win7 == 1 && win8 == 1, !r.won(1) && win7 == 0 && win8 == 0, r.hangingBefore());
    }

    int finish(int fallback) const {
        const int us = teamOf(me);
        std::array<int, 2> bank{};
        const bool knownScore = publicScore(bank);
        // Fourth hand at trick seven: the other seats have exactly one
        // unplayed card each. All six orders fit on the stack.
        std::array<int, 3> last{};
        Mask m = unseen;
        for (int& card : last) {
            card = lowestCard(m);
            m &= m - 1;
        }
        const int a = lowestCard(hand), b = lowestCard(hand & ~bit(a));
        long long diff = 0;
        do {
            std::array<int, 4> end{};
            bool admitted = true;
            for (int i = 0; i < 3; ++i) {
                const int seat = t.players[i];
                end[seat] = last[i];
                if (!(possible[seat] & bit(last[i]))) admitted = false;
            }
            if (!admitted) continue;
            for (int c : {a, b}) {
                const RoundResult result = lastTwo(c, end, bank);
                // If hidden sequence/carré ranks can change the contract
                // line, use the raw margin including capot instead.
                const int margin = knownScore ? result.sn - result.ew : result.sn_total - result.ew_total;
                diff += (c == a ? 1LL : -1LL) * (us ? -margin : margin);
            }
        } while (std::next_permutation(last.begin(), last.end()));
        return diff > 0 ? a : diff < 0 ? b : fallback;
    }

    // On taking the lead, can we cash every remaining card? Use the
    // public pool including partner, who can be forced to overtake in AT.
    bool canCash(Mask remaining) const {
        int highTrumps = 0;
        for (Mask m = remaining; m; m &= m - 1) {
            int c = lowestCard(m);
            if (above(c) & unseen) return false;
            highTrumps += suitOf(c) == trump;
        }
        return trump < 0 || !(remaining & ~suitMask(trump)) ||
               highTrumps >= popcount(unseen & suitMask(trump));
    }

    double follow(int c) const {
        const int winning = t.cards[winner];
        const int suit = suitOf(c);
        const bool beats = (suit == suitOf(winning) && (above(winning) & bit(c))) ||
                           (suit == trump && suitOf(winning) != trump);
        const bool ours = t.players[winner] == partner;
        const int top = beats ? c : winning;
        // In second and third seat exactly one opponent remains, on our left.
        const double survive = t.n == 3 ? 1 : 1 - risk(top, led, nextSeat(me));
        double win = (beats || ours) ? survive : 0;
        if (t.n == 1) {
            // Fourth-seat partner may cover us. Duck low especially when
            // partner is the only possible holder of a higher card.
            const double cover = risk(top, led, partner);
            win += (1 - win) * .65 * cover;
        }
        double trick = pot + (t.n == 1 ? 8 : t.n == 2 ? 4 : 0) + 4;

        // From trick three, keep the lead for a guaranteed sweep. On a
        // double the same line banks the remaining points for the contract.
        if ((sweep || doubled) && beats && survive == 1.0 &&
            (t.n != 1 || risk(c, led, partner) == 0) && canCash(hand & ~bit(c)))
            trick += 32;
        double score = win * trick + (2 * win - 1) * points(c) - keep(c) + belot(c);
        if (suit != led && suit != trump && !(hand & suitMask(suit) & ~bit(c)) &&
            trump >= 0 && (hand & suitMask(trump))) score += 3;
        return score;
    }
};

}  // namespace

int rulePolicy(const Round& r) {
    if (r.decision() != Round::Decision::kPlay) throw std::logic_error("rulePolicy: not playing");
    Mask legal = r.legalMoves();
    if (!(legal & (legal - 1))) return lowestCard(legal);
    const Table table(r);
    int best = kNoCard;
    double value = -1e9;
    for (; legal; legal &= legal - 1) {
        const int c = lowestCard(legal);
        const double v = table.t.n ? table.follow(c) : table.lead(c);
        if (v > value) {
            best = c;
            value = v;
        }
    }
    return table.t.n == 3 && r.trickNumber() == 7 ? table.finish(best) : best;
}

namespace {

// Scores are in tenths of a bidding unit. A protected nine/ten is worth
// appreciably more than a bare one. Belot and runs matter only with melds.
struct SuitPower {
    int trump, plain, all, side, meld;
};
const auto suitPower = [] {
    std::array<SuitPower, 256> table{};
    for (int mask = 0; mask < 256; ++mask) {
        auto has = [&](int r) { return (mask >> r) & 1; };
        const int n = popcount(mask), j = has(belote::kJ), nine = has(belote::k9);
        const int a = has(belote::kA), ten = has(belote::k10), king = has(belote::kK);
        const int pair = king * has(belote::kQ);
        const int run = mask & (mask >> 1) & (mask >> 2);
        const int meld = (run & (run >> 1)) ? 13 : run ? 6 : 0;
        const int side = 12 * a + ten * (a ? 10 : n >= 2 ? 4 : 1);
        table[mask] = {8 * n + 21 * j + nine * (n >= 2 ? 15 : 6) + 4 * a + 7 * pair,
                       side + 7 * a + ten * (n >= 2 ? 5 : 1) + king * (n >= 2 ? 3 : 1),
                       21 * j + nine * (n >= 2 ? 13 : 6) + a * (n >= 2 ? 6 : 3) + 7 * pair,
                       side, meld};
    }
    return table;
}();

struct BidPower {
    std::array<int, 6> value{};
    BidPower() = default;
    explicit BidPower(Mask h) {
        int side = 0, meld = 0;
        for (int s = 0; s < 4; ++s) {
            const auto& p = suitPower[(h >> (8 * s)) & 255];
            side += p.side;
            meld += p.meld;
            value[s] = p.trump - p.side;
            value[4] += p.plain;
            value[5] += p.all;
        }
        for (int s = 0; s < 4; ++s) value[s] += side + meld;
        value[5] += meld;
        // Open with trump control and length, multiple NT entries, or AT
        // winners across suits. A cheap suit needs less than a high contract.
        for (int s = 0; s < 4; ++s) {
            const int suit = int((h >> (8 * s)) & 255);
            if (popcount(suit) < 2 || !(suit & (bit(belote::kJ) | bit(belote::k9))))
                value[s] -= 18;
        }
    }
};

struct Auction {
    int rank = -1, declarer = -1;
    int multiplier = 1;
    std::array<int, 4> lastSuit{{-1, -1, -1, -1}};

    void record(int p, std::uint8_t b) {
        if (b == kPass) return;
        if (b == kDouble || b == kReDouble) {
            multiplier = b == kDouble ? 2 : 4;
            return;
        }
        rank = lowestCard(b);
        declarer = p;
        multiplier = 1;
        if (rank < 4) lastSuit[p] = rank;
    }
    int raise(int p) const {
        if (rank < 0) return kPass;
        const bool ours = teamOf(p) == teamOf(declarer);
        if (!ours && multiplier == 1) return kDouble;
        if (ours && multiplier == 2) return kReDouble;
        return kPass;
    }

    std::uint8_t choose(int p, const BidPower& hand) const {
        const bool ours = rank >= 0 && teamOf(declarer) == teamOf(p);
        const int raiseBid = raise(p);
        int best = 0;
        std::uint8_t bid = kPass;
        if (raiseBid) {
            // Partner's double suggests help. Redoubling one's own call
            // still requires extra strength after the opponents' warning.
            const int threshold = raiseBid == kDouble ? 57 : declarer == p ? 66 : 44;
            best = hand.value[rank] - threshold;
            if (best > 0) bid = std::uint8_t(raiseBid);
            else best = 0;
        }
        static constexpr int opening[] = {44, 44, 44, 44, 41, 43};
        for (int s = rank + 1; s < 6; ++s) {
            int v = hand.value[s] - opening[s];
            if (ours) v -= 17;  // respect a playable contract from partner
            if (s == 5 && lastSuit[partnerOf(p)] >= 0) v += 8;
            if (s == 4 && lastSuit[partnerOf(p)] >= 0) v += 3;
            if (v > best) {
                bid = std::uint8_t(1u << s);
                best = v;
            }
        }
        return bid;
    }
};

struct FiveCardHands {
    std::array<double, 56> probability{};
    // Compute the powers once per subset, even if this seat called repeatedly.
    std::array<BidPower, 56> powers;

    void prepare(Mask eight) {
        if (popcount(eight) != 8) throw std::invalid_argument("auctionLogLik: expected eight original cards");
        Mask cards[8];
        for (int i = 0; i < 8; ++i, eight &= eight - 1) cards[i] = bit(lowestCard(eight));
        int i = 0;
        for (unsigned subset = 31; subset < 256;) {
            Mask h = 0;
            for (unsigned m = subset; m; m &= m - 1) h |= cards[lowestCard(m)];
            powers[i++] = BidPower(h);
            // Gosper's next five set bits.
            const unsigned low = subset & -subset, next = subset + low;
            subset = next | (((next ^ subset) >> 2) / low);
        }
        probability.fill(1.0);
    }
};

}  // namespace

std::uint8_t naturalBid(const Round& r) {
    if (r.decision() != Round::Decision::kBid) throw std::logic_error("naturalBid: not bidding");
    Auction auction;
    for (auto [seat, bid] : r.bids()) auction.record(seat, bid);
    return auction.choose(r.toMove(), BidPower(r.hand(r.toMove())));
}

double auctionLogLik(const PublicState& ps, const Hands& hands) {
    Auction auction;
    std::array<FiveCardHands, 4> seats;
    std::array<bool, 4> used{};
    for (auto [seat, bid] : ps.bids) {
        const int actions = 6 - auction.rank + (auction.raise(seat) != kPass);
        if (seat != ps.me && actions > 1) {
            auto& h = seats[seat];
            if (!used[seat]) {
                h.prepare(hands[seat] | ps.playedBy(seat));
                used[seat] = true;
            }
            // A lapse uniformly among legal actions keeps unfamiliar bidding
            // styles possible. Forced passes have unit probability.
            const double noise = .25 / actions;
            for (unsigned i = 0; i < h.probability.size(); ++i)
                h.probability[i] *= noise + (auction.choose(seat, h.powers[i]) == bid ? .75 : 0.);
        }
        auction.record(seat, bid);
    }
    double logLik = 0;
    for (int p = 0; p < 4; ++p) {
        if (!used[p]) continue;
        double total = 0;
        for (double v : seats[p].probability) total += v;
        logLik += std::log(total / 56);
    }
    return logLik;
}

}  // namespace bg
