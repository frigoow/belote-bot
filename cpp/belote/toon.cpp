#include "belote/toon.h"

namespace belote {

ToonKind kindOf(const Toon& t) {
    if (t.carre) return ToonKind::kCarre;
    if (t.length == 3) return ToonKind::kDerde;
    if (t.length == 4) return ToonKind::kVierde;
    return ToonKind::kVijfde;
}

Announcement announcementOf(const std::vector<Toon>& toons, int trump) {
    const Toon* best = &toons.front();
    for (const Toon& t : toons)
        if (toonKey(t, trump) > toonKey(*best, trump)) best = &t;
    Announcement a;
    a.kind = kindOf(*best);
    if (best->carre) a.carre_rank = best->rank;
    return a;
}

int sequencePoints(int length) {
    if (length >= 5) return 100;
    if (length == 4) return 50;
    if (length == 3) return 20;
    return 0;
}

int carrePoints(int rank, const Config& cfg) {
    switch (rank) {
        case kJ: return 200;
        case k9: return 150;
        case kA: case k10: case kK: case kQ: return 100;
        default: return cfg.carre_low_points;  // 7s and 8s
    }
}

namespace {

void appendSequences(Mask hand, std::vector<Toon>& out) {
    for (int suit = 0; suit < kSuits; ++suit) {
        const unsigned byte = (hand >> (suit * 8)) & 0xFF;
        int rank = 0;
        while (rank < kRanks) {
            if (!(byte & (1u << rank))) { ++rank; continue; }
            int end = rank;
            while (end + 1 < kRanks && (byte & (1u << (end + 1)))) ++end;
            const int length = end - rank + 1;
            if (length >= 3) {
                Toon t;
                t.suit = suit;
                t.rank = end;
                t.length = length;
                t.cards = Mask((((1u << length) - 1) << rank) << (suit * 8));
                t.points = sequencePoints(length);
                out.push_back(t);
            }
            rank = end + 1;
        }
    }
}

}  // namespace

std::vector<Toon> findToons(Mask hand, const Config& cfg) {
    std::vector<Toon> carres;
    for (int rank = 0; rank < kRanks; ++rank) {
        Mask four = 0;
        for (int suit = 0; suit < kSuits; ++suit) four |= bit(cardOf(suit, rank));
        if ((hand & four) == four && carrePoints(rank, cfg) > 0) {
            Toon t;
            t.carre = true;
            t.rank = rank;
            t.length = 4;
            t.cards = four;
            t.points = carrePoints(rank, cfg);
            carres.push_back(t);
        }
    }

    // At most two carrés fit in 8 cards: try every subset of them.
    std::vector<Toon> best;
    int best_points = -1;
    const unsigned subsets = 1u << carres.size();
    for (unsigned s = 0; s < subsets; ++s) {
        std::vector<Toon> pick;
        Mask rest = hand;
        for (size_t i = 0; i < carres.size(); ++i) {
            if (s & (1u << i)) {
                pick.push_back(carres[i]);
                rest &= ~carres[i].cards;
            }
        }
        appendSequences(rest, pick);
        const int points = totalPoints(pick);
        // Break equal-point ties in favor of subsets with a carré (higher s),
        // because any carré beats any sequence in toonKey().
        if (points >= best_points) {
            best_points = points;
            best = std::move(pick);
        }
    }
    return best;
}

long toonKey(const Toon& t, int trump) {
    // carré > sequence; carrés by J 9 A 10 K Q 8 7 (the trump order);
    // sequences by length, then top card, then trump suit.
    if (t.carre) return (1L << 20) | (long(kTrumpStrength[t.rank]) << 8);
    return (long(t.length) << 8) | (long(t.rank) << 4) | (t.suit == trump ? 1 : 0);
}

int toonWinningTeam(const std::vector<Toon> (&declared)[kPlayers], int trump, int dealer) {
    long best_key = -1;
    int best_order = kPlayers;
    int best_player = -1;
    for (int p = 0; p < kPlayers; ++p) {
        const int order = (p - dealer - 1 + 2 * kPlayers) % kPlayers;
        for (const Toon& t : declared[p]) {
            const long key = toonKey(t, trump);
            if (key > best_key || (key == best_key && order < best_order)) {
                best_key = key;
                best_order = order;
                best_player = p;
            }
        }
    }
    return best_player < 0 ? -1 : teamOf(best_player);
}

int totalPoints(const std::vector<Toon>& toons) {
    int sum = 0;
    for (const Toon& t : toons) sum += t.points;
    return sum;
}

}  // namespace belote
