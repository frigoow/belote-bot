#pragma once

// Cards, masks and the point/strength tables of Belgian Belote (RULES.md §3).
//
// A card is an index 0..31: suit * 8 + rank. Ranks are in the natural
// sequence order 7 8 9 10 J Q K A (the "wies-volgorde" of RULES.md §5.1), so
// a sequence is a run of adjacent bits within one suit's byte. Trick strength
// is a separate table because it differs for trump and plain suits.
//
// Seats 0..3, clockwise; partners are p and p+2, so team = p % 2.

#include <cstdint>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace belote {

using Mask = std::uint32_t;

inline constexpr int kSuits = 4;
inline constexpr int kRanks = 8;
inline constexpr int kCards = 32;
inline constexpr int kPlayers = 4;
inline constexpr int kTricks = 8;
inline constexpr int kNoCard = -1;
inline constexpr int kNoSuit = -1;

enum Suit : int { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };
enum Rank : int { k7 = 0, k8 = 1, k9 = 2, k10 = 3, kJ = 4, kQ = 5, kK = 6, kA = 7 };

inline constexpr int suitOf(int card) { return card >> 3; }
inline constexpr int rankOf(int card) { return card & 7; }
inline constexpr int cardOf(int suit, int rank) { return suit * 8 + rank; }
inline constexpr Mask bit(int card) { return Mask(1) << card; }
inline constexpr Mask suitMask(int suit) { return Mask(0xFF) << (suit * 8); }

inline constexpr int teamOf(int player) { return player & 1; }
inline constexpr int partnerOf(int player) { return (player + 2) & 3; }
inline constexpr int nextPlayer(int player) { return (player + 1) & 3; }

// Strength within a trick, low to high, indexed by rank (7 8 9 10 J Q K A).
// Trump: J 9 A 10 K Q 8 7. Plain: A 10 K Q J 9 8 7.
inline constexpr int kTrumpStrength[kRanks] = {0, 1, 6, 4, 7, 2, 3, 5};
inline constexpr int kPlainStrength[kRanks] = {0, 1, 2, 6, 3, 4, 5, 7};

// Card points, indexed by rank. Trump suit 62, each plain suit 30.
inline constexpr int kTrumpPoints[kRanks] = {0, 0, 14, 10, 20, 3, 4, 11};
inline constexpr int kPlainPoints[kRanks] = {0, 0, 0, 10, 2, 3, 4, 11};

inline constexpr int kLastTrickBonus = 10;
inline constexpr int kDealPoints = 162;  // 152 card points + last trick
inline constexpr int kBelotPoints = 20;

inline constexpr int cardPoints(int card, int trump) {
    return suitOf(card) == trump ? kTrumpPoints[rankOf(card)] : kPlainPoints[rankOf(card)];
}

inline constexpr int cardStrength(int card, int trump) {
    return suitOf(card) == trump ? kTrumpStrength[rankOf(card)] : kPlainStrength[rankOf(card)];
}

inline int popcount(Mask m) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(m));
#else
    return __builtin_popcount(m);
#endif
}

// The mask must be non-zero.
inline int lowestCard(Mask m) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, m);
    return static_cast<int>(index);
#else
    return __builtin_ctz(m);
#endif
}

// "7C", "10H", "JS", "AD" -- suits C D H S (klaveren, ruiten, harten, schoppen).
std::string cardName(int card);
// Inverse of cardName; returns kNoCard on bad input.
int parseCard(const std::string& name);
// Space-separated card names, e.g. "JH 9H AH".
Mask parseCards(const std::string& names);
std::string maskString(Mask m);

}  // namespace belote
