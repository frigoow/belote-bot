#pragma once

// Card-play observation, chronological ply history, and candidate-action
// encoder for Bulgarian Bridge-Belot (Manille v1.2.3 neural stack).
//
// Strictly reads only bg::PublicState (zero hidden-card access).
// For Suit contracts, suits are canonicalized with trump -> suit 0 via
// (suit - trump + 4) & 3; for NoTrumps and AllTrumps, canon_base = 0.
// All seats are relative to ps.me (0=self, 1=left, 2=partner, 3=right).

#include "bg/bg.h"
#include "bg/view.h"

#include <array>
#include <cstdint>

namespace bg {

inline constexpr int kBgEncoderVersion = 1;
inline constexpr int kBgStateDim = 611;
inline constexpr int kBgPlyDim = 56;
inline constexpr int kBgMaxPlies = kCards;
inline constexpr int kBgActionDim = 44;
inline constexpr int kBgMaxLegal = 8;

inline constexpr int canonBase(std::uint8_t contract) {
    if (contract & kClubs) return 0;
    if (contract & kDiamonds) return 1;
    if (contract & kHearts) return 2;
    if (contract & kSpades) return 3;
    return 0;
}

inline constexpr int canonSuit(int suit, std::uint8_t contract) {
    return (suit - canonBase(contract) + 4) & 3;
}

inline constexpr int rawSuit(int csuit, std::uint8_t contract) {
    return (csuit + canonBase(contract)) & 3;
}

inline constexpr int canonCard(int card, std::uint8_t contract) {
    return canonSuit(suitOf(card), contract) * 8 + rankOf(card);
}

inline constexpr int rawCard(int ccard, std::uint8_t contract) {
    return rawSuit(suitOf(ccard), contract) * 8 + rankOf(ccard);
}

inline constexpr int relSeat(int seat, int observer) { return (seat - observer + kPlayers) & 3; }

// Encodes the current static observation into out[kBgStateDim] (zero-initialized).
// If `facts` is non-null, reuses precomputed inferFacts(ps).
void encodeState(const PublicState& ps, float* out, const Facts* facts = nullptr);

// Encodes the chronological play history into out[kBgMaxPlies * kBgPlyDim]
// (row-major, zero-padded after n_plies). Returns n_plies = int(ps.plays.size()).
int encodeHistory(const PublicState& ps, float* out);

// Encodes a single candidate card into out[kBgActionDim] (zero-initialized).
void encodeAction(const PublicState& ps, int card, float* out);

// Convenience helper: encodes all legal moves of ps into actions[kBgMaxLegal * kBgActionDim]
// and stores raw card ids in legal_cards[kBgMaxLegal] (-1 padded). Returns n_legal.
int encodeLegalActions(const PublicState& ps, float* actions_out, std::int32_t* legal_cards_out);

}  // namespace bg
