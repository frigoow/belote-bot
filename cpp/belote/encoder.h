#pragma once

// Card-play observation & action encoder (docs/PLAN_BELGIAN.md §3.1, M2).
//
// Strictly reads only belote::Observation (zero hidden-card access). All cards
// and suits are trump-canonicalized (trump -> suit 0 via (suit - trump) mod 4)
// and all seats are relative to obs.observer (0=self, 1=left, 2=partner,
// 3=right). Golden-tested bit-for-bit against python/belote/encoder.py.

#include "belote/card.h"
#include "belote/observation.h"

#include <array>
#include <vector>

namespace belote {

inline constexpr int kEncoderVersion = 1;
inline constexpr int kStateDim = 697;
inline constexpr int kPlyDim = 54;
inline constexpr int kMaxPlies = kCards;
inline constexpr int kActionDim = 41;

inline constexpr int canonSuit(int suit, int trump) { return (suit - trump + kSuits) & 3; }
inline constexpr int rawSuit(int csuit, int trump) { return (csuit + trump) & 3; }
inline constexpr int canonCard(int card, int trump) { return canonSuit(suitOf(card), trump) * kRanks + rankOf(card); }
inline constexpr int rawCard(int ccard, int trump) { return rawSuit(suitOf(ccard), trump) * kRanks + rankOf(ccard); }
inline constexpr int relSeat(int player, int observer) { return (player - observer + kPlayers) & 3; }

// Encodes the current static observation into out[kStateDim] (zero-initialized).
void encodeState(const Observation& obs, float* out);

// Encodes the chronological play history into out[kMaxPlies * kPlyDim]
// (row-major, zero-padded after n_plies). Returns n_plies = obs.plays.size().
int encodeHistory(const Observation& obs, float* out);

// Encodes a single candidate card into out[kActionDim] (zero-initialized).
void encodeAction(const Observation& obs, int card, float* out);

// Convenience return structs for C++ / Python bindings.
struct EncodedObs {
    std::array<float, kStateDim> state{};
    std::array<float, kMaxPlies * kPlyDim> history{};
    int n_plies = 0;
};

EncodedObs encodeObservation(const Observation& obs);

}  // namespace belote
