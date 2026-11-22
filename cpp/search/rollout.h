#pragma once

// Pemeja-style shared-world rollout card player (docs/PLAN_BELGIAN.md §5 M1).
//
// Samples W consistent worlds (or enumerates all worlds when count <= W) using
// WorldSampler, and evaluates every legal move on the exact same set of worlds
// (Common Random Numbers). Each world is played to completion with the fast
// heuristic play-out (ruleChooseCard) -- or solved exactly with solveEndgame()
// once tricksPlayed() >= ab_handoff_trick.

#include "belote/deal.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace belote {

struct RolloutOptions {
    int worlds = 32;
    int ab_handoff_trick = -1;  // >= 0: exact AB per world once tricksPlayed() >= this
    std::uint64_t seed = 0;     // 0: use thread_local random_device-seeded RNG
};

struct RolloutResult {
    int best_move = kNoCard;
    double value = 0;
    int worlds_used = 0;
    bool exact_enumeration = false;
    std::vector<std::pair<int, double>> move_values;
};

// Evaluates the current player's legal moves via shared-world rollouts.
RolloutResult rolloutSearch(const Deal& deal, const RolloutOptions& options = {});

}  // namespace belote
