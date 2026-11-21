#pragma once

// Contract-aware exact double-dummy alpha-beta endgame solver (PLAN.md §3.4).
//
// Solves a perfect-information Belote deal from any point in phase kPlay to
// the exact terminal deal score (card points + last trick + ronde + toon +
// belot + contract cliff + tie_rule + score_mode). Designed for the last 4-5
// tricks (<= 16-20 cards left), using bitset moves, 0-point equivalent-card
// pruning, and a trick-boundary Zobrist transposition table.

#include "belote/deal.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace belote {

enum class EndgameObjective {
    kUnits,  // recorded unit margin from recordDeal() (held-over excluded)
    kRaw,    // raw credited point margin
};

struct EndgameOptions {
    EndgameObjective objective = EndgameObjective::kUnits;
    int tt_log2 = 18;  // 2^18 entries (~4 MB)
};

struct EndgameResult {
    int best_move = kNoCard;
    // Exact minimax value from the root player's team's perspective.
    int value = 0;
    // Exact minimax value for every legal root move (card, value), from the
    // root player's team's perspective.
    std::vector<std::pair<int, int>> move_values;
    std::uint64_t nodes = 0;
};

// Solves the given deal exactly under perfect information. Must be in kPlay.
EndgameResult solveEndgame(const Deal& deal, const EndgameOptions& options = {});

// Plays an explicit sequence of legal cards to completion through the endgame
// solver's internal state and leaf evaluator (returns Team 0 minus Team 1
// score). Used to verify early-deal root setup (pending announcements, shows,
// belot forfeit) directly against Deal::play().
int evaluateEndgameLine(const Deal& deal, const std::vector<int>& cards,
                        EndgameObjective objective = EndgameObjective::kUnits);

}  // namespace belote
