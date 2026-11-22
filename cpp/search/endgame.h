#pragma once

// Contract-aware exact double-dummy alpha-beta endgame solver (docs/PLAN_BELGIAN.md §3.4).
//
// Solves a perfect-information Belote deal from any point in phase kPlay to
// the exact terminal deal score (card points + last trick + ronde + toon +
// belot + contract cliff + tie_rule + score_mode). Designed for the last 4-5
// tricks (<= 16-20 cards left), using bitset moves, 0-point equivalent-card
// pruning, and a trick-boundary Zobrist transposition table.

#include "belote/deal.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace belote {

class BidModel;

enum class EndgameObjective {
    kUnits,        // recorded unit margin from recordDeal() (held-over excluded)
    kRaw,          // raw credited point margin
    kMatchEquity,  // exact scaled match equity: round(10000 * (2*V_team0 - 1))
    kMatchFinish,  // bounded scoreboard heuristic for agents without an equity model
};

// Team 0 perspective. Terminal wins/losses score +/-1. Ongoing positions
// use the score lead discounted by one deal and the outstanding pot.
double matchFinishValue(const DealResult& result, const Config& cfg, std::array<int, 2> scores, int pot);

struct EndgameOptions {
    EndgameObjective objective = EndgameObjective::kUnits;
    int tt_log2 = 18;  // 2^18 entries (~4 MB)
    const BidModel* bid_model = nullptr;
    std::array<int, 2> match_scores = {0, 0};
    int match_pot = 0;
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

// Fast single-value exact minimax solve (returns value from Team 0's perspective).
// Faster than solveEndgame() when per-move root scores are not needed (e.g. rollout finish).
int solveEndgameValue(const Deal& deal, const EndgameOptions& options = {});

// Plays an explicit sequence of legal cards to completion through the endgame
// solver's internal state and leaf evaluator (returns Team 0 minus Team 1
// score). Used to verify early-deal root setup (pending announcements, shows,
// belot forfeit) directly against Deal::play().
int evaluateEndgameLine(const Deal& deal, const std::vector<int>& cards,
                        EndgameObjective objective = EndgameObjective::kUnits);

}  // namespace belote
