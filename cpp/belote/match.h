#pragma once

// The match (docs/RULES_BELGIAN.md §6–§7): scores recorded per deal, points held over by
// a hanging deal, dealer rotation and redeals, and the race to 101.

#include "belote/deal.h"
#include "belote/rules.h"

#include <array>

namespace belote {

// Raw points as written on the score sheet: units (nearest ten, 5 up) or raw.
// In kComplement mode this is the defenders' rounding.
int recordedPoints(int raw, const Config& cfg);

// What a decided deal writes on the score sheet: each team's recorded score
// and, for a hanging deal, the takers' recorded points that are held over.
struct RecordedDeal {
    std::array<int, 2> scores{0, 0};
    int held = 0;
};
RecordedDeal recordDeal(const DealResult& result, const Config& cfg);

// Project one deal onto the scoreboard, including a previous hanging pot.
// Returns the match winner, or -1 if the match continues.
int applyDealScore(const DealResult& result, const Config& cfg, std::array<int, 2>& scores, int& pot);

class Match {
public:
    explicit Match(const Config& cfg, int first_dealer = 0);

    const Config& config() const { return cfg_; }
    int dealer() const { return dealer_; }  // dealer of the next deal
    std::array<int, 2> scores() const { return scores_; }
    int pot() const { return pot_; }        // held-over points, already recorded
    bool finished() const { return winner_ >= 0; }
    int winner() const { return winner_; }
    int dealsRecorded() const { return deals_; }

    // Record a finished deal (including kAllPassed) and move the deal on.
    void record(const DealResult& result);

private:
    Config cfg_;
    int dealer_;
    std::array<int, 2> scores_{0, 0};
    int pot_ = 0;
    int consecutive_passes_ = 0;
    int deals_ = 0;
    int winner_ = -1;
};

}  // namespace belote
