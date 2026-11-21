#pragma once

// The match (RULES.md §6–§7): scores recorded per deal, points held over by
// a hanging deal, dealer rotation and redeals, and the race to 101.

#include "belote/deal.h"
#include "belote/rules.h"

#include <array>

namespace belote {

// Raw points as written on the score sheet: units (nearest ten, 5 up) or raw.
int recordedPoints(int raw, const Config& cfg);

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
