#include "belote/match.h"

#include <algorithm>
#include <stdexcept>

namespace belote {

int recordedPoints(int raw, const Config& cfg) {
    if (cfg.score_mode == ScoreMode::kRaw) return raw;
    return (raw + 5) / 10;  // 75 -> 8, 74 -> 7
}

Match::Match(const Config& cfg, int first_dealer) : cfg_(cfg), dealer_(first_dealer) {}

void Match::record(const DealResult& r) {
    if (finished()) throw std::logic_error("match is over");
    ++deals_;

    if (r.outcome == Outcome::kAllPassed) {
        ++consecutive_passes_;
        if (cfg_.redeal_next_dealer || consecutive_passes_ >= 2) {
            dealer_ = nextPlayer(dealer_);
            consecutive_passes_ = 0;
        }
        return;
    }
    if (r.outcome == Outcome::kNone) throw std::logic_error("deal not finished");
    consecutive_passes_ = 0;

    for (int t = 0; t < 2; ++t) scores_[t] += recordedPoints(r.credited[t], cfg_);

    // Held-over points go to whoever wins the next deal that is decided.
    if (r.outcome == Outcome::kHanging) {
        pot_ += recordedPoints(r.held, cfg_);
    } else if (pot_ > 0) {
        const int winner = r.outcome == Outcome::kMade ? r.takers : 1 - r.takers;
        scores_[winner] += pot_;
        pot_ = 0;
    }

    dealer_ = nextPlayer(dealer_);

    const int lead = scores_[0] >= scores_[1] ? 0 : 1;
    const int margin = scores_[lead] - scores_[1 - lead];
    if (scores_[lead] >= cfg_.match_target && margin >= std::max(1, cfg_.match_win_margin))
        winner_ = lead;
}

}  // namespace belote
