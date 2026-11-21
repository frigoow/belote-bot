#include "belote/match.h"

#include <algorithm>
#include <stdexcept>

namespace belote {

int recordedPoints(int raw, const Config& cfg) {
    if (cfg.score_mode == ScoreMode::kRaw) return raw;
    return (raw + 5) / 10;  // 75 -> 8, 74 -> 7
}

RecordedDeal recordDeal(const DealResult& r, const Config& cfg) {
    RecordedDeal rec;
    if (r.takers < 0 || r.outcome == Outcome::kNone || r.outcome == Outcome::kAllPassed) return rec;
    if (cfg.score_mode != ScoreMode::kComplement) {
        for (int t = 0; t < 2; ++t) rec.scores[t] = recordedPoints(r.credited[t], cfg);
        rec.held = recordedPoints(r.held, cfg);
        return rec;
    }
    // Defenders rounded; takers get the rest. Toon and belot are multiples of
    // ten and the deal is worth 162 (252 on a ronde), so the raw sum ends in 2
    // and its tens are exactly the deal's units.
    const int defenders = 1 - r.takers;
    const int deal_units = (r.credited[0] + r.credited[1] + r.held) / 10;
    rec.scores[defenders] = recordedPoints(r.credited[defenders], cfg);
    const int takers_units = std::max(0, deal_units - rec.scores[defenders]);
    if (r.outcome == Outcome::kHanging) rec.held = takers_units;
    else rec.scores[r.takers] = takers_units;
    return rec;
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

    const RecordedDeal rec = recordDeal(r, cfg_);
    for (int t = 0; t < 2; ++t) scores_[t] += rec.scores[t];

    // Held-over points go to whoever wins the next deal that is decided.
    if (r.outcome == Outcome::kHanging) {
        pot_ += rec.held;
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
