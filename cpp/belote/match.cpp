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

int applyDealScore(const DealResult& r, const Config& cfg, std::array<int, 2>& scores, int& pot) {
    if (r.outcome == Outcome::kNone) throw std::logic_error("deal not finished");
    if (r.outcome != Outcome::kAllPassed) {
        const RecordedDeal rec = recordDeal(r, cfg);
        for (int t = 0; t < 2; ++t) scores[t] += rec.scores[t];
        if (r.outcome == Outcome::kHanging) {
            pot += rec.held;
        } else {
            scores[r.outcome == Outcome::kMade ? r.takers : 1 - r.takers] += pot;
            pot = 0;
        }
    }
    const int lead = scores[0] >= scores[1] ? 0 : 1;
    return scores[lead] >= cfg.match_target && scores[lead] - scores[1 - lead] >= std::max(1, cfg.match_win_margin)
        ? lead : -1;
}

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

    winner_ = applyDealScore(r, cfg_, scores_, pot_);
    dealer_ = nextPlayer(dealer_);
}

}  // namespace belote
