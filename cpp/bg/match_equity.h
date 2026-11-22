#pragma once

// Exact 1-pass Match-Equity DP for Bulgarian Bridge-Belot (151-point target,
// hanging points, capot-cannot-end-game rule, must-have-scored rule — matching
// bg::GameScore::record in cpp/bg/bg.cpp).

#include "bg/bg.h"

#include <array>
#include <vector>

namespace bg {

class MatchEquityTable {
public:
    static constexpr int kMaxScore = 200;  // Grid covers 0..200; higher scores shift preserving diff.
    static constexpr int kHangBuckets = 3; // 0: 0, 1: 1..18 (~12), 2: 19+ (~26)

    // Singleton table solved once from empirical Bulgarian deal distributions (~3 ms).
    static const MatchEquityTable& instance();

    // Win probability in [0, 1] for `our` team BEFORE a deal starts at
    // (our_score, opp_score, hanging, we_lead).
    double preDealEquity(int our_score, int opp_score, int hanging, bool we_lead) const;

    // Exact post-deal win probability in [0, 1] for `team` given the pre-deal
    // game score (sn_before, ew_before, first_before) and the finished RoundResult.
    double postDealEquity(int sn_before, int ew_before, int first_before, std::uint8_t contract,
                          const RoundResult& res, int team) const;

    // Local sensitivity d(equity)/d(game_point_margin) at (our_score, opp_score, hanging, we_lead),
    // used to align deal-margin Q-GRU values with match-equity search values.
    double marginSlope(int our_score, int opp_score, int hanging, bool we_lead) const;

private:
    MatchEquityTable();
    static int hangBucket(int h) { return h <= 0 ? 0 : (h <= 18 ? 1 : 2); }
    static std::size_t idx(int a, int b, int hb, int lead) {
        return (((std::size_t(a) * (kMaxScore + 1) + std::size_t(b)) * kHangBuckets + std::size_t(hb)) * 2) +
               std::size_t(lead);
    }

    std::vector<double> v_;  // [201][201][3][2]
};

}  // namespace bg
