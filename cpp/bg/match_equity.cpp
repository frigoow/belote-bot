#include "bg/match_equity.h"
#include "bg/agent.h"
#include "bg/rule_policy.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <tuple>

namespace bg {

namespace {

struct Outcome {
    int lead_pts = 0;
    int other_pts = 0;
    int new_hb = 0;
    bool no_tricks = false;
    double prob = 0.0;
};

// Generates compact empirical transition distributions for each incoming hanging bucket
// by simulating fast Bulgarian deals under naturalBid + rulePolicy.
std::array<std::vector<Outcome>, MatchEquityTable::kHangBuckets> buildKernel() {
    static constexpr int kRepHang[MatchEquityTable::kHangBuckets] = {0, 12, 26};
    static constexpr int kDealsPerBucket = 1200;
    std::array<std::vector<Outcome>, MatchEquityTable::kHangBuckets> kernels;
    std::mt19937_64 rng(20261121ull);
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);

    for (int hb = 0; hb < MatchEquityTable::kHangBuckets; ++hb) {
        std::map<std::tuple<int, int, int, bool>, int> counts;
        int total = 0;
        for (int g = 0; g < kDealsPerBucket; ++g) {
            std::shuffle(deck.begin(), deck.end(), rng);
            Round r(deck, 0, kRepHang[hb]);
            while (r.decision() == Round::Decision::kBid) r.bid(naturalBid(r));
            if (r.finished()) {
                // Passed-out deal keeps hanging points and rotates lead; no one scores.
                ++counts[{0, 0, hb, true}];  // no_tricks=true prevents game termination on pass
                ++total;
                continue;
            }
            while (!r.finished()) {
                if (r.decision() == Round::Decision::kAnnounce) {
                    r.declare(chooseDeclarations(r.offeredAnnounces()));
                } else {
                    r.play(rulePolicy(r), true);
                }
            }
            const RoundResult& res = r.result();
            // Coarsen points slightly (step 2) to keep transition kernel compact (~120 entries).
            const int lp = (res.sn / 2) * 2;
            const int op = (res.ew / 2) * 2;
            const int nhb = res.hanging <= 0 ? 0 : (res.hanging <= 18 ? 1 : 2);
            ++counts[{lp, op, nhb, res.no_tricks}];
            ++total;
        }
        auto& out = kernels[std::size_t(hb)];
        out.reserve(counts.size());
        for (const auto& [key, cnt] : counts) {
            const auto& [lp, op, nhb, nt] = key;
            out.push_back({lp, op, nhb, nt, double(cnt) / double(total)});
        }
    }
    return kernels;
}

}  // namespace

const MatchEquityTable& MatchEquityTable::instance() {
    static const MatchEquityTable table;
    return table;
}

MatchEquityTable::MatchEquityTable()
    : v_(std::size_t(kMaxScore + 1) * std::size_t(kMaxScore + 1) * kHangBuckets * 2, 0.5) {
    const auto kernels = buildKernel();

    auto lookup = [&](int a, int b, int hb, int lead) -> double {
        // Above 151, if both teams are past 151 only the score difference matters.
        if (a > kMaxScore || b > kMaxScore) {
            const int excess = std::max(a - kMaxScore, b - kMaxScore);
            if (a - excess >= 151 && b - excess >= 151) {
                a -= excess;
                b -= excess;
            } else {
                a = std::min(a, kMaxScore);
                b = std::min(b, kMaxScore);
            }
        }
        a = std::clamp(a, 0, kMaxScore);
        b = std::clamp(b, 0, kMaxScore);
        return v_[idx(a, b, hb, lead)];
    };

    // Initialize with smooth heuristic prior for overtime states (a>=151 && b>=151).
    for (int a = 0; a <= kMaxScore; ++a) {
        for (int b = 0; b <= kMaxScore; ++b) {
            const double p = 1.0 / (1.0 + std::exp(-0.045 * double(a - b)));
            for (int hb = 0; hb < kHangBuckets; ++hb) {
                v_[idx(a, b, hb, 0)] = p;
                v_[idx(a, b, hb, 1)] = p;
            }
        }
    }

    // Solve overtime / high-score region (a >= 151 || b >= 151) with Gauss-Seidel sweeps,
    // then exact backward DP from sum = 300 down to 0 (with a few extra sweeps for pass/hanging cycles).
    auto updateCell = [&](int a, int b, int hb, int lead) -> double {
        double ev = 0.0;
        for (const Outcome& o : kernels[std::size_t(hb)]) {
            const int ga = lead ? o.lead_pts : o.other_pts;
            const int gb = lead ? o.other_pts : o.lead_pts;
            const int na = a + ga;
            const int nb = b + gb;
            const bool decided = !o.no_tricks;
            double val;
            if (decided && na >= 151 && na > nb && ga > 0) val = 1.0;
            else if (decided && nb >= 151 && nb > na && gb > 0) val = 0.0;
            else val = lookup(na, nb, o.new_hb, 1 - lead);
            ev += o.prob * val;
        }
        return ev;
    };

    for (int sweep = 0; sweep < 25; ++sweep) {
        double max_diff = 0.0;
        for (int sum = 2 * kMaxScore; sum >= 0; --sum) {
            const int a_min = std::max(0, sum - kMaxScore);
            const int a_max = std::min(kMaxScore, sum);
            for (int a = a_min; a <= a_max; ++a) {
                const int b = sum - a;
                for (int hb = kHangBuckets - 1; hb >= 0; --hb) {
                    for (int lead = 0; lead < 2; ++lead) {
                        const double nv = updateCell(a, b, hb, lead);
                        double& cell = v_[idx(a, b, hb, lead)];
                        max_diff = std::max(max_diff, std::abs(nv - cell));
                        cell = nv;
                    }
                }
            }
        }
        if (max_diff < 1e-5) break;
    }
}

double MatchEquityTable::preDealEquity(int our_score, int opp_score, int hanging, bool we_lead) const {
    int a = our_score, b = opp_score;
    if (a > kMaxScore || b > kMaxScore) {
        const int excess = std::max(a - kMaxScore, b - kMaxScore);
        if (a - excess >= 151 && b - excess >= 151) {
            a -= excess;
            b -= excess;
        } else {
            a = std::min(a, kMaxScore);
            b = std::min(b, kMaxScore);
        }
    }
    a = std::clamp(a, 0, kMaxScore);
    b = std::clamp(b, 0, kMaxScore);
    return v_[idx(a, b, hangBucket(hanging), we_lead ? 1 : 0)];
}

double MatchEquityTable::postDealEquity(int sn_before, int ew_before, int first_before, std::uint8_t contract,
                                        const RoundResult& res, int team) const {
    GameScore gs;
    gs.sn = sn_before;
    gs.ew = ew_before;
    gs.first = first_before;
    gs.record(contract, res);
    if (gs.winner >= 0) return gs.winner == team ? 1.0 : 0.0;
    const int our = team == 0 ? gs.sn : gs.ew;
    const int opp = team == 0 ? gs.ew : gs.sn;
    const bool we_lead = (teamOf(gs.first) == team);
    return preDealEquity(our, opp, gs.hanging, we_lead);
}

double MatchEquityTable::marginSlope(int our_score, int opp_score, int hanging, bool we_lead) const {
    // Symmetric finite difference over +-10 game-point margin around current score.
    const double e_plus = preDealEquity(our_score + 10, opp_score, hanging, we_lead);
    const double e_minus = preDealEquity(our_score, opp_score + 10, hanging, we_lead);
    return std::max(1e-4, (e_plus - e_minus) / 20.0);
}

}  // namespace bg
