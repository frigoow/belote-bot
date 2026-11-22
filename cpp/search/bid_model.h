#pragma once

// Plain-C++ M3 Bidding Stack (PLAN.md §4.2):
// - MatchEquityTable V[us, them, pot, we_deal] + overtime V_tie
// - S_4-equivariant DealOutcomeNet P(bucket | hand_5, blik, cand_trump, node_k)
// - Residual-anchored score-aware BidNet [V(Pass), V(Take)]
// - Optional Stage 3 test-time auction search over ordered 26-card deck completions.

#include "belote/deal.h"
#include "search/fast_policy.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace belote {

inline constexpr int kAuctionFeatDim = 59;
inline constexpr int kMatchCtxDim = 10;

struct OutcomeBucket {
    int du_taker = 0;
    int dt_def = 0;
    int held = 0;
    int taker_won = 0;
};

struct BidEval {
    Bid best_bid = Bid::pass();
    Bid best_take = Bid::pass();
    double v_pass = 0.5;
    double v_take = 0.5;
    double ev_take = 0.5;
    double v_redeal = 0.5;
    double best_r2_ev = 0.5;
};

// Exact S_4-equivariant encoding matching python/belote/bidding.py::encode_auction_take.
void encodeAuctionTake(Mask hand_5, int blik, int candidate_trump, int node_k, float* out);

class BidModel {
public:
    static BidModel load(const std::string& path);

    bool loaded() const { return target_ > 0; }
    int target() const { return target_; }
    int margin() const { return margin_; }
    int potCap() const { return pot_cap_; }
    int numBuckets() const { return int(buckets_.size()); }

    // Exact match equity lookup V[us, them, pot, we_deal], including overtime margin.
    double lookupV(int us, int them, int pot, int we_deal) const;

    // Post-deal match equity for `my_team` after `d` finishes (kDone) starting from (us, them, pot).
    double evalDealEquity(const DealResult& res, const Config& cfg, int dealer, int my_team, int us, int them,
                          int pot) const;

    // DealOutcomeNet probability distribution over buckets for taking `candidate_trump`.
    void predictOutcomeProbs(Mask hand_5, int blik, int candidate_trump, int node_k, float* probs_out) const;

    // Exact E[V | Take(candidate_trump)] at (us, them, pot, we_deal).
    double expectedVTake(Mask hand_5, int blik, int candidate_trump, int node_k, int us, int them, int pot,
                         int we_deal) const;

    // Full BidNet evaluation at the current auction state (deal in kBid1/kBid2).
    BidEval evalBid(const Deal& d, int us, int them, int pot) const;

    // Fast score-aware bid choice using DealOutcomeNet + BidNet (<0.15 ms/decision).
    Bid chooseBid(const Deal& d, int us = 0, int them = 0, int pot = 0) const;

    // Stage 3 test-time auction search (PLAN.md §4.2):
    // When |v_take - v_pass| < search_margin, samples `worlds` ordered completions
    // of the 26 unseen cards, completes each auction & deal with BidModel + FastPolicy,
    // and picks the legal bid maximizing exact post-deal match equity V.
    Bid chooseBidSearch(const Deal& d, const FastPolicyModel& play_policy, int us, int them, int pot,
                        int worlds = 200, double search_margin = 0.04, std::uint64_t seed = 0) const;

    // Log-probability of a seat making `chosen` bid given its initial 5-card hand,
    // blik, and auction node_k (for M4 auction likelihood weighting).
    double bidLogProb(Mask hand_5, int blik, int node_k, const Config& cfg, Bid chosen, int us, int them, int pot,
                      int we_deal, double temperature = 0.03) const;

private:
    struct Mlp3 {
        int in_dim = 0;
        int h1 = 0;
        int h2 = 0;
        int out_dim = 0;
        std::vector<float> w1, b1, ln1_w, ln1_b;
        std::vector<float> w2, b2, ln2_w, ln2_b;
        std::vector<float> w3, b3;

        void forward(const float* x, float* out) const;
    };

    void evalContext(int us, int them, int pot, int we_deal, double ev_take, double best_r2_ev, float* out) const;
    void evalBidNetPair(const float* af, const float* ctx, int node_k, double ev_take, double v_redeal,
                        double& v_pass_out, double& v_take_out) const;

    int feat_dim_ = kAuctionFeatDim;
    int ctx_dim_ = kMatchCtxDim;
    int target_ = 0;
    int margin_ = 2;
    int pot_cap_ = 64;
    std::vector<OutcomeBucket> buckets_;
    std::vector<float> v_table_;  // [target][target][pot_cap + 1][2]
    std::vector<float> v_tie_;    // [2*margin - 1][pot_cap + 1][2]
    Mlp3 don_;
    Mlp3 bidnet_;
};

}  // namespace belote
