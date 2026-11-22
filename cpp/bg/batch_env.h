#pragma once

// High-throughput multithreaded C++ vectorized self-play environment for
// Bulgarian Bridge-Belot DMC Q-GRU + Residual BeliefNet B_phi training
// (Manille v1.2.3 architecture).
//
// Maintains B independent bg::Round instances always in Decision::kPlay
// (auto-bidding with naturalBid + realistic threshold jitter + auto-declaring,
// skipping passed-out deals). Each env resets immediately on deal finish.

#include "bg/bg.h"
#include "bg/bid_model.h"
#include "bg/encoder.h"
#include "bg/view.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace bg {

enum class ContractMode : int {
    kAll = 0,
    kSuit = 1,
    kNoTrumps = 2,
    kAllTrumps = 3,
};

class BgBatchPlayEnv {
public:
    BgBatchPlayEnv(int num_envs, std::uint64_t seed = 1, double bid_noise = 0.0, double hanging_prob = 0.0,
                   int threads = 8, ContractMode mode = ContractMode::kAll, bool allow_double = false,
                   const std::string& bid_model_path = "", double bid_model_prob = 0.65);

    int numEnvs() const { return num_envs_; }

    // Fills caller-allocated contiguous buffers for all num_envs_ active decisions:
    //  - state:            [B, kBgStateDim]
    //  - history:          [B, kBgMaxPlies, kBgPlyDim] (if non-null)
    //  - n_plies:          [B] (int32)
    //  - legal_actions:    [B, kBgMaxLegal, kBgActionDim]
    //  - legal_cards:      [B, kBgMaxLegal] (raw card ids 0..31, -1 padded)
    //  - n_legal:          [B] (int32)
    //  - current_team:     [B] (int32, 0 or 1)
    //  - contract_out:     [B] (uint8, optional)
    //  - belief_targets:   [B, 32] (int8, optional: -1 if own/played, else 0/1/2 for relSeat 1/2/3)
    //  - belief_marginals: [B, 32, 3] (float32, optional: exact DealSampler::exactMarginals() in canon-card order!)
    //  - belief_allowed:   [B, 32, 3] (uint8, optional: 1 if exactMarginal > 0)
    void observeBatch(float* state, float* history, std::int32_t* n_plies, float* legal_actions,
                      std::int32_t* legal_cards, std::int32_t* n_legal, std::int32_t* current_team,
                      std::uint8_t* contract_out, std::int8_t* belief_targets, float* belief_marginals,
                      std::uint8_t* belief_allowed) const;

    // Steps every env b by playing legal_cards[b, action_idx[b]] (claiming belote when allowed).
    // When a deal finishes, writes done_out[b] = 1, reward_t0_out[b] = (res.sn - res.ew) / 40.0f,
    // and resets env b to a fresh bid-and-ready deal in Decision::kPlay.
    void stepBatch(const std::int32_t* action_idx, std::uint8_t* done_out, float* reward_t0_out);

    void resetAll(std::uint64_t seed);
    const Round& deal(int env) const { return deals_[std::size_t(env)]; }

private:
    Round makeFreshDeal(std::mt19937_64& rng) const;
    void cacheLegal(int env);

    int num_envs_ = 0;
    int threads_ = 1;
    double bid_noise_ = 0.0;
    double hanging_prob_ = 0.0;
    ContractMode mode_ = ContractMode::kAll;
    bool allow_double_ = false;
    double bid_model_prob_ = 0.65;
    BgBidModel bid_model_;
    std::vector<std::mt19937_64> rngs_;
    std::vector<Round> deals_;
    std::vector<std::array<std::int32_t, kBgMaxLegal>> cached_legal_;
    std::vector<int> cached_n_legal_;
};

}  // namespace bg
