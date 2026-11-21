#pragma once

// High-throughput C++ batched self-play environment for M2 DMC Q + BeliefNet
// training (PLAN.md §3.1, §5 M2).
//
// Maintains B independent deals always in Phase::kPlay (auto-dealing and
// auto-bidding with ruleBid + configurable bid jitter, skipping all-passes).
// Releases the GIL when called from Python so PyTorch GPU forward passes and
// C++ stepping/encoding pipeline cleanly.

#include "belote/deal.h"
#include "belote/encoder.h"
#include "belote/rules.h"

#include <cstdint>
#include <random>
#include <vector>

namespace belote {

inline constexpr int kMaxLegal = 8;

struct BatchStepResult {
    std::vector<std::uint8_t> done;  // 1 if env b finished a deal on this step
    std::vector<float> reward_t0;    // unitMargin(deal, 0) / 25.0f when done[b] == 1
};

class BatchPlayEnv {
public:
    BatchPlayEnv(int num_envs, const Config& cfg = Config::belgian(), std::uint64_t seed = 1,
                 double bid_threshold = 5.0, double bid_jitter = 0.8, int threads = 4);

    int numEnvs() const { return num_envs_; }
    const Config& config() const { return cfg_; }

    // Fills caller-allocated contiguous buffers for all num_envs_ deals:
    //  - state:          [B, kStateDim]
    //  - history:        [B, kMaxPlies, kPlyDim] (if non-null)
    //  - n_plies:        [B]
    //  - legal_actions:  [B, kMaxLegal, kActionDim]
    //  - legal_cards:    [B, kMaxLegal] (raw card ids 0..31, -1 padded)
    //  - n_legal:        [B]
    //  - current_team:   [B] (0 or 1)
    //  - belief_targets: [B, 32] (-1 if known/played, else 0/1/2 for rel_seat 1/2/3)
    //  - belief_allowed: [B, 32, 3] (1 if rel_seat k+1 may hold canon card c per inferConstraints)
    void observeBatch(float* state, float* history, std::int32_t* n_plies, float* legal_actions,
                      std::int32_t* legal_cards, std::int32_t* n_legal, std::int32_t* current_team,
                      std::int8_t* belief_targets, std::uint8_t* belief_allowed) const;

    // Steps every env b by playing legal_cards[b, action_idx[b]].
    // Finished deals write done[b] = 1, reward_t0[b] = unitMargin(d, 0) / 25.0f,
    // and immediately reset to a fresh bid-and-ready deal.
    void stepBatch(const std::int32_t* action_idx, std::uint8_t* done_out, float* reward_t0_out);

    // Resets all envs with a new seed (useful for reproducible validation sets).
    void resetAll(std::uint64_t seed);

    const Deal& deal(int env) const { return deals_[std::size_t(env)]; }

private:
    Deal makeFreshDeal(std::mt19937_64& rng) const;

    int num_envs_ = 0;
    int threads_ = 1;
    Config cfg_;
    double bid_threshold_ = 5.0;
    double bid_jitter_ = 0.8;
    std::vector<std::mt19937_64> rngs_;
    std::vector<Deal> deals_;
    std::vector<std::array<std::int32_t, kMaxLegal>> cached_legal_;
    std::vector<int> cached_n_legal_;
};

}  // namespace belote
