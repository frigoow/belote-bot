#pragma once

// Compact 256x128 ReLU FastPolicy in plain C++ (no external runtime needed),
// loaded from FastPolicy.export_bin() ("BFP1" flat FP32 format), plus the
// M2 FastPolicy + WorldSampler + Candidate Racing + exact AB hybrid search.

#include "belote/deal.h"
#include "belote/encoder.h"
#include "belote/observation.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace belote {

class FastPolicyModel {
public:
    static std::shared_ptr<const FastPolicyModel> load(const std::string& path);

    int stateDim() const { return state_dim_; }
    int hidden1() const { return h1_; }
    int hidden2() const { return h2_; }

    // Evaluates logits[32] in canonical card space for an encoded state[state_dim_].
    void forward(const float* state, float* logits_out) const;

    // Evaluates logits[32] directly from an Observation.
    void evalObservation(const Observation& obs, float* logits_out) const;

    // Greedy legal card from a Deal (applies Belot guard under belot_queen_first_forfeits).
    int chooseCard(const Deal& deal) const;

    // Log-probability of a specific card played at obs (masked softmax over legal moves),
    // used for FastPolicy play-history world weighting (PLAN.md §3.2).
    double logProbOfPlay(const Observation& obs, int card, double temperature = 1.0) const;

private:
    int state_dim_ = 0;
    int h1_ = 0;
    int h2_ = 0;
    int out_dim_ = 0;
    // Column-major transposed weights [in_dim][out_dim] for sparse-input AVX2 GEMV.
    std::vector<float> w1_t_, b1_;
    std::vector<float> w2_t_, b2_;
    std::vector<float> w3_t_, b3_;
};

struct HybridOptions {
    int worlds = 64;                 // max worlds (or fixed worlds if race == false)
    int ab_handoff_trick = 4;        // >= 0: exact AB endgame from this trick on
    bool weight_play_history = true; // weight sampled worlds by FastPolicy play likelihood
    double play_lik_temp = 1.0;      // softmax temperature for play-history log-likelihood
    bool race = false;               // Manille-style CRN paired-z candidate racing
    int race_min_worlds = 24;        // minimum worlds before pruning candidates
    int race_step = 16;              // evaluate this many worlds per racing round
    double race_z = 2.5;             // prune move if leader beats it by > race_z * SE
    std::uint64_t seed = 0;
};

struct HybridResult {
    int best_move = kNoCard;
    double value = 0;
    int worlds_used = 0;
    bool exact_enumeration = false;
    std::vector<std::pair<int, double>> move_values;
};

// Shared-world FastPolicy play-outs + optional FastPolicy play-history
// likelihood weighting + optional CRN candidate racing + exact AB endgame.
HybridResult hybridSearch(const Deal& deal, const FastPolicyModel& policy, const HybridOptions& options = {});

}  // namespace belote
