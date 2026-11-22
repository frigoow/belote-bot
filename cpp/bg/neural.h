#pragma once

// Zero-dependency plain-C++ inference for our Manille v1.2.3 Bulgarian Belot
// neural stack:
//  - BgFastPolicyModel (BGP1): 611 -> 256 -> 128 -> 32 sparse-input ReLU MLP
//    for ultra-fast rollout play-outs (~1.5 us/card) + play-history likelihood.
//  - BgBeliefModel (BGB1): Residual BeliefNet B_phi predicting logit deltas on
//    top of DealSampler::exactMarginals(), wired directly into
//    DealSampler::sampleWeighted() (true Manille Belief-DP, ~0.15 ms/root).
//  - BgQGRUModel (BGQ1): Dueling Q-GRU blueprint for root Q-blend + late-deal
//    taper (~0.3 ms/root).

#include "bg/bg.h"
#include "bg/encoder.h"
#include "bg/view.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace bg {

class BgFastPolicyModel {
public:
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }

    // Writes 32 logits in canonical-card order into logits_out[32].
    void forwardState(const float* state, float* logits_out) const;

    // Picks the highest-logit legal card for `r` (or `ps`).
    int chooseCard(const PublicState& ps, const Facts* facts = nullptr) const;
    int chooseCard(const Round& r) const;

    // Returns log-probability of `card` under softmax(logits / temp) over legal moves.
    double legalLogProb(const PublicState& ps, int card, double temp = 1.0, const Facts* facts = nullptr) const;

private:
    bool loaded_ = false;
    int state_dim_ = 0, h1_ = 0, h2_ = 0, out_dim_ = 0;
    std::vector<float> w1_t_;  // [state_dim, h1] transposed for sparse-input FMA
    std::vector<float> b1_;
    std::vector<float> w2_;    // [h2, h1]
    std::vector<float> b2_;
    std::vector<float> w3_;    // [out_dim, h2]
    std::vector<float> b3_;
};

namespace detail {

struct Linear {
    int in_dim = 0, out_dim = 0;
    std::vector<float> w;  // [out_dim, in_dim]
    std::vector<float> b;  // [out_dim]
};

struct LayerNorm {
    int dim = 0;
    std::vector<float> w;
    std::vector<float> b;
};

struct ResBlock {
    Linear fc1, fc2;
    LayerNorm ln1, ln2;
};

struct GruLayer {
    int in_dim = 0, hidden = 0;
    std::vector<float> w_ih;       // [3*hidden, in_dim]
    std::vector<float> w_hh;       // [3*hidden, hidden]
    std::vector<float> b_ih;       // [3*hidden]
    std::vector<float> b_hh;       // [3*hidden]
    std::vector<float> empty_emb;  // [hidden]
};

}  // namespace detail

class BgBeliefModel {
public:
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }

    // Raw forward pass matching Python BgBeliefNet::forward (returns 32x3 logits in canon-card, rel-seat order).
    void forwardLogits(const float* state, const float* history, int n_plies,
                       const float* sampler_marginals_canon, float* logits_out) const;

    // Computes per-card seat probabilities mapped directly to DealSampler::others()
    // order [raw_card][sampler_k] for DealSampler::sampleWeighted().
    std::array<std::array<double, 3>, kCards> samplerWeights(const PublicState& ps, const DealSampler& sampler,
                                                             double temp = 1.0) const;

private:
    bool loaded_ = false;
    int state_dim_ = 0, ply_dim_ = 0, state_hidden_ = 0, gru_hidden_ = 0;
    detail::Linear state_fc_;
    detail::LayerNorm state_ln_;
    detail::GruLayer hist_enc_;
    detail::ResBlock trunk1_, trunk2_;
    detail::Linear delta_head_;
};

class BgQGRUModel {
public:
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }

    // Evaluates Dueling Q(s, h, a) for all n_legal candidate actions (returns Q in scaled units, margin / 40).
    void forwardQ(const float* state, const float* history, int n_plies, const float* legal_actions, int n_legal,
                  float* q_out) const;

    // Convenience: returns (raw_card, expected game-point margin = 40 * Q) for every legal move of ps.
    std::vector<std::pair<int, double>> evalLegal(const PublicState& ps, const Facts* facts = nullptr) const;

    int chooseCard(const PublicState& ps, const Facts* facts = nullptr) const;

private:
    bool loaded_ = false;
    int state_dim_ = 0, ply_dim_ = 0, action_dim_ = 0;
    int state_hidden_ = 0, gru_hidden_ = 0, act_hidden_ = 0, adv_hidden_ = 0;
    detail::Linear state_fc_;
    detail::LayerNorm state_ln_;
    detail::GruLayer hist_enc_;
    detail::ResBlock trunk1_, trunk2_;
    // DuelingQHead
    detail::Linear v_fc1_, v_fc2_;
    detail::LayerNorm v_ln1_;
    detail::Linear act_fc_;
    detail::LayerNorm act_ln_;
    detail::Linear adv_fc1_, adv_fc2_;
    detail::LayerNorm adv_ln1_;
    detail::ResBlock adv_res_;
};

}  // namespace bg
