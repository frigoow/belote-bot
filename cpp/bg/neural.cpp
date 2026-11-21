#include "bg/neural.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace bg {

namespace {

bool readBytes(std::ifstream& in, void* dst, std::size_t n) {
    in.read(reinterpret_cast<char*>(dst), std::streamsize(n));
    return bool(in);
}

bool readLinear(std::ifstream& in, detail::Linear& l, int in_dim, int out_dim) {
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    l.w.resize(std::size_t(out_dim) * std::size_t(in_dim));
    l.b.resize(std::size_t(out_dim));
    return readBytes(in, l.w.data(), l.w.size() * sizeof(float)) &&
           readBytes(in, l.b.data(), l.b.size() * sizeof(float));
}

bool readLN(std::ifstream& in, detail::LayerNorm& ln, int dim) {
    ln.dim = dim;
    ln.w.resize(std::size_t(dim));
    ln.b.resize(std::size_t(dim));
    return readBytes(in, ln.w.data(), ln.w.size() * sizeof(float)) &&
           readBytes(in, ln.b.data(), ln.b.size() * sizeof(float));
}

bool readResBlock(std::ifstream& in, detail::ResBlock& rb, int dim) {
    return readLinear(in, rb.fc1, dim, dim) && readLN(in, rb.ln1, dim) &&
           readLinear(in, rb.fc2, dim, dim) && readLN(in, rb.ln2, dim);
}

bool readGru(std::ifstream& in, detail::GruLayer& g, int in_dim, int hidden) {
    g.in_dim = in_dim;
    g.hidden = hidden;
    g.w_ih.resize(std::size_t(3 * hidden) * std::size_t(in_dim));
    g.w_hh.resize(std::size_t(3 * hidden) * std::size_t(hidden));
    g.b_ih.resize(std::size_t(3 * hidden));
    g.b_hh.resize(std::size_t(3 * hidden));
    g.empty_emb.resize(std::size_t(hidden));
    return readBytes(in, g.w_ih.data(), g.w_ih.size() * sizeof(float)) &&
           readBytes(in, g.w_hh.data(), g.w_hh.size() * sizeof(float)) &&
           readBytes(in, g.b_ih.data(), g.b_ih.size() * sizeof(float)) &&
           readBytes(in, g.b_hh.data(), g.b_hh.size() * sizeof(float)) &&
           readBytes(in, g.empty_emb.data(), g.empty_emb.size() * sizeof(float));
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void applyLinear(const detail::Linear& l, const float* x, float* y) {
    const int in_d = l.in_dim;
    const int out_d = l.out_dim;
    const float* w = l.w.data();
    const float* b = l.b.data();
    for (int o = 0; o < out_d; ++o) {
        const float* row = w + std::size_t(o) * std::size_t(in_d);
        float acc = b[o];
        for (int i = 0; i < in_d; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

void applyLN(const detail::LayerNorm& ln, float* x) {
    const int d = ln.dim;
    float mean = 0.0f;
    for (int i = 0; i < d; ++i) mean += x[i];
    mean /= float(d);
    float var = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= float(d);
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    const float* w = ln.w.data();
    const float* b = ln.b.data();
    for (int i = 0; i < d; ++i) x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

void applyResBlock(const detail::ResBlock& rb, const float* x, float* out, std::vector<float>& scratch) {
    const int d = rb.fc1.out_dim;
    scratch.resize(std::size_t(d));
    applyLinear(rb.fc1, x, scratch.data());
    applyLN(rb.ln1, scratch.data());
    for (int i = 0; i < d; ++i) scratch[std::size_t(i)] = silu(scratch[std::size_t(i)]);
    applyLinear(rb.fc2, scratch.data(), out);
    applyLN(rb.ln2, out);
    for (int i = 0; i < d; ++i) out[i] = silu(x[i] + out[i]);
}

void applyGru(const detail::GruLayer& g, const float* history, int n_plies, float* h_out) {
    const int H = g.hidden;
    const int I = g.in_dim;
    if (n_plies <= 0) {
        std::memcpy(h_out, g.empty_emb.data(), std::size_t(H) * sizeof(float));
        return;
    }
    std::memset(h_out, 0, std::size_t(H) * sizeof(float));
    std::vector<float> gi(std::size_t(3 * H));
    std::vector<float> gh(std::size_t(3 * H));

    for (int t = 0; t < n_plies; ++t) {
        const float* x = history + std::size_t(t * I);
        for (int o = 0; o < 3 * H; ++o) {
            const float* wir = g.w_ih.data() + std::size_t(o) * std::size_t(I);
            float ai = g.b_ih[std::size_t(o)];
            for (int i = 0; i < I; ++i) {
                if (x[i] != 0.0f) ai += wir[i] * x[i];
            }
            gi[std::size_t(o)] = ai;

            const float* whr = g.w_hh.data() + std::size_t(o) * std::size_t(H);
            float ah = g.b_hh[std::size_t(o)];
            for (int i = 0; i < H; ++i) ah += whr[i] * h_out[i];
            gh[std::size_t(o)] = ah;
        }
        for (int j = 0; j < H; ++j) {
            const float r = sigmoid(gi[std::size_t(j)] + gh[std::size_t(j)]);
            const float z = sigmoid(gi[std::size_t(H + j)] + gh[std::size_t(H + j)]);
            const float n = std::tanh(gi[std::size_t(2 * H + j)] + r * gh[std::size_t(2 * H + j)]);
            h_out[j] = (1.0f - z) * n + z * h_out[j];
        }
    }
}

Trick currentTrick(const PublicState& ps) {
    Trick t;
    const std::size_t rem = ps.plays.size() % kPlayers;
    const std::size_t base = ps.plays.size() - rem;
    for (std::size_t i = 0; i < rem; ++i) {
        t.cards[i] = ps.plays[base + i].card;
        t.players[i] = ps.plays[base + i].seat;
    }
    t.n = int(rem);
    return t;
}

}  // namespace

// ================================================================== BgFastPolicyModel

bool BgFastPolicyModel::load(const std::string& path) {
    loaded_ = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4]{};
    std::int32_t hdr[5]{};
    if (!readBytes(in, magic, 4) || std::memcmp(magic, "BGP1", 4) != 0) return false;
    if (!readBytes(in, hdr, sizeof(hdr))) return false;
    if (hdr[0] != kBgEncoderVersion || hdr[1] != kBgStateDim || hdr[4] != 32) return false;
    state_dim_ = hdr[1];
    h1_ = hdr[2];
    h2_ = hdr[3];
    out_dim_ = hdr[4];

    std::vector<float> w1(std::size_t(h1_) * std::size_t(state_dim_));
    b1_.resize(std::size_t(h1_));
    w2_.resize(std::size_t(h2_) * std::size_t(h1_));
    b2_.resize(std::size_t(h2_));
    w3_.resize(std::size_t(out_dim_) * std::size_t(h2_));
    b3_.resize(std::size_t(out_dim_));

    if (!readBytes(in, w1.data(), w1.size() * sizeof(float)) ||
        !readBytes(in, b1_.data(), b1_.size() * sizeof(float)) ||
        !readBytes(in, w2_.data(), w2_.size() * sizeof(float)) ||
        !readBytes(in, b2_.data(), b2_.size() * sizeof(float)) ||
        !readBytes(in, w3_.data(), w3_.size() * sizeof(float)) ||
        !readBytes(in, b3_.data(), b3_.size() * sizeof(float))) {
        return false;
    }

    // Transpose W1 [h1, state_dim] -> W1_T [state_dim, h1] for fast sparse-input FMA.
    w1_t_.resize(std::size_t(state_dim_) * std::size_t(h1_));
    for (int j = 0; j < h1_; ++j) {
        for (int i = 0; i < state_dim_; ++i) {
            w1_t_[std::size_t(i) * std::size_t(h1_) + std::size_t(j)] =
                w1[std::size_t(j) * std::size_t(state_dim_) + std::size_t(i)];
        }
    }
    loaded_ = true;
    return true;
}

void BgFastPolicyModel::forwardState(const float* state, float* logits_out) const {
    alignas(32) float z1[512];
    alignas(32) float z2[256];
    std::memcpy(z1, b1_.data(), std::size_t(h1_) * sizeof(float));
    for (int i = 0; i < state_dim_; ++i) {
        const float xi = state[i];
        if (xi == 0.0f) continue;
        const float* col = w1_t_.data() + std::size_t(i) * std::size_t(h1_);
        for (int j = 0; j < h1_; ++j) z1[j] += xi * col[j];
    }
    for (int j = 0; j < h1_; ++j)
        if (z1[j] < 0.0f) z1[j] = 0.0f;

    for (int o = 0; o < h2_; ++o) {
        const float* row = w2_.data() + std::size_t(o) * std::size_t(h1_);
        float acc = b2_[std::size_t(o)];
        for (int j = 0; j < h1_; ++j) acc += row[j] * z1[j];
        z2[o] = acc > 0.0f ? acc : 0.0f;
    }

    for (int o = 0; o < out_dim_; ++o) {
        const float* row = w3_.data() + std::size_t(o) * std::size_t(h2_);
        float acc = b3_[std::size_t(o)];
        for (int j = 0; j < h2_; ++j) acc += row[j] * z2[j];
        logits_out[o] = acc;
    }
}

int BgFastPolicyModel::chooseCard(const PublicState& ps, const Facts* facts) const {
    const Mask legal = legalCards(ps.hand, ps.contract, currentTrick(ps));
    if (!(legal & (legal - 1))) return lowestCard(legal);
    float state[kBgStateDim];
    float logits[32];
    encodeState(ps, state, facts);
    forwardState(state, logits);
    int best = kNoCard;
    float best_v = -1e30f;
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        const float v = logits[canonCard(c, ps.contract)];
        if (v > best_v) {
            best_v = v;
            best = c;
        }
    }
    return best;
}

int BgFastPolicyModel::chooseCard(const Round& r) const {
    const Mask legal = r.legalMoves();
    if (!(legal & (legal - 1))) return lowestCard(legal);
    return chooseCard(publicState(r, r.toMove()));
}

double BgFastPolicyModel::legalLogProb(const PublicState& ps, int card, double temp, const Facts* facts) const {
    const Mask legal = legalCards(ps.hand, ps.contract, currentTrick(ps));
    if (!(legal & (legal - 1))) return 0.0;
    float state[kBgStateDim];
    float logits[32];
    encodeState(ps, state, facts);
    forwardState(state, logits);
    const double inv_t = 1.0 / std::max(temp, 1e-6);
    double top = -1e30;
    for (Mask m = legal; m; m &= m - 1) {
        const double v = double(logits[canonCard(lowestCard(m), ps.contract)]) * inv_t;
        if (v > top) top = v;
    }
    double denom = 0.0;
    for (Mask m = legal; m; m &= m - 1) {
        const double v = double(logits[canonCard(lowestCard(m), ps.contract)]) * inv_t;
        denom += std::exp(v - top);
    }
    const double chosen_v = double(logits[canonCard(card, ps.contract)]) * inv_t;
    return (chosen_v - top) - std::log(denom);
}

// ================================================================== BgBeliefModel

bool BgBeliefModel::load(const std::string& path) {
    loaded_ = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4]{};
    std::int32_t hdr[5]{};
    if (!readBytes(in, magic, 4) || std::memcmp(magic, "BGB1", 4) != 0) return false;
    if (!readBytes(in, hdr, sizeof(hdr))) return false;
    if (hdr[0] != kBgEncoderVersion || hdr[1] != kBgStateDim || hdr[2] != kBgPlyDim) return false;
    state_dim_ = hdr[1];
    ply_dim_ = hdr[2];
    state_hidden_ = hdr[3];
    gru_hidden_ = hdr[4];
    const int trunk_dim = state_hidden_ + gru_hidden_;

    if (!readLinear(in, state_fc_, state_dim_ + 96, state_hidden_) ||
        !readLN(in, state_ln_, state_hidden_) ||
        !readGru(in, hist_enc_, ply_dim_, gru_hidden_) ||
        !readResBlock(in, trunk1_, trunk_dim) ||
        !readResBlock(in, trunk2_, trunk_dim) ||
        !readLinear(in, delta_head_, trunk_dim, 32 * 3)) {
        return false;
    }
    loaded_ = true;
    return true;
}

void BgBeliefModel::forwardLogits(const float* state, const float* history, int n_plies,
                                  const float* sampler_marginals_canon, float* logits_out) const {
    const int trunk_dim = state_hidden_ + gru_hidden_;
    std::vector<float> s_in(static_cast<std::size_t>(state_dim_ + 96));
    std::memcpy(s_in.data(), state, std::size_t(state_dim_) * sizeof(float));
    std::memcpy(s_in.data() + state_dim_, sampler_marginals_canon, 96 * sizeof(float));

    std::vector<float> cat(static_cast<std::size_t>(trunk_dim));
    applyLinear(state_fc_, s_in.data(), cat.data());
    applyLN(state_ln_, cat.data());
    for (int i = 0; i < state_hidden_; ++i) cat[std::size_t(i)] = silu(cat[std::size_t(i)]);

    applyGru(hist_enc_, history, n_plies, cat.data() + state_hidden_);

    std::vector<float> h1(static_cast<std::size_t>(trunk_dim));
    std::vector<float> h2(static_cast<std::size_t>(trunk_dim));
    std::vector<float> scratch;
    applyResBlock(trunk1_, cat.data(), h1.data(), scratch);
    applyResBlock(trunk2_, h1.data(), h2.data(), scratch);

    float delta[96];
    applyLinear(delta_head_, h2.data(), delta);

    for (int i = 0; i < 96; ++i) {
        const float p = sampler_marginals_canon[i];
        if (p <= 0.0f) {
            logits_out[i] = -1e9f;
        } else {
            logits_out[i] = std::log(std::max(p, 1e-6f)) + delta[i];
        }
    }
}

std::array<std::array<double, 3>, kCards> BgBeliefModel::samplerWeights(const PublicState& ps,
                                                                        const DealSampler& sampler,
                                                                        double temp) const {
    const auto raw_marg = sampler.exactMarginals();
    const auto& others = sampler.others();
    int k_to_rk[3];
    for (int k = 0; k < 3; ++k) k_to_rk[k] = relSeat(others[std::size_t(k)], ps.me) - 1;

    float marg_canon[96]{};
    for (int raw_c = 0; raw_c < kCards; ++raw_c) {
        const int cc = canonCard(raw_c, ps.contract);
        for (int k = 0; k < 3; ++k) {
            marg_canon[cc * 3 + k_to_rk[k]] = float(raw_marg[std::size_t(raw_c)][std::size_t(k)]);
        }
    }

    float state[kBgStateDim];
    float history[kBgMaxPlies * kBgPlyDim];
    encodeState(ps, state, &sampler.facts());
    const int n_plies = encodeHistory(ps, history);

    float logits[96];
    forwardLogits(state, history, n_plies, marg_canon, logits);

    // Softmax per hidden card -> map back to [raw_card][sampler_k].
    std::array<std::array<double, 3>, kCards> weights{};
    const double inv_t = 1.0 / std::max(temp, 1e-6);
    for (int raw_c = 0; raw_c < kCards; ++raw_c) {
        const int cc = canonCard(raw_c, ps.contract);
        double l[3];
        double top = -1e30;
        bool any = false;
        for (int k = 0; k < 3; ++k) {
            const float m = marg_canon[cc * 3 + k_to_rk[k]];
            if (m > 0.0f) {
                l[k] = double(logits[cc * 3 + k_to_rk[k]]) * inv_t;
                if (l[k] > top) top = l[k];
                any = true;
            } else {
                l[k] = -1e9;
            }
        }
        if (!any) continue;
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
            if (l[k] > -1e8) {
                weights[std::size_t(raw_c)][std::size_t(k)] = std::exp(std::max(-30.0, l[k] - top));
                sum += weights[std::size_t(raw_c)][std::size_t(k)];
            }
        }
        if (sum > 0.0) {
            for (int k = 0; k < 3; ++k) weights[std::size_t(raw_c)][std::size_t(k)] /= sum;
        }
    }
    return weights;
}

// ================================================================== BgQGRUModel

bool BgQGRUModel::load(const std::string& path) {
    loaded_ = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4]{};
    std::int32_t hdr[8]{};
    if (!readBytes(in, magic, 4) || std::memcmp(magic, "BGQ1", 4) != 0) return false;
    if (!readBytes(in, hdr, sizeof(hdr))) return false;
    if (hdr[0] != kBgEncoderVersion || hdr[1] != kBgStateDim || hdr[2] != kBgPlyDim || hdr[3] != kBgActionDim)
        return false;
    state_dim_ = hdr[1];
    ply_dim_ = hdr[2];
    action_dim_ = hdr[3];
    state_hidden_ = hdr[4];
    gru_hidden_ = hdr[5];
    act_hidden_ = hdr[6];
    adv_hidden_ = hdr[7];
    const int trunk_dim = state_hidden_ + gru_hidden_;

    if (!readLinear(in, state_fc_, state_dim_, state_hidden_) ||
        !readLN(in, state_ln_, state_hidden_) ||
        !readGru(in, hist_enc_, ply_dim_, gru_hidden_) ||
        !readResBlock(in, trunk1_, trunk_dim) ||
        !readResBlock(in, trunk2_, trunk_dim) ||
        !readLinear(in, v_fc1_, trunk_dim, adv_hidden_) ||
        !readLN(in, v_ln1_, adv_hidden_) ||
        !readLinear(in, v_fc2_, adv_hidden_, 1) ||
        !readLinear(in, act_fc_, action_dim_, act_hidden_) ||
        !readLN(in, act_ln_, act_hidden_) ||
        !readLinear(in, adv_fc1_, trunk_dim + act_hidden_, adv_hidden_) ||
        !readLN(in, adv_ln1_, adv_hidden_) ||
        !readResBlock(in, adv_res_, adv_hidden_) ||
        !readLinear(in, adv_fc2_, adv_hidden_, 1)) {
        return false;
    }
    loaded_ = true;
    return true;
}

void BgQGRUModel::forwardQ(const float* state, const float* history, int n_plies, const float* legal_actions,
                           int n_legal, float* q_out) const {
    const int trunk_dim = state_hidden_ + gru_hidden_;
    std::vector<float> cat(static_cast<std::size_t>(trunk_dim));
    applyLinear(state_fc_, state, cat.data());
    applyLN(state_ln_, cat.data());
    for (int i = 0; i < state_hidden_; ++i) cat[std::size_t(i)] = silu(cat[std::size_t(i)]);
    applyGru(hist_enc_, history, n_plies, cat.data() + state_hidden_);

    std::vector<float> h1(static_cast<std::size_t>(trunk_dim));
    std::vector<float> trunk(static_cast<std::size_t>(trunk_dim));
    std::vector<float> scratch;
    applyResBlock(trunk1_, cat.data(), h1.data(), scratch);
    applyResBlock(trunk2_, h1.data(), trunk.data(), scratch);

    // Value head: V(s)
    std::vector<float> vh(static_cast<std::size_t>(adv_hidden_));
    applyLinear(v_fc1_, trunk.data(), vh.data());
    applyLN(v_ln1_, vh.data());
    for (int i = 0; i < adv_hidden_; ++i) vh[std::size_t(i)] = silu(vh[std::size_t(i)]);
    float v_val = 0.0f;
    applyLinear(v_fc2_, vh.data(), &v_val);

    // Advantage head over each legal candidate action
    std::vector<float> ae(static_cast<std::size_t>(act_hidden_));
    std::vector<float> ta(static_cast<std::size_t>(trunk_dim + act_hidden_));
    std::memcpy(ta.data(), trunk.data(), std::size_t(trunk_dim) * sizeof(float));
    std::vector<float> ah(static_cast<std::size_t>(adv_hidden_));
    std::vector<float> ar(static_cast<std::size_t>(adv_hidden_));
    float adv[kBgMaxLegal]{};
    float mean_adv = 0.0f;
    for (int k = 0; k < n_legal; ++k) {
        applyLinear(act_fc_, legal_actions + std::size_t(k * action_dim_), ae.data());
        applyLN(act_ln_, ae.data());
        for (int i = 0; i < act_hidden_; ++i) ta[std::size_t(trunk_dim + i)] = silu(ae[std::size_t(i)]);
        applyLinear(adv_fc1_, ta.data(), ah.data());
        applyLN(adv_ln1_, ah.data());
        for (int i = 0; i < adv_hidden_; ++i) ah[std::size_t(i)] = silu(ah[std::size_t(i)]);
        applyResBlock(adv_res_, ah.data(), ar.data(), scratch);
        applyLinear(adv_fc2_, ar.data(), &adv[k]);
        mean_adv += adv[k];
    }
    mean_adv /= float(std::max(1, n_legal));
    for (int k = 0; k < n_legal; ++k) q_out[k] = v_val + (adv[k] - mean_adv);
}

std::vector<std::pair<int, double>> BgQGRUModel::evalLegal(const PublicState& ps, const Facts* facts) const {
    float state[kBgStateDim];
    float history[kBgMaxPlies * kBgPlyDim];
    float actions[kBgMaxLegal * kBgActionDim];
    std::int32_t legal_cards[kBgMaxLegal];
    encodeState(ps, state, facts);
    const int n_plies = encodeHistory(ps, history);
    const int n_legal = encodeLegalActions(ps, actions, legal_cards);
    float q[kBgMaxLegal]{};
    forwardQ(state, history, n_plies, actions, n_legal, q);
    std::vector<std::pair<int, double>> out;
    out.reserve(std::size_t(n_legal));
    for (int k = 0; k < n_legal; ++k) out.push_back({legal_cards[k], double(q[k]) * 40.0});
    return out;
}

int BgQGRUModel::chooseCard(const PublicState& ps, const Facts* facts) const {
    const auto vals = evalLegal(ps, facts);
    int best = vals.empty() ? kNoCard : vals[0].first;
    double best_v = -1e30;
    for (const auto& [c, v] : vals) {
        if (v > best_v) {
            best_v = v;
            best = c;
        }
    }
    return best;
}

}  // namespace bg
