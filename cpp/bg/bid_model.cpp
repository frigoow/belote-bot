#include "bg/bid_model.h"
#include "bg/agent.h"
#include "bg/rule_policy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace bg {

namespace {

inline float dotProd(const float* a, const float* b, int n) {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    float sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
#else
    float sum = 0.0f;
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

bool readBytes(std::ifstream& in, void* dst, std::size_t n) {
    in.read(reinterpret_cast<char*>(dst), std::streamsize(n));
    return bool(in);
}

bool readLinear(std::ifstream& in, detail::Linear& l, int in_dim, int out_dim) {
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    l.w.resize(static_cast<std::size_t>(out_dim) * static_cast<std::size_t>(in_dim));
    l.b.resize(static_cast<std::size_t>(out_dim));
    return readBytes(in, l.w.data(), l.w.size() * sizeof(float)) &&
           readBytes(in, l.b.data(), l.b.size() * sizeof(float));
}

bool readLN(std::ifstream& in, detail::LayerNorm& ln, int dim) {
    ln.dim = dim;
    ln.w.resize(static_cast<std::size_t>(dim));
    ln.b.resize(static_cast<std::size_t>(dim));
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
    g.w_ih.resize(static_cast<std::size_t>(3 * hidden) * static_cast<std::size_t>(in_dim));
    g.w_hh.resize(static_cast<std::size_t>(3 * hidden) * static_cast<std::size_t>(hidden));
    g.b_ih.resize(static_cast<std::size_t>(3 * hidden));
    g.b_hh.resize(static_cast<std::size_t>(3 * hidden));
    g.empty_emb.resize(static_cast<std::size_t>(hidden));
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
        y[o] = b[o] + dotProd(w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in_d), x, in_d);
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
    for (int i = 0; i < d; ++i) x[i] = (x[i] - mean) * inv * ln.w[static_cast<std::size_t>(i)] + ln.b[static_cast<std::size_t>(i)];
}

void applyResBlock(const detail::ResBlock& rb, const float* x, float* out, std::vector<float>& scratch) {
    const int d = rb.fc1.out_dim;
    scratch.resize(static_cast<std::size_t>(d));
    applyLinear(rb.fc1, x, scratch.data());
    applyLN(rb.ln1, scratch.data());
    for (int i = 0; i < d; ++i) scratch[static_cast<std::size_t>(i)] = silu(scratch[static_cast<std::size_t>(i)]);
    applyLinear(rb.fc2, scratch.data(), out);
    applyLN(rb.ln2, out);
    for (int i = 0; i < d; ++i) out[i] = silu(x[i] + out[i]);
}

void applyGru(const detail::GruLayer& g, const float* seq, int n_steps, float* h_out) {
    const int H = g.hidden;
    const int I = g.in_dim;
    if (n_steps <= 0) {
        std::memcpy(h_out, g.empty_emb.data(), static_cast<std::size_t>(H) * sizeof(float));
        return;
    }
    std::memset(h_out, 0, static_cast<std::size_t>(H) * sizeof(float));
    thread_local std::vector<float> gi;
    thread_local std::vector<float> gh;
    gi.resize(static_cast<std::size_t>(3 * H));
    gh.resize(static_cast<std::size_t>(3 * H));
    for (int t = 0; t < n_steps; ++t) {
        const float* x = seq + static_cast<std::size_t>(t * I);
        for (int o = 0; o < 3 * H; ++o) {
            gi[static_cast<std::size_t>(o)] = g.b_ih[static_cast<std::size_t>(o)] +
                                              dotProd(g.w_ih.data() + static_cast<std::size_t>(o * I), x, I);
            gh[static_cast<std::size_t>(o)] = g.b_hh[static_cast<std::size_t>(o)] +
                                              dotProd(g.w_hh.data() + static_cast<std::size_t>(o * H), h_out, H);
        }
        for (int j = 0; j < H; ++j) {
            const float r = sigmoid(gi[static_cast<std::size_t>(j)] + gh[static_cast<std::size_t>(j)]);
            const float z = sigmoid(gi[static_cast<std::size_t>(H + j)] + gh[static_cast<std::size_t>(H + j)]);
            const float n = std::tanh(gi[static_cast<std::size_t>(2 * H + j)] + r * gh[static_cast<std::size_t>(2 * H + j)]);
            h_out[j] = (1.0f - z) * n + z * h_out[j];
        }
    }
}

int longestRunInSuit(int suit_bits8) {
    int best = 0, cur = 0;
    for (int r = 0; r < 8; ++r) {
        if (suit_bits8 & (1 << r)) {
            ++cur;
            if (cur > best) best = cur;
        } else {
            cur = 0;
        }
    }
    return best;
}

}  // namespace

BidPublicState bidPublicState(const Round& r, int sn, int ew) {
    BidPublicState bs;
    bs.first = r.first();
    bs.me = r.toMove();
    bs.hand5 = r.hand(bs.me);
    bs.avail = r.availableBids();
    bs.bids = r.bids();
    bs.sn = sn;
    bs.ew = ew;
    bs.hanging = r.hangingBefore();
    return bs;
}

std::uint16_t naturalBidMask(const BidPublicState& bs, bool allow_double) {
    std::uint16_t mask = 1u;  // Pass (index 0) is always allowed.
    const Mask h = bs.hand5;
    const int partner = partnerOf(bs.me);
    bool partner_bid = false;
    for (const auto& [seat, b] : bs.bids) {
        if (seat == partner && b != kPass) partner_bid = true;
    }

    int n_jacks = 0, n_nines = 0, n_aces = 0, n_tens = 0;
    for (int s = 0; s < 4; ++s) {
        const int sb = int((h >> (8 * s)) & 255u);
        const int cnt = popcount(Mask(sb));
        const bool has_j = (sb & (1 << belote::kJ)) != 0;
        const bool has_9 = (sb & (1 << belote::k9)) != 0;
        const bool has_a = (sb & (1 << belote::kA)) != 0;
        const bool has_10 = (sb & (1 << belote::k10)) != 0;
        const bool has_honor = (sb & ((1 << belote::kA) | (1 << belote::k10) | (1 << belote::kK) | (1 << belote::kQ))) != 0;
        if (has_j) ++n_jacks;
        if (has_9) ++n_nines;
        if (has_a) ++n_aces;
        if (has_10) ++n_tens;

        const std::uint8_t flag = std::uint8_t(1u << s);
        if (bs.avail & flag) {
            // Safe natural suit mask: blocks 0/1-card or trash relay bids while keeping real hands
            // like J+x, 9+x, A-10-K-Q, K-Q-A, or 4-card suits legal!
            if (((has_j || has_9) && cnt >= 2) || (cnt >= 3 && has_honor) || cnt >= 4) {
                mask |= std::uint16_t(1u << (1 + s));
            }
        }
    }

    if ((bs.avail & kNoTrumps) && (n_aces >= 1 || (partner_bid && n_tens >= 2))) {
        mask |= std::uint16_t(1u << 5);
    }
    if ((bs.avail & kAllTrumps) && (n_jacks >= 1 || (n_nines >= 2 && (partner_bid || (n_aces + n_tens) >= 1)))) {
        mask |= std::uint16_t(1u << 6);
    }
    if (allow_double) {
        if (bs.avail & kDouble) mask |= std::uint16_t(1u << 7);
        if (bs.avail & kReDouble) mask |= std::uint16_t(1u << 8);
    }
    return mask;
}

int encodeBidState(const BidPublicState& bs, bool allow_double, float* suit_feats, float* global_feats,
                   float* bid_hist, float* legal_mask) {
    std::memset(suit_feats, 0, std::size_t(4 * kBgBidSuitDim) * sizeof(float));
    std::memset(global_feats, 0, std::size_t(kBgBidGlobalDim) * sizeof(float));
    std::memset(bid_hist, 0, std::size_t(kBgMaxBids * kBgBidStepDim) * sizeof(float));

    const std::uint16_t nmask = naturalBidMask(bs, allow_double);
    for (int i = 0; i < kBgNumBids; ++i) legal_mask[i] = (nmask & (1u << i)) ? 1.0f : 0.0f;

    // Replay auction history to extract current contract, declarer, consecutive passes, and per-seat flags.
    std::uint8_t cur_contract = kPass;
    int cur_declarer = -1;
    int consec_passes = 0;
    bool partner_bid_suit[4]{};
    bool opp_bid_suit[4]{};
    float seat_summary[4][5]{};  // [rel_seat]: any_suit, nt, at, dbl/rdbl, has_passed

    const int n_bids = std::min(int(bs.bids.size()), kBgMaxBids);
    const std::size_t start_idx = bs.bids.size() > std::size_t(kBgMaxBids) ? bs.bids.size() - kBgMaxBids : 0;
    for (std::size_t idx = 0; idx < bs.bids.size(); ++idx) {
        const auto [seat, b] = bs.bids[idx];
        const int rs = relSeat(seat, bs.me);
        const int bi = bidToIndex(b);
        if (idx >= start_idx) {
            float* row = bid_hist + (idx - start_idx) * kBgBidStepDim;
            row[rs] = 1.0f;
            row[4 + bi] = 1.0f;
        }
        if (b == kPass) {
            ++consec_passes;
            seat_summary[rs][4] = 1.0f;
        } else {
            consec_passes = 0;
            if (b == kDouble || b == kReDouble) {
                cur_contract &= std::uint8_t(~(kDouble | kReDouble));
                cur_contract |= b;
                seat_summary[rs][3] = 1.0f;
            } else {
                cur_contract = b;
                cur_declarer = seat;
                if (b & 15u) {
                    const int s = trumpSuit(b);
                    seat_summary[rs][0] = 1.0f;
                    if (rs == 2) partner_bid_suit[s] = true;
                    else if (rs == 1 || rs == 3) opp_bid_suit[s] = true;
                } else if (b & kNoTrumps) {
                    seat_summary[rs][1] = 1.0f;
                } else if (b & kAllTrumps) {
                    seat_summary[rs][2] = 1.0f;
                }
            }
        }
    }

    // Per-suit features (4 x 16) — identical schema across all 4 suits so BgBidNet's shared suit tower
    // extracts trump/plain/AT/meld strength cleanly while explicit rank s/3 preserves C < D < H < S!
    const Mask h = bs.hand5;
    int n_jacks = 0, n_nines = 0, n_aces = 0, n_tens = 0;
    int n_voids = 0, n_singletons = 0, max_len = 0, best_seq = 0;
    int rank_counts[8]{};
    for (int s = 0; s < 4; ++s) {
        float* sf = suit_feats + s * kBgBidSuitDim;
        const int sb = int((h >> (8 * s)) & 255u);
        int cnt = 0;
        for (int r = 0; r < 8; ++r) {
            if (sb & (1 << r)) {
                sf[r] = 1.0f;
                ++cnt;
                ++rank_counts[r];
            }
        }
        const int run = longestRunInSuit(sb);
        if (run > best_seq) best_seq = run;
        if (cnt == 0) ++n_voids;
        if (cnt == 1) ++n_singletons;
        if (cnt > max_len) max_len = cnt;
        if (sb & (1 << belote::kJ)) ++n_jacks;
        if (sb & (1 << belote::k9)) ++n_nines;
        if (sb & (1 << belote::kA)) ++n_aces;
        if (sb & (1 << belote::k10)) ++n_tens;

        sf[8] = float(cnt) / 5.0f;
        sf[9] = ((sb & (1 << belote::kJ)) && (sb & (1 << belote::k9))) ? 1.0f : 0.0f;
        sf[10] = ((sb & (1 << belote::kK)) && (sb & (1 << belote::kQ))) ? 1.0f : 0.0f;
        sf[11] = run >= 3 ? 1.0f : 0.0f;
        sf[12] = float(s) / 3.0f;
        sf[13] = (nmask & (1u << (1 + s))) ? 1.0f : 0.0f;
        sf[14] = partner_bid_suit[s] ? 1.0f : 0.0f;
        sf[15] = opp_bid_suit[s] ? 1.0f : 0.0f;
    }

    // Global features (64)
    for (int i = 0; i < kBgNumBids; ++i) global_feats[i] = legal_mask[i];
    const std::uint8_t base_c = cur_contract & 63u;
    global_feats[9 + bidToIndex(base_c)] = 1.0f;  // 9..15: none, C, D, H, S, NT, AT
    global_feats[16] = (cur_contract & kDouble) ? 1.0f : 0.0f;
    global_feats[17] = (cur_contract & kReDouble) ? 1.0f : 0.0f;
    if (cur_declarer >= 0) {
        global_feats[18 + relSeat(cur_declarer, bs.me)] = 1.0f;
        global_feats[22] = (teamOf(cur_declarer) == teamOf(bs.me)) ? 1.0f : 0.0f;
    }
    global_feats[23 + relSeat(bs.first, bs.me)] = 1.0f;
    global_feats[27] = (teamOf(bs.first) == teamOf(bs.me)) ? 1.0f : 0.0f;
    global_feats[28 + std::min(consec_passes, 3)] = 1.0f;

    global_feats[32] = float(n_jacks) / 4.0f;
    global_feats[33] = float(n_nines) / 4.0f;
    global_feats[34] = float(n_aces) / 4.0f;
    global_feats[35] = float(n_tens) / 4.0f;
    global_feats[36] = rank_counts[belote::kJ] == 4 ? 1.0f : 0.0f;
    global_feats[37] = rank_counts[belote::k9] == 4 ? 1.0f : 0.0f;
    global_feats[38] = (rank_counts[belote::k10] == 4 || rank_counts[belote::kQ] == 4 ||
                        rank_counts[belote::kK] == 4 || rank_counts[belote::kA] == 4)
                           ? 1.0f
                           : 0.0f;
    global_feats[39] = float(best_seq) / 5.0f;
    global_feats[40] = float(n_voids) / 3.0f;
    global_feats[41] = float(n_singletons) / 4.0f;
    global_feats[42] = float(max_len) / 5.0f;
    global_feats[43] = float(std::min<int>(int(bs.bids.size()), 12)) / 12.0f;

    for (int r = 0; r < 4; ++r) {
        for (int k = 0; k < 5; ++k) global_feats[44 + r * 5 + k] = seat_summary[r][k];
    }
    return n_bids;
}

// ================================================================== BgBidModel

bool BgBidModel::load(const std::string& path) {
    loaded_ = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4]{};
    std::int32_t hdr[8]{};
    if (!readBytes(in, magic, 4) || std::memcmp(magic, "BGBD", 4) != 0) return false;
    if (!readBytes(in, hdr, sizeof(hdr))) return false;
    if (hdr[0] != kBgBidVersion || hdr[1] != kBgBidSuitDim || hdr[2] != kBgBidGlobalDim ||
        hdr[3] != kBgBidStepDim || hdr[7] != kBgNumBids) {
        return false;
    }
    suit_hidden_ = hdr[4];
    global_hidden_ = hdr[5];
    gru_hidden_ = hdr[6];
    trunk_dim_ = 4 * suit_hidden_ + global_hidden_ + gru_hidden_;

    if (!readLinear(in, suit_fc1_, kBgBidSuitDim, suit_hidden_) ||
        !readLN(in, suit_ln1_, suit_hidden_) ||
        !readLinear(in, suit_fc2_, suit_hidden_, suit_hidden_) ||
        !readLN(in, suit_ln2_, suit_hidden_) ||
        !readLinear(in, glob_fc_, kBgBidGlobalDim, global_hidden_) ||
        !readLN(in, glob_ln_, global_hidden_) ||
        !readGru(in, hist_gru_, kBgBidStepDim, gru_hidden_) ||
        !readResBlock(in, trunk1_, trunk_dim_) ||
        !readResBlock(in, trunk2_, trunk_dim_) ||
        !readLinear(in, head_, trunk_dim_, kBgNumBids)) {
        return false;
    }
    loaded_ = true;
    return true;
}

void BgBidModel::forwardQ(const float* suit_feats, const float* global_feats, const float* bid_hist, int n_bids,
                          float* q_out) const {
    thread_local std::vector<float> cat;
    thread_local std::vector<float> s_tmp;
    thread_local std::vector<float> h1;
    thread_local std::vector<float> h2;
    thread_local std::vector<float> scratch;
    cat.resize(static_cast<std::size_t>(trunk_dim_));
    s_tmp.resize(static_cast<std::size_t>(suit_hidden_));
    h1.resize(static_cast<std::size_t>(trunk_dim_));
    h2.resize(static_cast<std::size_t>(trunk_dim_));

    // Apply shared per-suit tower identically to each of the 4 suits.
    for (int s = 0; s < 4; ++s) {
        applyLinear(suit_fc1_, suit_feats + s * kBgBidSuitDim, s_tmp.data());
        applyLN(suit_ln1_, s_tmp.data());
        for (int i = 0; i < suit_hidden_; ++i) s_tmp[static_cast<std::size_t>(i)] = silu(s_tmp[static_cast<std::size_t>(i)]);
        float* dst = cat.data() + s * suit_hidden_;
        applyLinear(suit_fc2_, s_tmp.data(), dst);
        applyLN(suit_ln2_, dst);
        for (int i = 0; i < suit_hidden_; ++i) dst[i] = silu(dst[i]);
    }

    float* g_dst = cat.data() + 4 * suit_hidden_;
    applyLinear(glob_fc_, global_feats, g_dst);
    applyLN(glob_ln_, g_dst);
    for (int i = 0; i < global_hidden_; ++i) g_dst[i] = silu(g_dst[i]);

    applyGru(hist_gru_, bid_hist, n_bids, g_dst + global_hidden_);

    applyResBlock(trunk1_, cat.data(), h1.data(), scratch);
    applyResBlock(trunk2_, h1.data(), h2.data(), scratch);
    applyLinear(head_, h2.data(), q_out);
}

std::uint16_t BgBidModel::evalBids(const BidPublicState& bs, bool allow_double,
                                   std::array<double, kBgNumBids>& q_pts) const {
    float sf[4 * kBgBidSuitDim];
    float gf[kBgBidGlobalDim];
    float bh[kBgMaxBids * kBgBidStepDim];
    float lm[kBgNumBids];
    const int nb = encodeBidState(bs, allow_double, sf, gf, bh, lm);
    float q[kBgNumBids]{};
    forwardQ(sf, gf, bh, nb, q);
    for (int i = 0; i < kBgNumBids; ++i) q_pts[static_cast<std::size_t>(i)] = double(q[i]) * 30.0;
    return naturalBidMask(bs, allow_double);
}

std::uint8_t BgBidModel::chooseBid(const BidPublicState& bs, bool allow_double, double double_margin) const {
    std::array<double, kBgNumBids> q_pts{};
    const std::uint16_t mask = evalBids(bs, allow_double, q_pts);
    int best = 0;
    double best_v = q_pts[0];
    for (int i = 1; i < kBgNumBids; ++i) {
        if (!(mask & (1u << i))) continue;
        double v = q_pts[static_cast<std::size_t>(i)];
        if (i >= 7) v -= double_margin;
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return indexToBid(best);
}

std::uint8_t BgBidModel::chooseBid(const Round& r, bool allow_double, double double_margin) const {
    return chooseBid(bidPublicState(r), allow_double, double_margin);
}

// ================================================================== BgBatchBidEnv

BgBatchBidEnv::BgBatchBidEnv(const std::string& fast_at_path, const std::string& fast_suit_path,
                             const std::string& fast_nt_path, int threads)
    : threads_(std::max(1, threads)) {
    if (!fast_at_.load(fast_at_path)) throw std::runtime_error("BgBatchBidEnv: failed to load " + fast_at_path);
    if (!fast_suit_.load(fast_suit_path)) throw std::runtime_error("BgBatchBidEnv: failed to load " + fast_suit_path);
    if (!fast_nt_.load(fast_nt_path)) throw std::runtime_error("BgBatchBidEnv: failed to load " + fast_nt_path);
}

bool BgBatchBidEnv::loadBidModel(const std::string& path) { return bid_model_.load(path); }

int BgBatchBidEnv::playFastDeal(Round r) const {
    if (r.finished()) return 0;  // passed out
    const std::uint8_t c = r.contract();
    const BgFastPolicyModel& fp = (c & kAllTrumps) ? fast_at_ : ((c & kNoTrumps) ? fast_nt_ : fast_suit_);
    while (!r.finished()) {
        if (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
        } else {
            r.play(fp.chooseCard(r), true);
        }
    }
    return r.result().sn - r.result().ew;
}

std::uint8_t BgBatchBidEnv::pickPolicyBid(const Round& r, int seat_policy, std::mt19937_64& rng, double eps,
                                          bool allow_double) const {
    // seat_policy: 0 = BidModel (or naturalBid if unloaded), 1 = naturalBid(0), 2 = naturalBidTuned(+10 AT)
    if (seat_policy == 1 || !bid_model_.loaded()) {
        return naturalBidTuned(r, 0, 0, allow_double);
    }
    if (seat_policy == 2) {
        return naturalBidTuned(r, 10, 0, allow_double);
    }
    const BidPublicState bs = bidPublicState(r);
    if (eps > 0.0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < eps) {
        const std::uint16_t m = naturalBidMask(bs, allow_double);
        int opts[kBgNumBids], n = 0;
        for (int i = 0; i < kBgNumBids; ++i)
            if (m & (1u << i)) opts[n++] = i;
        return indexToBid(opts[rng() % static_cast<std::uint64_t>(n)]);
    }
    return bid_model_.chooseBid(bs, allow_double);
}

CounterfactualBidBatch BgBatchBidEnv::generateBatch(int num_deals, std::uint64_t seed, double explore_eps,
                                                    bool allow_double) const {
    struct LocalBuf {
        int n = 0;
        std::vector<float> sf, gf, bh, lm, tgt;
        std::vector<std::int32_t> nb;
    };

    const int nt = std::min(threads_, std::max(1, num_deals));
    std::vector<LocalBuf> bufs(static_cast<std::size_t>(nt));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nt));

    auto worker = [&](int tid, int begin, int end) {
        LocalBuf& out = bufs[static_cast<std::size_t>(tid)];
        const int est = (end - begin) * 3;
        out.sf.reserve(static_cast<std::size_t>(est * 4 * kBgBidSuitDim));
        out.gf.reserve(static_cast<std::size_t>(est * kBgBidGlobalDim));
        out.bh.reserve(static_cast<std::size_t>(est * kBgMaxBids * kBgBidStepDim));
        out.nb.reserve(static_cast<std::size_t>(est));
        out.lm.reserve(static_cast<std::size_t>(est * kBgNumBids));
        out.tgt.reserve(static_cast<std::size_t>(est * kBgNumBids));

        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);

        for (int g = begin; g < end; ++g) {
            std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(g + 1) * 0xBF58476D1CE4E5B9ull);
            std::shuffle(deck.begin(), deck.end(), rng);
            const int first = g % kPlayers;

            // Mixed-table distribution (table_type independent of first = g % 4 so all 4 seats are balanced):
            //   50%: BidNet x4 (self-play)
            //   25%: Mixed partner (one BidNet + one naturalBid on each team, rotating which seat)
            //   25%: BidNet team vs aggressive-AT naturalBidTuned(+10) opponents (Master-like table!)
            int seat_pol[4] = {0, 0, 0, 0};
            const int table_type = (g >> 2) & 3;
            if (table_type == 2) {
                const int flip = (g >> 4) & 1;
                seat_pol[0] = flip ? 1 : 0; seat_pol[2] = flip ? 0 : 1;
                seat_pol[1] = flip ? 1 : 0; seat_pol[3] = flip ? 0 : 1;
            } else if (table_type == 3) {
                const int our_team = (g >> 4) & 1;
                for (int p = 0; p < 4; ++p) seat_pol[p] = (teamOf(p) == our_team) ? 0 : 2;
            }

            Round r(deck, first);
            while (r.decision() == Round::Decision::kBid) {
                const int me = r.toMove();
                const int my_team = teamOf(me);
                const BidPublicState bs = bidPublicState(r);
                const std::uint16_t nmask = naturalBidMask(bs, allow_double);

                // Only generate counterfactual training nodes when it's a BidNet seat with >= 2 legal natural bids!
                int n_opts = 0;
                for (int i = 0; i < kBgNumBids; ++i)
                    if (nmask & (1u << i)) ++n_opts;

                if (seat_pol[me] == 0 && n_opts >= 2) {
                    float sf[4 * kBgBidSuitDim], gf[kBgBidGlobalDim], bh[kBgMaxBids * kBgBidStepDim], lm[kBgNumBids];
                    const int nb = encodeBidState(bs, allow_double, sf, gf, bh, lm);
                    float tgt[kBgNumBids]{};

                    // Branch over EVERY legal bid on this exact 32-card deal!
                    for (int bi = 0; bi < kBgNumBids; ++bi) {
                        if (!(nmask & (1u << bi))) continue;
                        Round rb = r;
                        rb.bid(indexToBid(bi));
                        while (rb.decision() == Round::Decision::kBid) {
                            const int s2 = rb.toMove();
                            rb.bid(pickPolicyBid(rb, seat_pol[s2], rng, 0.0, allow_double));
                        }
                        const int m0 = playFastDeal(std::move(rb));
                        const int my_m = my_team == 0 ? m0 : -m0;
                        tgt[bi] = float(my_m) / 30.0f;
                    }

                    out.sf.insert(out.sf.end(), sf, sf + 4 * kBgBidSuitDim);
                    out.gf.insert(out.gf.end(), gf, gf + kBgBidGlobalDim);
                    out.bh.insert(out.bh.end(), bh, bh + kBgMaxBids * kBgBidStepDim);
                    out.nb.push_back(nb);
                    out.lm.insert(out.lm.end(), lm, lm + kBgNumBids);
                    out.tgt.insert(out.tgt.end(), tgt, tgt + kBgNumBids);
                    ++out.n;
                }

                // Advance actual auction trajectory with mild exploration.
                const std::uint8_t act = pickPolicyBid(r, seat_pol[me], rng, explore_eps, allow_double);
                r.bid(act);
            }
        }
    };

    const int chunk = (num_deals + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int b = t * chunk;
        const int e = std::min(num_deals, b + chunk);
        if (b < e) pool.emplace_back(worker, t, b, e);
    }
    for (auto& th : pool) th.join();

    CounterfactualBidBatch res;
    for (const auto& lb : bufs) res.n_samples += lb.n;
    res.suit_feats.reserve(static_cast<std::size_t>(res.n_samples * 4 * kBgBidSuitDim));
    res.global_feats.reserve(static_cast<std::size_t>(res.n_samples * kBgBidGlobalDim));
    res.bid_hist.reserve(static_cast<std::size_t>(res.n_samples * kBgMaxBids * kBgBidStepDim));
    res.n_bids.reserve(static_cast<std::size_t>(res.n_samples));
    res.legal_mask.reserve(static_cast<std::size_t>(res.n_samples * kBgNumBids));
    res.targets.reserve(static_cast<std::size_t>(res.n_samples * kBgNumBids));
    for (const auto& lb : bufs) {
        res.suit_feats.insert(res.suit_feats.end(), lb.sf.begin(), lb.sf.end());
        res.global_feats.insert(res.global_feats.end(), lb.gf.begin(), lb.gf.end());
        res.bid_hist.insert(res.bid_hist.end(), lb.bh.begin(), lb.bh.end());
        res.n_bids.insert(res.n_bids.end(), lb.nb.begin(), lb.nb.end());
        res.legal_mask.insert(res.legal_mask.end(), lb.lm.begin(), lb.lm.end());
        res.targets.insert(res.targets.end(), lb.tgt.begin(), lb.tgt.end());
    }
    return res;
}

BgBatchBidEnv::DuelSummary BgBatchBidEnv::evalVsNatural(int num_deals, std::uint64_t seed, int opp_at_bonus,
                                                        bool mixed_partner, bool allow_double) const {
    std::vector<double> pair_scores(static_cast<std::size_t>(num_deals), 0.0);
    std::mutex mu;
    long c_suit = 0, c_nt = 0, c_at = 0, c_pass = 0;
    long suit_total = 0, suit_j9 = 0, nt_total = 0, nt_ace = 0, at_total = 0, at_j = 0;

    const int nt = std::min(threads_, std::max(1, num_deals));
    std::vector<std::thread> pool;
    auto worker = [&](int begin, int end) {
        long ls = 0, lnt = 0, lat = 0, lp = 0;
        long stot = 0, sj9 = 0, ntot = 0, na = 0, atot = 0, aj = 0;
        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);
        for (int i = begin; i < end; ++i) {
            std::mt19937_64 rng(seed * 1000003ull + std::uint64_t(i));
            std::shuffle(deck.begin(), deck.end(), rng);
            const int first = i % kPlayers;
            double pair = 0.0;
            for (int side = 0; side < 2; ++side) {
                Round r(deck, first);
                while (r.decision() == Round::Decision::kBid) {
                    const int s = r.toMove();
                    const bool our_team = (teamOf(s) == side);
                    const bool is_ours = our_team && (!mixed_partner || s == side);
                    std::uint8_t b;
                    if (is_ours && bid_model_.loaded()) {
                        b = bid_model_.chooseBid(r, allow_double);
                        // Naturalness audit on our non-pass bids!
                        const Mask h = r.hand(s);
                        if (b & 15u) {
                            ++stot;
                            const int su = trumpSuit(b);
                            if (h & (bit(cardOf(su, belote::kJ)) | bit(cardOf(su, belote::k9)))) ++sj9;
                        } else if (b & kNoTrumps) {
                            ++ntot;
                            bool has_a = false;
                            for (int su = 0; su < 4; ++su) has_a |= (h & bit(cardOf(su, belote::kA))) != 0;
                            if (has_a) ++na;
                        } else if (b & kAllTrumps) {
                            ++atot;
                            bool has_j = false;
                            for (int su = 0; su < 4; ++su) has_j |= (h & bit(cardOf(su, belote::kJ))) != 0;
                            if (has_j) ++aj;
                        }
                    } else {
                        b = naturalBidTuned(r, opp_at_bonus, 0, allow_double);
                    }
                    r.bid(b);
                }
                if (side == 0) {
                    if (r.finished()) ++lp;
                    else if (r.contract() & kAllTrumps) ++lat;
                    else if (r.contract() & kNoTrumps) ++lnt;
                    else ++ls;
                }
                const int m0 = playFastDeal(std::move(r));
                pair += 0.5 * double(side == 0 ? m0 : -m0);
            }
            pair_scores[static_cast<std::size_t>(i)] = pair;
        }
        std::lock_guard<std::mutex> lk(mu);
        c_suit += ls; c_nt += lnt; c_at += lat; c_pass += lp;
        suit_total += stot; suit_j9 += sj9; nt_total += ntot; nt_ace += na; at_total += atot; at_j += aj;
    };
    const int chunk = (num_deals + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int b = t * chunk;
        const int e = std::min(num_deals, b + chunk);
        if (b < e) pool.emplace_back(worker, b, e);
    }
    for (auto& th : pool) th.join();

    DuelSummary out;
    out.deals = num_deals;
    double mean = 0.0, var = 0.0;
    for (double x : pair_scores) mean += x / double(num_deals);
    if (num_deals > 1) {
        for (double x : pair_scores) var += (x - mean) * (x - mean) / double(num_deals - 1);
    }
    out.mean = mean;
    out.ci = num_deals > 1 ? 1.96 * std::sqrt(var / double(num_deals)) : 0.0;
    const double tot = double(std::max<long>(1, c_suit + c_nt + c_at + c_pass));
    out.pct_suit = 100.0 * double(c_suit) / tot;
    out.pct_nt = 100.0 * double(c_nt) / tot;
    out.pct_at = 100.0 * double(c_at) / tot;
    out.pct_pass = 100.0 * double(c_pass) / tot;
    out.suit_has_j9 = suit_total > 0 ? 100.0 * double(suit_j9) / double(suit_total) : 100.0;
    out.nt_has_ace = nt_total > 0 ? 100.0 * double(nt_ace) / double(nt_total) : 100.0;
    out.at_has_j = at_total > 0 ? 100.0 * double(at_j) / double(at_total) : 100.0;
    return out;
}

}  // namespace bg
