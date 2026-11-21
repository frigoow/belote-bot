#include "search/bid_model.h"

#include "agents/agent.h"
#include "belote/match.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace belote {
namespace {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline double clampProb(double p) { return std::clamp(p, 1e-4, 1.0 - 1e-4); }

inline double logit(double p) {
    p = clampProb(p);
    return std::log(p / (1.0 - p));
}

inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

void layerNormSiLU(const float* in, const float* gamma, const float* beta, int n, float* out) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += double(in[i]);
    mean /= double(n);
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = double(in[i]) - mean;
        var += d * d;
    }
    var /= double(n);
    const float inv_std = float(1.0 / std::sqrt(var + 1e-5));
    for (int i = 0; i < n; ++i) {
        const float norm = (in[i] - float(mean)) * inv_std;
        out[i] = silu(norm * gamma[i] + beta[i]);
    }
}

void linearForward(const float* x, const std::vector<float>& w, const std::vector<float>& b, int in_dim, int out_dim,
                   float* out) {
    for (int o = 0; o < out_dim; ++o) {
        const float* row = w.data() + std::size_t(o) * std::size_t(in_dim);
        float acc = b[std::size_t(o)];
        for (int i = 0; i < in_dim; ++i) acc += row[i] * x[i];
        out[o] = acc;
    }
}

std::mt19937_64& bidSearchRng() {
    static thread_local std::mt19937_64 rng(0xB1D5EABC12345678ull);
    return rng;
}

}  // namespace

void BidModel::Mlp3::forward(const float* x, float* out) const {
    float buf1[512];
    float act1[512];
    float buf2[512];
    float act2[512];
    linearForward(x, w1, b1, in_dim, h1, buf1);
    layerNormSiLU(buf1, ln1_w.data(), ln1_b.data(), h1, act1);
    linearForward(act1, w2, b2, h1, h2, buf2);
    layerNormSiLU(buf2, ln2_w.data(), ln2_b.data(), h2, act2);
    linearForward(act2, w3, b3, h2, out_dim, out);
}

void encodeAuctionTake(Mask hand_5, int blik, int candidate_trump, int node_k, float* out) {
    std::memset(out, 0, std::size_t(kAuctionFeatDim) * sizeof(float));
    const int blik_suit = suitOf(blik);
    const int blik_rank = rankOf(blik);
    const int is_r2 = (node_k >= 4) ? 1 : 0;

    std::array<int, 4> suit_bytes{};
    for (int s = 0; s < 4; ++s) suit_bytes[std::size_t(s)] = int((hand_5 >> (s * 8)) & 0xFFu);

    std::array<int, 4> order{};
    if (is_r2) {
        int others[2];
        int k = 0;
        for (int s = 0; s < 4; ++s) {
            if (s != candidate_trump && s != blik_suit) others[k++] = s;
        }
        if (suit_bytes[std::size_t(others[1])] > suit_bytes[std::size_t(others[0])]) std::swap(others[0], others[1]);
        order = {candidate_trump, blik_suit, others[0], others[1]};
    } else {
        int others[3];
        int k = 0;
        for (int s = 0; s < 4; ++s) {
            if (s != candidate_trump) others[k++] = s;
        }
        // Stable descending sort by suit_bytes (matching Python's Timsort on [0..3]).
        std::stable_sort(others, others + 3, [&](int a, int b) {
            return suit_bytes[std::size_t(a)] > suit_bytes[std::size_t(b)];
        });
        order = {candidate_trump, others[0], others[1], others[2]};
    }

    int idx = 0;
    for (int slot = 0; slot < 4; ++slot) {
        const int byte = suit_bytes[std::size_t(order[std::size_t(slot)])];
        for (int r = 0; r < 8; ++r) {
            if (byte & (1 << r)) out[idx + slot * 8 + r] = 1.0f;
        }
    }
    idx += 32;

    out[idx + blik_rank] = 1.0f;
    idx += 8;
    out[idx++] = float(is_r2);
    if (node_k >= 0 && node_k < 8) out[idx + node_k] = 1.0f;
    idx += 8;

    const Mask known6 = hand_5 | bit(blik);
    const Mask trumps = (known6 >> (candidate_trump * 8)) & 0xFFu;
    const int n_trumps = popcount(trumps);
    const float has_j = (trumps & (1u << kJ)) ? 1.0f : 0.0f;
    const float has_9 = (trumps & (1u << k9)) ? 1.0f : 0.0f;
    const float has_a = (trumps & (1u << kA)) ? 1.0f : 0.0f;
    const float has_10 = (trumps & (1u << k10)) ? 1.0f : 0.0f;
    const float has_belot = ((trumps & ((1u << kQ) | (1u << kK))) == ((1u << kQ) | (1u << kK))) ? 1.0f : 0.0f;

    int n_side_aces = 0;
    int n_side_voids = 0;
    for (int s = 0; s < 4; ++s) {
        if (s == candidate_trump) continue;
        const Mask sm = (known6 >> (s * 8)) & 0xFFu;
        if (sm & (1u << kA)) ++n_side_aces;
        if (sm == 0) ++n_side_voids;
    }
    const float rule_val = float(ruleHandValue(known6, candidate_trump)) / 10.0f;
    const float is_dealer_partner = ((node_k % 4) == 1) ? 1.0f : 0.0f;

    out[idx + 0] = float(n_trumps) / 6.0f;
    out[idx + 1] = has_j;
    out[idx + 2] = has_9;
    out[idx + 3] = has_a;
    out[idx + 4] = has_10;
    out[idx + 5] = has_belot;
    out[idx + 6] = float(n_side_aces) / 3.0f;
    out[idx + 7] = float(n_side_voids) / 3.0f;
    out[idx + 8] = rule_val;
    out[idx + 9] = is_dealer_partner;
}

BidModel BidModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("BidModel: cannot open " + path);
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "BBM1", 4) != 0) throw std::runtime_error("BidModel: bad magic in " + path);

    std::int32_t hdr[8];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) throw std::runtime_error("BidModel: truncated header in " + path);

    BidModel m;
    m.feat_dim_ = hdr[0];
    m.ctx_dim_ = hdr[1];
    const int don_h = hdr[2];
    const int n_buckets = hdr[3];
    const int bn_h = hdr[4];
    m.target_ = hdr[5];
    m.margin_ = hdr[6];
    m.pot_cap_ = hdr[7];

    if (m.feat_dim_ != kAuctionFeatDim || m.ctx_dim_ != kMatchCtxDim || don_h > 512 || bn_h > 512 || n_buckets <= 0 ||
        m.target_ <= 0) {
        throw std::runtime_error("BidModel: unsupported dimensions in " + path);
    }

    m.buckets_.resize(std::size_t(n_buckets));
    for (int i = 0; i < n_buckets; ++i) {
        std::int32_t b[4];
        f.read(reinterpret_cast<char*>(b), sizeof(b));
        m.buckets_[std::size_t(i)] = OutcomeBucket{b[0], b[1], b[2], b[3]};
    }

    const int eff_m = std::max(1, m.margin_);
    const int n_diff = 2 * eff_m - 1;
    const std::size_t v_size =
        std::size_t(m.target_) * std::size_t(m.target_) * std::size_t(m.pot_cap_ + 1) * std::size_t(2);
    const std::size_t vt_size = std::size_t(n_diff) * std::size_t(m.pot_cap_ + 1) * std::size_t(2);
    m.v_table_.resize(v_size);
    m.v_tie_.resize(vt_size);
    f.read(reinterpret_cast<char*>(m.v_table_.data()), std::streamsize(v_size * sizeof(float)));
    f.read(reinterpret_cast<char*>(m.v_tie_.data()), std::streamsize(vt_size * sizeof(float)));

    auto readVec = [&](std::vector<float>& dst, std::size_t n) {
        dst.resize(n);
        f.read(reinterpret_cast<char*>(dst.data()), std::streamsize(n * sizeof(float)));
    };
    auto readMlp = [&](Mlp3& net, int in_d, int h, int out_d) {
        net.in_dim = in_d;
        net.h1 = h;
        net.h2 = h;
        net.out_dim = out_d;
        readVec(net.w1, std::size_t(h) * std::size_t(in_d));
        readVec(net.b1, std::size_t(h));
        readVec(net.ln1_w, std::size_t(h));
        readVec(net.ln1_b, std::size_t(h));
        readVec(net.w2, std::size_t(h) * std::size_t(h));
        readVec(net.b2, std::size_t(h));
        readVec(net.ln2_w, std::size_t(h));
        readVec(net.ln2_b, std::size_t(h));
        readVec(net.w3, std::size_t(out_d) * std::size_t(h));
        readVec(net.b3, std::size_t(out_d));
    };

    readMlp(m.don_, m.feat_dim_, don_h, n_buckets);
    readMlp(m.bidnet_, m.feat_dim_ + m.ctx_dim_, bn_h, 2);
    if (!f) throw std::runtime_error("BidModel: truncated weights in " + path);
    return m;
}

double BidModel::lookupV(int us, int them, int pot, int we_deal) const {
    const int m = std::max(1, margin_);
    const int p_idx = std::clamp(pot, 0, pot_cap_);
    if (us >= target_ || them >= target_) {
        const int diff = us - them;
        if (diff >= m) return 1.0;
        if (diff <= -m) return 0.0;
        const std::size_t di = std::size_t(diff + (m - 1));
        const std::size_t idx = (di * std::size_t(pot_cap_ + 1) + std::size_t(p_idx)) * 2u + std::size_t(we_deal & 1);
        return double(v_tie_[idx]);
    }
    const int u_idx = std::clamp(us, 0, target_ - 1);
    const int t_idx = std::clamp(them, 0, target_ - 1);
    const std::size_t stride_p = 2u;
    const std::size_t stride_t = std::size_t(pot_cap_ + 1) * stride_p;
    const std::size_t stride_u = std::size_t(target_) * stride_t;
    const std::size_t idx =
        std::size_t(u_idx) * stride_u + std::size_t(t_idx) * stride_t + std::size_t(p_idx) * stride_p +
        std::size_t(we_deal & 1);
    return double(v_table_[idx]);
}

double BidModel::evalDealEquity(const DealResult& res, const Config& cfg, int dealer, int my_team, int us, int them,
                                int pot) const {
    const int we_deal = (teamOf(dealer) == my_team) ? 1 : 0;
    const int next_we_deal = 1 - we_deal;
    if (res.outcome == Outcome::kAllPassed || res.outcome == Outcome::kNone) {
        return lookupV(us, them, pot, next_we_deal);
    }
    const RecordedDeal rec = recordDeal(res, cfg);
    if (res.outcome == Outcome::kHanging) {
        const int nu = us + rec.scores[my_team];
        const int nt = them + rec.scores[1 - my_team];
        const int npot = std::min(pot_cap_, pot + rec.held);
        return lookupV(nu, nt, npot, next_we_deal);
    }
    const int winner = (res.outcome == Outcome::kMade) ? res.takers : (1 - res.takers);
    const int nu = us + rec.scores[my_team] + (winner == my_team ? pot : 0);
    const int nt = them + rec.scores[1 - my_team] + (winner != my_team ? pot : 0);
    return lookupV(nu, nt, 0, next_we_deal);
}

void BidModel::predictOutcomeProbs(Mask hand_5, int blik, int candidate_trump, int node_k, float* probs_out) const {
    float feat[kAuctionFeatDim];
    encodeAuctionTake(hand_5, blik, candidate_trump, node_k, feat);
    don_.forward(feat, probs_out);
    const int k = int(buckets_.size());
    float max_l = probs_out[0];
    for (int i = 1; i < k; ++i) max_l = std::max(max_l, probs_out[i]);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        probs_out[i] = std::exp(probs_out[i] - max_l);
        sum += double(probs_out[i]);
    }
    const float inv = float(1.0 / sum);
    for (int i = 0; i < k; ++i) probs_out[i] *= inv;
}

double BidModel::expectedVTake(Mask hand_5, int blik, int candidate_trump, int node_k, int us, int them, int pot,
                               int we_deal) const {
    std::vector<float> probs(buckets_.size());
    predictOutcomeProbs(hand_5, blik, candidate_trump, node_k, probs.data());
    double val = 0.0;
    const int next_d = 1 - (we_deal & 1);
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        const float pr = probs[i];
        if (pr <= 0.0f) continue;
        const OutcomeBucket& b = buckets_[i];
        if (b.held > 0) {
            val += double(pr) * lookupV(us + b.du_taker, them + b.dt_def, std::min(pot_cap_, pot + b.held), next_d);
        } else {
            const int nu = us + b.du_taker + (b.taker_won ? pot : 0);
            const int nt = them + b.dt_def + (b.taker_won ? 0 : pot);
            val += double(pr) * lookupV(nu, nt, 0, next_d);
        }
    }
    return val;
}

void BidModel::evalContext(int us, int them, int pot, int we_deal, double ev_take, double best_r2_ev,
                           float* out) const {
    const double v_now = lookupV(us, them, pot, we_deal);
    const double v_redeal = lookupV(us, them, pot, 1 - we_deal);
    const double dv_us = lookupV(us + 8, them, 0, 1 - we_deal) - v_now;
    const double dv_them = lookupV(us, them + 8, 0, 1 - we_deal) - v_now;
    const double v_made = lookupV(us + 9 + pot, them + 7, 0, 1 - we_deal);
    const double v_nat = lookupV(us, them + 16 + pot, 0, 1 - we_deal);
    out[0] = float(us) / float(target_);
    out[1] = float(them) / float(target_);
    out[2] = float(pot) / float(std::max(1, pot_cap_));
    out[3] = float(we_deal);
    out[4] = float(v_redeal);
    out[5] = float(ev_take);
    out[6] = float(dv_us);
    out[7] = float(dv_them);
    out[8] = float(v_made - v_nat);
    out[9] = float(best_r2_ev);
}

void BidModel::evalBidNetPair(const float* af, const float* ctx, int node_k, double ev_take, double v_redeal,
                              double& v_pass_out, double& v_take_out) const {
    float inp[kAuctionFeatDim + kMatchCtxDim];
    std::memcpy(inp, af, std::size_t(kAuctionFeatDim) * sizeof(float));
    std::memcpy(inp + kAuctionFeatDim, ctx, std::size_t(kMatchCtxDim) * sizeof(float));
    float deltas[2];
    bidnet_.forward(inp, deltas);

    // Exact residual anchoring matching python/belote/bidding.py::BidNet.forward:
    // At node_k == 7 (last bidder), Pass causes immediate redeal so V(Pass) = v_redeal exactly.
    if (node_k == 7) {
        v_pass_out = v_redeal;
    } else {
        v_pass_out = sigmoid(logit(v_redeal) + double(deltas[0]));
    }
    v_take_out = sigmoid(logit(ev_take) + double(deltas[1]));
}

BidEval BidModel::evalBid(const Deal& d, int us, int them, int pot) const {
    const int me = d.currentPlayer();
    const int we_deal = (teamOf(d.dealer()) == teamOf(me)) ? 1 : 0;
    const Mask hand_5 = d.hand(me);
    const int blik = d.blik();
    const int blik_suit = suitOf(blik);
    const int node_k = std::max(0, d.bidsMade());
    const double v_redeal = lookupV(us, them, pot, 1 - we_deal);

    // Best Round-2 backup suit EV when deciding in Round 1 (node_k < 4).
    double best_r2_ev = v_redeal;
    if (node_k < 4) {
        best_r2_ev = -1.0;
        for (int s = 0; s < 4; ++s) {
            if (!d.config().round2_allow_blik_suit && s == blik_suit) continue;
            const double ev_s = expectedVTake(hand_5, blik, s, node_k + 4, us, them, pot, we_deal);
            best_r2_ev = std::max(best_r2_ev, ev_s);
        }
    }

    BidEval out;
    out.v_redeal = v_redeal;
    out.best_r2_ev = best_r2_ev;
    out.v_take = -1.0;

    const auto legal = d.legalBids();
    for (const Bid& b : legal) {
        if (b.kind == Bid::kPass) continue;
        const int cand_trump = (b.kind == Bid::kTake) ? blik_suit : b.suit;
        const double ev_t = expectedVTake(hand_5, blik, cand_trump, node_k, us, them, pot, we_deal);
        float af[kAuctionFeatDim];
        float ctx[kMatchCtxDim];
        encodeAuctionTake(hand_5, blik, cand_trump, node_k, af);
        evalContext(us, them, pot, we_deal, ev_t, best_r2_ev, ctx);
        double vp = 0.0, vt = 0.0;
        evalBidNetPair(af, ctx, node_k, ev_t, v_redeal, vp, vt);
        if (vt > out.v_take) {
            out.v_take = vt;
            out.v_pass = vp;
            out.ev_take = ev_t;
            out.best_take = b;
        }
    }
    out.best_bid = (out.v_take > out.v_pass) ? out.best_take : Bid::pass();
    return out;
}

Bid BidModel::chooseBid(const Deal& d, int us, int them, int pot) const {
    return evalBid(d, us, them, pot).best_bid;
}

Bid BidModel::chooseBidSearch(const Deal& d, const FastPolicyModel& play_policy, int us, int them, int pot, int worlds,
                              double search_margin, std::uint64_t seed) const {
    const BidEval fast_ev = evalBid(d, us, them, pot);
    if (worlds <= 0 || std::abs(fast_ev.v_take - fast_ev.v_pass) >= search_margin) return fast_ev.best_bid;

    // Stage 3 test-time auction search (PLAN.md §4.2):
    // Sample `worlds` ordered completions of the 26 unseen cards and complete every
    // deal through Deal::bid() (deck order matters because completion cards go
    // clockwise from dealer+1 and the taker receives 2 instead of 3).
    const int me = d.currentPlayer();
    const int my_team = teamOf(me);
    const int dealer = d.dealer();
    const int blik = d.blik();
    const Mask my_hand = d.hand(me);
    const int node_k = std::max(0, d.bidsMade());
    const auto legal = d.legalBids();

    // Identify the 26 unseen cards (32 - my 5 cards - face-up blik).
    std::array<int, 26> unseen{};
    int u_cnt = 0;
    for (int c = 0; c < kCards; ++c) {
        if (c == blik || (my_hand & bit(c))) continue;
        unseen[std::size_t(u_cnt++)] = c;
    }

    // Indices in Deal's 32-card deck corresponding to my 5 initial cards:
    // Deal deals 3 cards clockwise from dealer+1 (indices 0..11), then 2 cards (indices 12..19),
    // then deck[20] = blik, then deck[21..31] = stock completion.
    const int rel_me = (me - (dealer + 1) + 4) & 3;
    std::array<int, 5> my_cards{};
    {
        int k = 0;
        for (Mask m = my_hand; m; m &= m - 1) my_cards[std::size_t(k++)] = lowestCard(m);
    }

    // Evaluate all legal bids.
    std::vector<Bid> cands;
    cands.push_back(Bid::pass());
    if (node_k < 4) {
        cands.push_back(Bid::take());
    } else {
        for (const Bid& b : legal) {
            if (b.kind != Bid::kPass) cands.push_back(b);
        }
    }

    std::vector<double> logw(static_cast<std::size_t>(worlds), 0.0);
    std::vector<std::vector<double>> vals(cands.size(), std::vector<double>(static_cast<std::size_t>(worlds), 0.0));
    std::mt19937_64 local_rng(seed);
    std::mt19937_64& rng = seed ? local_rng : bidSearchRng();

    for (int w = 0; w < worlds; ++w) {
        std::shuffle(unseen.begin(), unseen.end(), rng);

        // Build full 32-card ordered deck placing my 5 cards into my seat's 3+2 slots and blik at 20.
        std::array<int, kCards> deck{};
        int ui = 0;
        for (int p_off = 0; p_off < 4; ++p_off) {
            for (int j = 0; j < 3; ++j) {
                deck[std::size_t(p_off * 3 + j)] = (p_off == rel_me) ? my_cards[std::size_t(j)] : unseen[std::size_t(ui++)];
            }
        }
        for (int p_off = 0; p_off < 4; ++p_off) {
            for (int j = 0; j < 2; ++j) {
                deck[std::size_t(12 + p_off * 2 + j)] =
                    (p_off == rel_me) ? my_cards[std::size_t(3 + j)] : unseen[std::size_t(ui++)];
            }
        }
        deck[20] = blik;
        for (int j = 21; j < kCards; ++j) deck[std::size_t(j)] = unseen[std::size_t(ui++)];

        Deal base(d.config(), dealer, deck);
        // Weight this sampled world by prior passers' auction likelihood.
        double lw = 0.0;
        for (int k = 0; k < node_k; ++k) {
            const int sp = base.currentPlayer();
            if (sp != me) {
                const int sp_team = teamOf(sp);
                const int sp_us = (sp_team == my_team) ? us : them;
                const int sp_them = (sp_team == my_team) ? them : us;
                const int sp_we_deal = (teamOf(dealer) == sp_team) ? 1 : 0;
                lw += bidLogProb(base.hand(sp), blik, k, d.config(), Bid::pass(), sp_us, sp_them, pot, sp_we_deal, 0.04);
            }
            base.bid(Bid::pass());
        }
        logw[std::size_t(w)] = lw;

        for (std::size_t ci = 0; ci < cands.size(); ++ci) {
            Deal sim = base;
            sim.bid(cands[ci]);
            while (sim.phase() == Phase::kBid1 || sim.phase() == Phase::kBid2) {
                const int sp = sim.currentPlayer();
                const int sp_team = teamOf(sp);
                const int sp_us = (sp_team == my_team) ? us : them;
                const int sp_them = (sp_team == my_team) ? them : us;
                sim.bid(chooseBid(sim, sp_us, sp_them, pot));
            }
            while (sim.phase() == Phase::kPlay) {
                const int sp = sim.currentPlayer();
                if (sim.canSetDeclare(sp)) sim.setDeclare(sp, smartChooseDeclare(sim));
                sim.play(play_policy.chooseCard(sim));
            }
            vals[ci][std::size_t(w)] = evalDealEquity(sim.result(), d.config(), dealer, my_team, us, them, pot);
        }
    }

    double max_lw = -std::numeric_limits<double>::infinity();
    for (double lw : logw) max_lw = std::max(max_lw, lw);
    std::vector<double> wt(static_cast<std::size_t>(worlds));
    double sum_w = 0.0;
    for (int w = 0; w < worlds; ++w) {
        wt[std::size_t(w)] = std::exp(std::max(-20.0, logw[std::size_t(w)] - max_lw));
        sum_w += wt[std::size_t(w)];
    }
    for (double& x : wt) x /= sum_w;

    std::size_t best_ci = 0;
    double best_val = -1.0;
    for (std::size_t ci = 0; ci < cands.size(); ++ci) {
        double mu = 0.0;
        for (int w = 0; w < worlds; ++w) mu += wt[std::size_t(w)] * vals[ci][std::size_t(w)];
        if (mu > best_val) {
            best_val = mu;
            best_ci = ci;
        }
    }
    return cands[best_ci];
}

double BidModel::bidLogProb(Mask hand_5, int blik, int node_k, const Config& cfg, Bid chosen, int us, int them, int pot,
                            int we_deal, double temperature) const {
    const int blik_suit = suitOf(blik);
    const double v_redeal = lookupV(us, them, pot, 1 - we_deal);

    double best_r2_ev = v_redeal;
    if (node_k < 4) {
        best_r2_ev = -1.0;
        for (int s = 0; s < 4; ++s) {
            if (!cfg.round2_allow_blik_suit && s == blik_suit) continue;
            best_r2_ev = std::max(best_r2_ev, expectedVTake(hand_5, blik, s, node_k + 4, us, them, pot, we_deal));
        }
    }

    // Score Pass vs each legal non-pass action.
    double vals[5];
    Bid acts[5];
    int n_acts = 0;
    int chosen_idx = 0;

    // Pass slot at index 0; its value comes from the best candidate's BidNet V(Pass).
    acts[n_acts++] = Bid::pass();
    double best_vt = -1.0;
    double best_vp = v_redeal;

    if (node_k < 4) {
        const double ev_t = expectedVTake(hand_5, blik, blik_suit, node_k, us, them, pot, we_deal);
        float af[kAuctionFeatDim], ctx[kMatchCtxDim];
        encodeAuctionTake(hand_5, blik, blik_suit, node_k, af);
        evalContext(us, them, pot, we_deal, ev_t, best_r2_ev, ctx);
        double vp = 0.0, vt = 0.0;
        evalBidNetPair(af, ctx, node_k, ev_t, v_redeal, vp, vt);
        best_vp = vp;
        acts[n_acts] = Bid::take();
        vals[n_acts] = vt;
        if (chosen == Bid::take()) chosen_idx = n_acts;
        ++n_acts;
    } else {
        for (int s = 0; s < 4; ++s) {
            if (!cfg.round2_allow_blik_suit && s == blik_suit) continue;
            const double ev_t = expectedVTake(hand_5, blik, s, node_k, us, them, pot, we_deal);
            float af[kAuctionFeatDim], ctx[kMatchCtxDim];
            encodeAuctionTake(hand_5, blik, s, node_k, af);
            evalContext(us, them, pot, we_deal, ev_t, best_r2_ev, ctx);
            double vp = 0.0, vt = 0.0;
            evalBidNetPair(af, ctx, node_k, ev_t, v_redeal, vp, vt);
            if (vt > best_vt) {
                best_vt = vt;
                best_vp = vp;
            }
            const Bid b = Bid::choose(s);
            acts[n_acts] = b;
            vals[n_acts] = vt;
            if (chosen == b) chosen_idx = n_acts;
            ++n_acts;
        }
    }
    vals[0] = best_vp;

    const double inv_t = 1.0 / std::max(1e-4, temperature);
    double max_z = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_acts; ++i) max_z = std::max(max_z, vals[i] * inv_t);
    double sum_e = 0.0;
    for (int i = 0; i < n_acts; ++i) sum_e += std::exp(vals[i] * inv_t - max_z);
    return (vals[chosen_idx] * inv_t - max_z) - std::log(sum_e);
}

}  // namespace belote
