#include "bg/bid_model.h"
#include "bg/agent.h"
#include "bg/match_equity.h"
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
#endif

namespace bg {

namespace {

using namespace cardbot::nn;

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

std::uint16_t rawLegalBidMask(const BidPublicState& bs, bool allow_double) {
    std::uint16_t mask = 1u;  // Pass
    for (int s = 0; s < 4; ++s) {
        if (bs.avail & std::uint8_t(1u << s)) mask |= std::uint16_t(1u << (1 + s));
    }
    if (bs.avail & kNoTrumps) mask |= std::uint16_t(1u << 5);
    if (bs.avail & kAllTrumps) mask |= std::uint16_t(1u << 6);
    if (allow_double) {
        if (bs.avail & kDouble) mask |= std::uint16_t(1u << 7);
        if (bs.avail & kReDouble) mask |= std::uint16_t(1u << 8);
    }
    return mask;
}

std::uint16_t bidActionMask(const BidPublicState& bs, bool allow_double, bool use_natural_mask) {
    return use_natural_mask ? naturalBidMask(bs, allow_double) : rawLegalBidMask(bs, allow_double);
}

int encodeBidState(const BidPublicState& bs, bool allow_double, float* suit_feats, float* global_feats,
                   float* bid_hist, float* legal_mask, bool use_natural_mask) {
    std::memset(suit_feats, 0, std::size_t(4 * kBgBidSuitDim) * sizeof(float));
    std::memset(global_feats, 0, std::size_t(kBgBidGlobalDim) * sizeof(float));
    std::memset(bid_hist, 0, std::size_t(kBgMaxBids * kBgBidStepDim) * sizeof(float));

    const std::uint16_t nmask = bidActionMask(bs, allow_double, use_natural_mask);
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
    if ((hdr[0] != kBgBidVersion && hdr[0] != kBgBidVersionUnmasked) || hdr[1] != kBgBidSuitDim ||
        hdr[2] != kBgBidGlobalDim || hdr[3] != kBgBidStepDim || hdr[7] != kBgNumBids) {
        return false;
    }
    use_natural_mask_ = (hdr[0] == kBgBidVersion);
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
    const int nb = encodeBidState(bs, allow_double, sf, gf, bh, lm, use_natural_mask_);
    float q[kBgNumBids]{};
    forwardQ(sf, gf, bh, nb, q);
    for (int i = 0; i < kBgNumBids; ++i) q_pts[static_cast<std::size_t>(i)] = double(q[i]) * 30.0;
    return bidActionMask(bs, allow_double, use_natural_mask_);
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

std::uint8_t BgBidModel::chooseBidMatchEquity(const BidPublicState& bs, bool allow_double, double double_margin,
                                              const BidMatchEquityOptions& me_opts, std::mt19937_64& rng) const {
    if (!me_opts.enabled || me_opts.worlds <= 0 ||
        (std::max(bs.sn, bs.ew) < me_opts.min_score && bs.hanging <= 0)) {
        return chooseBid(bs, allow_double, double_margin);
    }

    std::array<double, kBgNumBids> q_pts{};
    const std::uint16_t mask = evalBids(bs, allow_double, q_pts);

    // Collect candidate bids: always Pass (0) + top_k non-pass bids by Q
    // + Double/Redouble if legal and within dbl_window of best Q.
    std::vector<int> non_pass;
    double best_q = q_pts[0];
    for (int i = 1; i < kBgNumBids; ++i) {
        if (!(mask & (1u << i))) continue;
        if (q_pts[static_cast<std::size_t>(i)] > best_q) best_q = q_pts[static_cast<std::size_t>(i)];
        non_pass.push_back(i);
    }
    if (non_pass.empty()) return kPass;

    std::sort(non_pass.begin(), non_pass.end(),
              [&](int a, int b) { return q_pts[static_cast<std::size_t>(a)] > q_pts[static_cast<std::size_t>(b)]; });

    std::vector<int> cands;
    cands.push_back(0);  // Pass is always a candidate
    const int k_lim = std::min(int(non_pass.size()), std::max(1, me_opts.top_k));
    for (int k = 0; k < k_lim; ++k) cands.push_back(non_pass[static_cast<std::size_t>(k)]);
    for (int i = 7; i < kBgNumBids; ++i) {
        if (!(mask & (1u << i))) continue;
        if (q_pts[static_cast<std::size_t>(i)] < best_q - me_opts.dbl_window) continue;
        if (std::find(cands.begin(), cands.end(), i) == cands.end()) cands.push_back(i);
    }
    if (cands.size() <= 1) return indexToBid(cands[0]);

    // Extract our 5 initial cards and the 27 unknown cards.
    int my5[5];
    int n_my = 0;
    int rem27[27];
    int n_rem = 0;
    for (int c = 0; c < kCards; ++c) {
        if (bs.hand5 & bit(c)) {
            if (n_my < 5) my5[n_my++] = c;
        } else {
            if (n_rem < 27) rem27[n_rem++] = c;
        }
    }
    if (n_my != 5 || n_rem != 27) return chooseBid(bs, allow_double, double_margin);

    const int my_team = teamOf(bs.me);
    const int my_score = my_team == 0 ? bs.sn : bs.ew;
    const int opp_score = my_team == 0 ? bs.ew : bs.sn;
    const auto& me_tbl = MatchEquityTable::instance();
    const double passout_eq = me_tbl.preDealEquity(my_score, opp_score, bs.hanging, teamOf(nextSeat(bs.first)) == my_team);

    auto pickFast = [&](std::uint8_t c) -> const BgFastPolicyModel* {
        const BgFastPolicyModel* sp = (c & kAllTrumps) ? me_opts.fast_at
                                    : (c & kNoTrumps)  ? me_opts.fast_nt
                                                       : me_opts.fast_suit;
        if (sp && sp->loaded()) return sp;
        if (me_opts.fast_fallback && me_opts.fast_fallback->loaded()) return me_opts.fast_fallback;
        return nullptr;
    };

    const int W = me_opts.worlds;
    std::vector<double> log_w(static_cast<std::size_t>(W), 0.0);
    std::vector<std::vector<double>> eqs(cands.size(), std::vector<double>(static_cast<std::size_t>(W), 0.5));
    const double temp = std::max(0.5, me_opts.prior_temp);

    for (int w = 0; w < W; ++w) {
        std::shuffle(rem27, rem27 + 27, rng);
        std::array<int, kCards> deck{};
        int r_idx = 0;
        // Round::dealCards(0, 5) deals seat s's 5 initial cards from deck[4*i + s] (i=0..4).
        for (int pos = 0; pos < 20; ++pos) {
            if ((pos & 3) == bs.me) {
                deck[static_cast<std::size_t>(pos)] = my5[pos >> 2];
            } else {
                deck[static_cast<std::size_t>(pos)] = rem27[r_idx++];
            }
        }
        for (int pos = 20; pos < kCards; ++pos) {
            deck[static_cast<std::size_t>(pos)] = rem27[r_idx++];
        }

        // Replay prior auction bids and accumulate importance log-weight from non-self seats.
        Round rw(deck, bs.first, bs.hanging);
        double lw = 0.0;
        bool valid_replay = true;
        for (const auto& [seat, actual_bid] : bs.bids) {
            if (rw.decision() != Round::Decision::kBid) {
                valid_replay = false;
                break;
            }
            if (rw.toMove() != seat) {
                // Forced pass was already auto-recorded by Round::moveToNextBidder().
                continue;
            }
            if (seat != bs.me) {
                std::array<double, kBgNumBids> pq{};
                const std::uint16_t pm = evalBids(bidPublicState(rw, bs.sn, bs.ew), allow_double, pq);
                const int abi = bidToIndex(actual_bid);
                int n_leg = 0;
                double mx = -1e9;
                for (int j = 0; j < kBgNumBids; ++j) {
                    if (pm & (1u << j)) {
                        ++n_leg;
                        if (pq[static_cast<std::size_t>(j)] > mx) mx = pq[static_cast<std::size_t>(j)];
                    }
                }
                if (n_leg > 1) {
                    if (pm & (1u << abi)) {
                        double z = 0.0;
                        for (int j = 0; j < kBgNumBids; ++j) {
                            if (pm & (1u << j)) z += std::exp((pq[static_cast<std::size_t>(j)] - mx) / temp);
                        }
                        const double p_soft = std::exp((pq[static_cast<std::size_t>(abi)] - mx) / temp) / z;
                        const double p_mix = 0.97 * p_soft + 0.03 / double(n_leg);
                        lw += std::log(p_mix);
                    } else {
                        lw += std::log(0.03 / double(n_leg));
                    }
                }
            }
            rw.bid(actual_bid);
        }
        if (!valid_replay || rw.decision() != Round::Decision::kBid || rw.toMove() != bs.me) {
            log_w[static_cast<std::size_t>(w)] = -1e9;
            continue;
        }
        log_w[static_cast<std::size_t>(w)] = lw;

        // Evaluate EVERY candidate bid on this exact paired 32-card world (CRN)!
        for (std::size_t ci = 0; ci < cands.size(); ++ci) {
            Round rb = rw;
            rb.bid(indexToBid(cands[ci]));
            while (rb.decision() == Round::Decision::kBid) {
                rb.bid(chooseBid(rb, allow_double, double_margin));
            }
            if (rb.finished()) {
                eqs[ci][static_cast<std::size_t>(w)] = passout_eq;
                continue;
            }
            const BgFastPolicyModel* fp = pickFast(rb.contract());
            while (!rb.finished()) {
                if (rb.decision() == Round::Decision::kAnnounce) {
                    rb.declare(chooseDeclarations(rb.offeredAnnounces()));
                } else {
                    const int card = fp ? fp->chooseCard(rb) : rulePolicy(rb);
                    rb.play(card, true);
                }
            }
            eqs[ci][static_cast<std::size_t>(w)] =
                me_tbl.postDealEquity(bs.sn, bs.ew, bs.first, rb.contract(), rb.result(), my_team);
        }
    }

    // Normalize importance weights via log-sum-exp.
    double max_lw = -1e18;
    for (double lw : log_w)
        if (lw > max_lw) max_lw = lw;
    if (max_lw < -1e8) return chooseBid(bs, allow_double, double_margin);

    double w_sum = 0.0;
    std::vector<double> wts(static_cast<std::size_t>(W), 0.0);
    for (int w = 0; w < W; ++w) {
        if (log_w[static_cast<std::size_t>(w)] > -1e8) {
            wts[static_cast<std::size_t>(w)] = std::exp(log_w[static_cast<std::size_t>(w)] - max_lw);
            w_sum += wts[static_cast<std::size_t>(w)];
        }
    }
    if (w_sum <= 1e-12) return chooseBid(bs, allow_double, double_margin);

    int best_bi = cands[0];
    double best_score = -1e18;
    for (std::size_t ci = 0; ci < cands.size(); ++ci) {
        double mean_eq = 0.0;
        for (int w = 0; w < W; ++w) mean_eq += wts[static_cast<std::size_t>(w)] * eqs[ci][static_cast<std::size_t>(w)];
        mean_eq /= w_sum;
        const int bi = cands[ci];
        double score = mean_eq + 1e-4 * q_pts[static_cast<std::size_t>(bi)];
        if (bi >= 7) score -= me_opts.dbl_eq_edge;
        if (score > best_score) {
            best_score = score;
            best_bi = bi;
        }
    }
    return indexToBid(best_bi);
}

double BgBidModel::auctionLogLik(const PublicState& ps, const Hands& hands, bool allow_double, double temp,
                                 int n_subsets) const {
    if (!loaded_ || ps.bids.empty()) return 0.0;
    const double t = std::max(0.5, temp);
    const int k_sub = std::clamp(n_subsets, 1, 8);

    // Pre-extract each non-self seat's 8 cards and build k_sub evenly-covering 5-card subsets
    // (deterministic hash rotation so all 8 cards are covered evenly).
    std::array<std::array<Mask, 8>, kPlayers> sub5{};
    std::array<std::array<double, 8>, kPlayers> prob{};
    std::array<bool, kPlayers> used{};
    for (int p = 0; p < kPlayers; ++p) {
        if (p == ps.me) continue;
        const Mask h8 = hands[static_cast<std::size_t>(p)] | ps.playedBy(p);
        int cards[8];
        int nc = 0;
        for (Mask m = h8; m && nc < 8; m &= m - 1) cards[nc++] = lowestCard(m);
        if (nc < 5) continue;
        const int rot = int((std::uint32_t(h8) * 2654435761u) >> 29) & 7;
        for (int s = 0; s < k_sub; ++s) {
            Mask m5 = 0;
            const int start = (rot + s * 3) % nc;
            for (int j = 0; j < 5; ++j) m5 |= bit(cards[(start + j) % nc]);
            sub5[static_cast<std::size_t>(p)][static_cast<std::size_t>(s)] = m5;
            prob[static_cast<std::size_t>(p)][static_cast<std::size_t>(s)] = 1.0;
        }
    }

    // Walk ps.bids maintaining public auction state (contract + declarer) so we know exact avail.
    std::uint8_t contract = kPass;
    int declarer = -1;
    std::vector<std::pair<int, std::uint8_t>> prefix;
    prefix.reserve(ps.bids.size());

    auto availFor = [&](int seat) -> std::uint8_t {
        const std::uint8_t clean = contract & std::uint8_t(~(kDouble | kReDouble));
        std::uint8_t av = kPass;
        for (std::uint8_t f : {kClubs, kDiamonds, kHearts, kSpades, kNoTrumps, kAllTrumps}) {
            if (clean < f) av |= f;
        }
        if (contract != kPass && !(contract & kReDouble) && declarer >= 0) {
            const bool decl_team = teamOf(seat) == teamOf(declarer);
            if (contract & kDouble) {
                if (decl_team) av |= kReDouble;
            } else if (!decl_team) {
                av |= kDouble;
            }
        }
        return av;
    };

    for (const auto& [seat, bid] : ps.bids) {
        const std::uint8_t av = availFor(seat);
        if (seat != ps.me && av != kPass) {
            used[static_cast<std::size_t>(seat)] = true;
            BidPublicState bs;
            bs.first = ps.first;
            bs.me = seat;
            bs.avail = av;
            bs.bids = prefix;
            bs.sn = ps.sn;
            bs.ew = ps.ew;
            bs.hanging = ps.hanging;
            const int abi = bidToIndex(bid);

            for (int s = 0; s < k_sub; ++s) {
                bs.hand5 = sub5[static_cast<std::size_t>(seat)][static_cast<std::size_t>(s)];
                std::array<double, kBgNumBids> q_pts{};
                const std::uint16_t pm = evalBids(bs, allow_double, q_pts);
                int n_leg = 0;
                double mx = -1e9;
                for (int j = 0; j < kBgNumBids; ++j) {
                    if (pm & (1u << j)) {
                        ++n_leg;
                        if (q_pts[static_cast<std::size_t>(j)] > mx) mx = q_pts[static_cast<std::size_t>(j)];
                    }
                }
                if (n_leg > 1) {
                    double p_act = 0.05 / double(n_leg);
                    if (pm & (1u << abi)) {
                        double z = 0.0;
                        for (int j = 0; j < kBgNumBids; ++j) {
                            if (pm & (1u << j)) z += std::exp((q_pts[static_cast<std::size_t>(j)] - mx) / t);
                        }
                        p_act += 0.95 * (std::exp((q_pts[static_cast<std::size_t>(abi)] - mx) / t) / z);
                    }
                    prob[static_cast<std::size_t>(seat)][static_cast<std::size_t>(s)] *= p_act;
                }
            }
        }
        prefix.push_back({seat, bid});
        if (bid != kPass) {
            if (bid == kDouble || bid == kReDouble) {
                contract = std::uint8_t((contract & ~(kDouble | kReDouble)) | bid);
            } else {
                contract = bid;
                declarer = seat;
            }
        }
    }

    double log_lik = 0.0;
    for (int p = 0; p < kPlayers; ++p) {
        if (!used[static_cast<std::size_t>(p)]) continue;
        double sum = 0.0;
        for (int s = 0; s < k_sub; ++s) sum += prob[static_cast<std::size_t>(p)][static_cast<std::size_t>(s)];
        log_lik += std::log(std::max(sum / double(k_sub), 1e-12));
    }
    return log_lik;
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
                                          bool allow_double, bool use_natural_mask) const {
    // seat_policy: 0 = BidModel (or naturalBid if unloaded), 1 = naturalBid(0), 2 = naturalBidTuned(+10 AT)
    if (seat_policy == 1 || !bid_model_.loaded()) {
        return naturalBidTuned(r, 0, 0, allow_double);
    }
    if (seat_policy == 2) {
        return naturalBidTuned(r, 10, 0, allow_double);
    }
    const BidPublicState bs = bidPublicState(r);
    if (eps > 0.0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < eps) {
        const std::uint16_t m = bidActionMask(bs, allow_double, use_natural_mask);
        int opts[kBgNumBids], n = 0;
        for (int i = 0; i < kBgNumBids; ++i)
            if (m & (1u << i)) opts[n++] = i;
        return indexToBid(opts[rng() % static_cast<std::uint64_t>(n)]);
    }
    return bid_model_.chooseBid(bs, allow_double);
}

CounterfactualBidBatch BgBatchBidEnv::generateBatch(int num_deals, std::uint64_t seed, double explore_eps,
                                                    bool allow_double, bool use_natural_mask) const {
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
                const std::uint16_t nmask = bidActionMask(bs, allow_double, use_natural_mask);

                // Only generate counterfactual training nodes when it's a BidNet seat with >= 2 legal bids!
                int n_opts = 0;
                for (int i = 0; i < kBgNumBids; ++i)
                    if (nmask & (1u << i)) ++n_opts;

                if (seat_pol[me] == 0 && n_opts >= 2) {
                    float sf[4 * kBgBidSuitDim], gf[kBgBidGlobalDim], bh[kBgMaxBids * kBgBidStepDim], lm[kBgNumBids];
                    const int nb = encodeBidState(bs, allow_double, sf, gf, bh, lm, use_natural_mask);
                    float tgt[kBgNumBids]{};

                    // Branch over EVERY legal bid on this exact 32-card deal!
                    for (int bi = 0; bi < kBgNumBids; ++bi) {
                        if (!(nmask & (1u << bi))) continue;
                        Round rb = r;
                        rb.bid(indexToBid(bi));
                        while (rb.decision() == Round::Decision::kBid) {
                            const int s2 = rb.toMove();
                            rb.bid(pickPolicyBid(rb, seat_pol[s2], rng, 0.0, allow_double, use_natural_mask));
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
                const std::uint8_t act = pickPolicyBid(r, seat_pol[me], rng, explore_eps, allow_double, use_natural_mask);
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
    long c_suit = 0, c_nt = 0, c_at = 0, c_pass = 0, c_dbl = 0, c_rdbl = 0;
    long suit_total = 0, suit_j9 = 0, nt_total = 0, nt_ace = 0, at_total = 0, at_j = 0;

    const int nt = std::min(threads_, std::max(1, num_deals));
    std::vector<std::thread> pool;
    auto worker = [&](int begin, int end) {
        long ls = 0, lnt = 0, lat = 0, lp = 0, ld = 0, lr = 0;
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
                    else {
                        if (r.contract() & kAllTrumps) ++lat;
                        else if (r.contract() & kNoTrumps) ++lnt;
                        else ++ls;
                        if (r.contract() & kReDouble) ++lr;
                        else if (r.contract() & kDouble) ++ld;
                    }
                }
                const int m0 = playFastDeal(std::move(r));
                pair += 0.5 * double(side == 0 ? m0 : -m0);
            }
            pair_scores[static_cast<std::size_t>(i)] = pair;
        }
        std::lock_guard<std::mutex> lk(mu);
        c_suit += ls; c_nt += lnt; c_at += lat; c_pass += lp; c_dbl += ld; c_rdbl += lr;
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
    out.pct_dbl = 100.0 * double(c_dbl) / tot;
    out.pct_rdbl = 100.0 * double(c_rdbl) / tot;
    out.suit_has_j9 = suit_total > 0 ? 100.0 * double(suit_j9) / double(suit_total) : 100.0;
    out.nt_has_ace = nt_total > 0 ? 100.0 * double(nt_ace) / double(nt_total) : 100.0;
    out.at_has_j = at_total > 0 ? 100.0 * double(at_j) / double(at_total) : 100.0;
    return out;
}

}  // namespace bg
