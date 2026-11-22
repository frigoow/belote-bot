#pragma once

// Phase B: Bulgarian Bridge-Belot Neural Bidding (BgBidModel + All-Action
// Counterfactual Batch Generator).
//
// Strictly reads only the acting seat's 5 initial cards + public auction record
// (zero hidden-card access). Uses a shared per-suit feature tower + explicit
// suit rank s/3.0 (preserving Bulgarian ordered suits: C < D < H < S) and a
// chronological bid-history GRU.

#include "bg/bg.h"
#include "bg/neural.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace bg {

inline constexpr int kBgBidVersion = 1;          // masked (naturalBidMask)
inline constexpr int kBgBidVersionUnmasked = 2;  // unmasked (pure legal Belot bids)
inline constexpr int kBgNumBids = 9;             // 0:Pass, 1:C, 2:D, 3:H, 4:S, 5:NT, 6:AT, 7:x2, 8:x4
inline constexpr int kBgBidSuitDim = 16;    // per-suit features (shared across 4 suits)
inline constexpr int kBgBidGlobalDim = 64;  // global auction + seat + hand summary
inline constexpr int kBgBidStepDim = 13;    // relSeat(4) + bidIndex(9) per prior bid
inline constexpr int kBgMaxBids = 16;

// Maps BidType flag (kPass, kClubs..kReDouble) <-> action index 0..8.
inline constexpr int bidToIndex(std::uint8_t b) {
    switch (b) {
        case kPass: return 0;
        case kClubs: return 1;
        case kDiamonds: return 2;
        case kHearts: return 3;
        case kSpades: return 4;
        case kNoTrumps: return 5;
        case kAllTrumps: return 6;
        case kDouble: return 7;
        case kReDouble: return 8;
        default: return 0;
    }
}

inline constexpr std::uint8_t indexToBid(int idx) {
    switch (idx) {
        case 0: return kPass;
        case 1: return kClubs;
        case 2: return kDiamonds;
        case 3: return kHearts;
        case 4: return kSpades;
        case 5: return kNoTrumps;
        case 6: return kAllTrumps;
        case 7: return kDouble;
        case 8: return kReDouble;
        default: return kPass;
    }
}

// Public view of an auction decision (5-card hand + public bids + match score).
struct BidPublicState {
    int first = 0;
    int me = 0;
    Mask hand5 = 0;
    std::uint8_t avail = 0;  // Round::availableBids() (non-pass legal flags)
    std::vector<std::pair<int, std::uint8_t>> bids;
    int sn = 0, ew = 0, hanging = 0;
};

BidPublicState bidPublicState(const Round& r, int sn = 0, int ew = 0);

// Returns a 9-bit mask of legal bids. When use_natural_mask is true, applies
// our soft natural sanity mask; when false, returns ALL legal Belot bids (pure rules).
std::uint16_t rawLegalBidMask(const BidPublicState& bs, bool allow_double = false);
std::uint16_t naturalBidMask(const BidPublicState& bs, bool allow_double = false);
std::uint16_t bidActionMask(const BidPublicState& bs, bool allow_double = false, bool use_natural_mask = true);

// Encodes BidPublicState into:
//  - suit_feats[4 * kBgBidSuitDim]
//  - global_feats[kBgBidGlobalDim]
//  - bid_hist[kBgMaxBids * kBgBidStepDim]
//  - legal_mask[kBgNumBids] (1.0f if allowed by bidActionMask, else 0.0f)
// Returns n_bids = min(bs.bids.size(), kBgMaxBids).
int encodeBidState(const BidPublicState& bs, bool allow_double, float* suit_feats, float* global_feats,
                   float* bid_hist, float* legal_mask, bool use_natural_mask = true);

// Plain-C++ AVX2+FMA inference for BgBidNet (BGBD binary format).
class BgBidModel {
public:
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }
    bool useNaturalMask() const { return use_natural_mask_; }

    // Predicts Q(s, b) (scaled game-point margin / 30.0f) for all 9 bid actions.
    void forwardQ(const float* suit_feats, const float* global_feats, const float* bid_hist, int n_bids,
                  float* q_out) const;

    // Picks the highest-Q legal bid (using natural mask if version==1, pure legal bids if version==2).
    std::uint8_t chooseBid(const BidPublicState& bs, bool allow_double = false, double double_margin = 0.0) const;
    std::uint8_t chooseBid(const Round& r, bool allow_double = false, double double_margin = 0.0) const;

    // Writes per-action Q values (in game points = 30 * Q) and returns legal mask.
    std::uint16_t evalBids(const BidPublicState& bs, bool allow_double, std::array<double, kBgNumBids>& q_pts) const;

private:
    bool loaded_ = false;
    bool use_natural_mask_ = true;
    int suit_hidden_ = 0, global_hidden_ = 0, gru_hidden_ = 0, trunk_dim_ = 0;
    // Shared per-suit tower (applied identically to each of the 4 suits)
    detail::Linear suit_fc1_, suit_fc2_;
    detail::LayerNorm suit_ln1_, suit_ln2_;
    // Global tower
    detail::Linear glob_fc_;
    detail::LayerNorm glob_ln_;
    // Bid-history GRU
    detail::GruLayer hist_gru_;
    // Trunk + 9-action value head
    detail::ResBlock trunk1_, trunk2_;
    detail::Linear head_;
};

// High-throughput multithreaded C++ counterfactual bid-dataset generator.
// Holds our 3 Specialist FastPolicy models (AT, Suit, NT) and an updatable
// BgBidModel. At every sampled auction decision, branches over ALL legal bids
// on the exact same 32-card deal, finishes each branch's auction, and plays out
// the resulting deal with our 3 Specialist FastPolicy models!
struct CounterfactualBidBatch {
    int n_samples = 0;
    std::vector<float> suit_feats;    // [N, 4, 16]
    std::vector<float> global_feats;  // [N, 64]
    std::vector<float> bid_hist;      // [N, 16, 13]
    std::vector<std::int32_t> n_bids; // [N]
    std::vector<float> legal_mask;    // [N, 9]
    std::vector<float> targets;       // [N, 9] (deal margin / 30.0f for legal bids, 0 otherwise)
};

class BgBatchBidEnv {
public:
    BgBatchBidEnv(const std::string& fast_at_path, const std::string& fast_suit_path,
                  const std::string& fast_nt_path, int threads = 12);

    bool loadBidModel(const std::string& path);

    // Generates all-action counterfactual training samples across `num_deals` deals.
    // Table mix: 50% self-play (BidNet x4), 25% mixed partner (BidNet + naturalBid),
    // 25% vs aggressive-AT natural opponents (naturalBidTuned(+10)).
    // Before a BidModel is loaded, uses naturalBid / naturalBidTuned(+10) as behavior policy.
    CounterfactualBidBatch generateBatch(int num_deals, std::uint64_t seed, double explore_eps = 0.08,
                                         bool allow_double = false, bool use_natural_mask = true) const;

    // Fast C++ mirrored-deal bidding duel: A (loaded BgBidModel) vs B (naturalBid / naturalBidTuned)
    // with BOTH sides playing cards via our 3 Specialist FastPolicy models!
    struct DuelSummary {
        double mean = 0.0;
        double ci = 0.0;
        int deals = 0;
        double pct_suit = 0.0, pct_nt = 0.0, pct_at = 0.0, pct_pass = 0.0;
        double pct_dbl = 0.0, pct_rdbl = 0.0;
        double suit_has_j9 = 0.0, nt_has_ace = 0.0, at_has_j = 0.0;
    };
    DuelSummary evalVsNatural(int num_deals, std::uint64_t seed, int opp_at_bonus = 0, bool mixed_partner = false,
                              bool allow_double = false) const;

private:
    int playFastDeal(Round r) const;
    std::uint8_t pickPolicyBid(const Round& r, int mode, std::mt19937_64& rng, double eps, bool allow_double,
                               bool use_natural_mask) const;

    BgFastPolicyModel fast_at_, fast_suit_, fast_nt_;
    BgBidModel bid_model_;
    int threads_ = 1;
};

}  // namespace bg
