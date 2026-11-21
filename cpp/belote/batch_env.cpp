#include "belote/batch_env.h"

#include "agents/agent.h"
#include "belote/match.h"
#include "belote/observation.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace belote {

BatchPlayEnv::BatchPlayEnv(int num_envs, const Config& cfg, std::uint64_t seed, double bid_threshold,
                           double bid_jitter)
    : num_envs_(num_envs), cfg_(cfg), bid_threshold_(bid_threshold), bid_jitter_(bid_jitter), rng_(seed) {
    if (num_envs_ <= 0) throw std::invalid_argument("BatchPlayEnv: num_envs must be positive");
    deals_.reserve(std::size_t(num_envs_));
    cached_legal_.resize(std::size_t(num_envs_));
    cached_n_legal_.resize(std::size_t(num_envs_), 0);
    for (int i = 0; i < num_envs_; ++i) deals_.push_back(makeFreshDeal());
    // Cache initial legal cards so stepBatch works even before observeBatch.
    for (int b = 0; b < num_envs_; ++b) {
        Mask m = deals_[std::size_t(b)].legalMoves();
        int k = 0;
        cached_legal_[std::size_t(b)].fill(-1);
        for (; m; m &= m - 1) cached_legal_[std::size_t(b)][std::size_t(k++)] = lowestCard(m);
        cached_n_legal_[std::size_t(b)] = k;
    }
}

Deal BatchPlayEnv::makeFreshDeal() {
    std::uniform_real_distribution<double> jitter_dist(-bid_jitter_, bid_jitter_);
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    for (;;) {
        std::shuffle(deck.begin(), deck.end(), rng_);
        const int dealer = int(rng_() % kPlayers);
        Deal d(cfg_, dealer, deck);
        while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) {
            const double thr = bid_jitter_ > 0.0 ? (bid_threshold_ + jitter_dist(rng_)) : bid_threshold_;
            d.bid(ruleBid(d, thr));
        }
        if (d.phase() == Phase::kPlay) return d;
    }
}

void BatchPlayEnv::resetAll(std::uint64_t seed) {
    rng_.seed(seed);
    for (int b = 0; b < num_envs_; ++b) {
        deals_[std::size_t(b)] = makeFreshDeal();
        Mask m = deals_[std::size_t(b)].legalMoves();
        int k = 0;
        cached_legal_[std::size_t(b)].fill(-1);
        for (; m; m &= m - 1) cached_legal_[std::size_t(b)][std::size_t(k++)] = lowestCard(m);
        cached_n_legal_[std::size_t(b)] = k;
    }
}

void BatchPlayEnv::observeBatch(float* state, float* history, std::int32_t* n_plies, float* legal_actions,
                                std::int32_t* legal_cards, std::int32_t* n_legal, std::int32_t* current_team,
                                std::int8_t* belief_targets, std::uint8_t* belief_allowed) const {
    for (int b = 0; b < num_envs_; ++b) {
        const Deal& d = deals_[std::size_t(b)];
        const int me = d.currentPlayer();
        const int trump = d.trump();
        const Observation obs = observe(d, me);

        encodeState(obs, state + std::size_t(b) * kStateDim);
        if (history) {
            const int np = encodeHistory(obs, history + std::size_t(b) * kMaxPlies * kPlyDim);
            if (n_plies) n_plies[b] = np;
        } else if (n_plies) {
            n_plies[b] = int(obs.plays.size());
        }

        const int nl = cached_n_legal_[std::size_t(b)];
        n_legal[b] = nl;
        current_team[b] = teamOf(me);

        float* act_base = legal_actions + std::size_t(b) * kMaxLegal * kActionDim;
        std::memset(act_base, 0, sizeof(float) * kMaxLegal * kActionDim);
        std::int32_t* card_base = legal_cards + std::size_t(b) * kMaxLegal;
        for (int k = 0; k < kMaxLegal; ++k) {
            const int c = cached_legal_[std::size_t(b)][std::size_t(k)];
            card_base[k] = c;
            if (k < nl) encodeAction(obs, c, act_base + std::size_t(k) * kActionDim);
        }

        if (belief_targets || belief_allowed) {
            const Constraints con = inferConstraints(obs);
            const Mask known = obs.hand | obs.played();
            std::int8_t* bt = belief_targets ? (belief_targets + std::size_t(b) * kCards) : nullptr;
            std::uint8_t* ba = belief_allowed ? (belief_allowed + std::size_t(b) * kCards * 3) : nullptr;
            if (bt) std::memset(bt, -1, kCards);
            if (ba) std::memset(ba, 0, kCards * 3);

            for (int c = 0; c < kCards; ++c) {
                if (known & bit(c)) continue;
                const int cc = canonCard(c, trump);
                if (bt) {
                    for (int r = 1; r <= 3; ++r) {
                        const int p = (me + r) & 3;
                        if (d.hand(p) & bit(c)) {
                            bt[cc] = std::int8_t(r - 1);
                            break;
                        }
                    }
                }
                if (ba) {
                    int pinned_r = -1;
                    for (int r = 1; r <= 3; ++r) {
                        const int p = (me + r) & 3;
                        if (con.pinned[p] & bit(c)) pinned_r = r - 1;
                    }
                    for (int k = 0; k < 3; ++k) {
                        const int p = (me + k + 1) & 3;
                        if (con.excluded[p] & bit(c)) continue;
                        if (pinned_r >= 0 && pinned_r != k) continue;
                        ba[cc * 3 + k] = 1;
                    }
                }
            }
        }
    }
}

void BatchPlayEnv::stepBatch(const std::int32_t* action_idx, std::uint8_t* done_out, float* reward_t0_out) {
    for (int b = 0; b < num_envs_; ++b) {
        Deal& d = deals_[std::size_t(b)];
        const int idx = action_idx[b];
        if (idx < 0 || idx >= cached_n_legal_[std::size_t(b)])
            throw std::out_of_range("BatchPlayEnv::stepBatch: action index out of range");
        const int card = cached_legal_[std::size_t(b)][std::size_t(idx)];
        d.play(card);
        if (d.phase() == Phase::kDone) {
            const RecordedDeal rec = recordDeal(d.result(), d.config());
            done_out[b] = 1;
            reward_t0_out[b] = float(rec.scores[0] - rec.scores[1]) / 25.0f;
            d = makeFreshDeal();
        } else {
            done_out[b] = 0;
            reward_t0_out[b] = 0.0f;
        }
        Mask m = d.legalMoves();
        int k = 0;
        cached_legal_[std::size_t(b)].fill(-1);
        for (; m; m &= m - 1) cached_legal_[std::size_t(b)][std::size_t(k++)] = lowestCard(m);
        cached_n_legal_[std::size_t(b)] = k;
    }
}

}  // namespace belote
