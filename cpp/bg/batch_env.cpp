#include "bg/batch_env.h"
#include "bg/agent.h"
#include "bg/rule_policy.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace bg {

BgBatchPlayEnv::BgBatchPlayEnv(int num_envs, std::uint64_t seed, double bid_noise, double hanging_prob, int threads,
                               ContractMode mode, bool allow_double, const std::string& bid_model_path,
                               double bid_model_prob)
    : num_envs_(num_envs), threads_(std::max(1, threads)), bid_noise_(bid_noise), hanging_prob_(hanging_prob),
      mode_(mode), allow_double_(allow_double), bid_model_prob_(bid_model_prob), rngs_(std::size_t(num_envs)),
      cached_legal_(std::size_t(num_envs)), cached_n_legal_(std::size_t(num_envs), 0) {
    if (num_envs <= 0) throw std::invalid_argument("BgBatchPlayEnv: num_envs must be > 0");
    if (!bid_model_path.empty()) {
        if (!bid_model_.load(bid_model_path)) {
            throw std::runtime_error("BgBatchPlayEnv: failed to load bid model " + bid_model_path);
        }
    }
    resetAll(seed);
}

Round BgBatchPlayEnv::makeFreshDeal(std::mt19937_64& rng) const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);

    for (int attempt = 0; attempt < 400; ++attempt) {
        std::shuffle(deck.begin(), deck.end(), rng);
        const int first = int(rng() % kPlayers);
        int hanging = 0;
        if (hanging_prob_ > 0.0 && u(rng) < hanging_prob_) {
            static constexpr int kHangChoices[5] = {8, 10, 13, 16, 26};
            hanging = kHangChoices[rng() % 5];
        }
        Round r(deck, first, hanging);
        const bool use_bid_net = bid_model_.loaded() && (u(rng) < bid_model_prob_);
        // On 15% of BgBidNet doubled deals, gently lower double_margin so Q-GRU gets rich
        // exposure to realistic competitive doubled contracts without distorting normal auctions.
        const double dbl_margin = (use_bid_net && allow_double_ && u(rng) < 0.15) ? -6.0 : 0.0;
        // Pick natural bidding tuning appropriate for the requested specialist contract mode
        // so acceptance is fast (~40-65%) while keeping every auction 100% natural.
        int at_bonus = 0;
        int nt_bonus = 0;
        if (!use_bid_net) {
            if (mode_ == ContractMode::kAll) {
                at_bonus = (rng() & 1u) ? 8 : 0;
            } else if (mode_ == ContractMode::kAllTrumps) {
                at_bonus = 10;  // realistic aggressive AT tables (~60% AT, matching Master's 68% AT)
            } else if (mode_ == ContractMode::kSuit) {
                at_bonus = 0;
            } else if (mode_ == ContractMode::kNoTrumps) {
                at_bonus = -10;
                nt_bonus = 6;
            }
        }
        while (r.decision() == Round::Decision::kBid) {
            std::uint8_t bid = use_bid_net ? bid_model_.chooseBid(r, allow_double_, dbl_margin)
                                           : naturalBidTuned(r, at_bonus, nt_bonus, allow_double_);
            if (bid_noise_ > 0.0 && u(rng) < bid_noise_) {
                const std::uint8_t avail = r.availableBids();
                std::uint8_t opts[9];
                int n = 0;
                opts[n++] = kPass;
                for (int b = 0; b < 8; ++b) {
                    const std::uint8_t f = std::uint8_t(1u << b);
                    if (!allow_double_ && (f == kDouble || f == kReDouble)) continue;
                    if (avail & f) opts[n++] = f;
                }
                if (n > 1) bid = opts[rng() % std::uint64_t(n)];
            }
            r.bid(bid);
        }
        if (r.finished()) continue;
        const std::uint8_t c = r.contract();
        if (mode_ == ContractMode::kSuit && !(c & 15u)) continue;
        if (mode_ == ContractMode::kNoTrumps && !(c & kNoTrumps)) continue;
        if (mode_ == ContractMode::kAllTrumps && !(c & kAllTrumps)) continue;
        while (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
        }
        if (r.decision() == Round::Decision::kPlay) return r;
    }
    throw std::runtime_error("BgBatchPlayEnv: failed to generate matching deal");
}

void BgBatchPlayEnv::cacheLegal(int env) {
    const Mask legal = deals_[std::size_t(env)].legalMoves();
    auto& arr = cached_legal_[std::size_t(env)];
    arr.fill(-1);
    int n = 0;
    for (Mask m = legal; m && n < kBgMaxLegal; m &= m - 1) arr[std::size_t(n++)] = lowestCard(m);
    cached_n_legal_[std::size_t(env)] = n;
}

void BgBatchPlayEnv::resetAll(std::uint64_t seed) {
    if (int(deals_.size()) != num_envs_) {
        deals_.assign(std::size_t(num_envs_), Round(std::array<int, kCards>{}, 0));
    }
    auto worker = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            rngs_[std::size_t(i)].seed(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(i + 1) * 0xBF58476D1CE4E5B9ull);
            deals_[std::size_t(i)] = makeFreshDeal(rngs_[std::size_t(i)]);
            cacheLegal(i);
        }
    };
    const int nt = std::min(threads_, num_envs_);
    if (nt <= 1) {
        worker(0, num_envs_);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(nt));
    const int chunk = (num_envs_ + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int begin = t * chunk;
        const int end = std::min(num_envs_, begin + chunk);
        if (begin < end) pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
}

void BgBatchPlayEnv::observeBatch(float* state, float* history, std::int32_t* n_plies, float* legal_actions,
                                  std::int32_t* legal_cards, std::int32_t* n_legal, std::int32_t* current_team,
                                  std::uint8_t* contract_out, std::int8_t* belief_targets, float* belief_marginals,
                                  std::uint8_t* belief_allowed) const {
    auto worker = [&](int begin, int end) {
        for (int b = begin; b < end; ++b) {
            const Round& r = deals_[std::size_t(b)];
            if (r.finished()) {
                std::memset(state + std::size_t(b) * kBgStateDim, 0, std::size_t(kBgStateDim) * sizeof(float));
                if (legal_actions)
                    std::memset(legal_actions + std::size_t(b) * kBgMaxLegal * kBgActionDim, 0,
                                std::size_t(kBgMaxLegal * kBgActionDim) * sizeof(float));
                if (legal_cards)
                    std::fill(legal_cards + std::size_t(b) * kBgMaxLegal,
                              legal_cards + std::size_t(b + 1) * kBgMaxLegal, -1);
                if (n_plies) n_plies[b] = 0;
                if (n_legal) n_legal[b] = 1;
                if (current_team) current_team[b] = 0;
                if (contract_out) contract_out[b] = r.contract();
                if (belief_targets)
                    std::fill(belief_targets + std::size_t(b) * 32, belief_targets + std::size_t(b + 1) * 32,
                              std::int8_t(-1));
                if (belief_marginals)
                    std::memset(belief_marginals + std::size_t(b) * 96, 0, 96 * sizeof(float));
                if (belief_allowed)
                    std::memset(belief_allowed + std::size_t(b) * 96, 0, 96);
                continue;
            }
            const int me = r.toMove();
            const PublicState ps = publicState(r, me);
            const std::uint8_t contract = ps.contract;

            const bool need_sampler = (belief_marginals != nullptr || belief_allowed != nullptr);
            if (need_sampler) {
                const DealSampler sampler(ps);
                encodeState(ps, state + std::size_t(b) * kBgStateDim, &sampler.facts());
                const auto raw_marg = sampler.exactMarginals();
                const auto& others = sampler.others();
                // Map sampler's others_[k] (absolute order) -> relSeat(seat, me) - 1 (0=left, 1=partner, 2=right).
                int k_to_rk[3];
                for (int k = 0; k < 3; ++k) k_to_rk[k] = relSeat(others[std::size_t(k)], me) - 1;

                float* bm_row = belief_marginals ? belief_marginals + std::size_t(b) * 32 * 3 : nullptr;
                std::uint8_t* ba_row = belief_allowed ? belief_allowed + std::size_t(b) * 32 * 3 : nullptr;
                if (bm_row) std::fill(bm_row, bm_row + 96, 0.0f);
                if (ba_row) std::fill(ba_row, ba_row + 96, std::uint8_t(0));
                for (int raw_c = 0; raw_c < kCards; ++raw_c) {
                    const int cc = canonCard(raw_c, contract);
                    for (int k = 0; k < 3; ++k) {
                        const int rk = k_to_rk[k];
                        const float p = float(raw_marg[std::size_t(raw_c)][std::size_t(k)]);
                        if (bm_row) bm_row[cc * 3 + rk] = p;
                        if (ba_row) ba_row[cc * 3 + rk] = p > 0.0f ? 1 : 0;
                    }
                }
            } else {
                encodeState(ps, state + std::size_t(b) * kBgStateDim, nullptr);
            }

            if (history) {
                n_plies[b] = encodeHistory(ps, history + std::size_t(b) * kBgMaxPlies * kBgPlyDim);
            } else if (n_plies) {
                n_plies[b] = std::int32_t(ps.plays.size());
            }

            if (legal_actions && legal_cards) {
                n_legal[b] = encodeLegalActions(ps, legal_actions + std::size_t(b) * kBgMaxLegal * kBgActionDim,
                                                legal_cards + std::size_t(b) * kBgMaxLegal);
            } else if (n_legal) {
                n_legal[b] = cached_n_legal_[std::size_t(b)];
            }

            if (current_team) current_team[b] = teamOf(me);
            if (contract_out) contract_out[b] = contract;

            if (belief_targets) {
                std::int8_t* bt_row = belief_targets + std::size_t(b) * 32;
                std::fill(bt_row, bt_row + 32, std::int8_t(-1));
                for (int rk = 0; rk < 3; ++rk) {
                    const int seat = (me + 1 + rk) & 3;
                    for (Mask m = r.hand(seat); m; m &= m - 1) {
                        const int cc = canonCard(lowestCard(m), contract);
                        bt_row[cc] = std::int8_t(rk);
                    }
                }
            }
        }
    };

    const int nt = std::min(threads_, num_envs_);
    if (nt <= 1) {
        worker(0, num_envs_);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(nt));
    const int chunk = (num_envs_ + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int begin = t * chunk;
        const int end = std::min(num_envs_, begin + chunk);
        if (begin < end) pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
}

void BgBatchPlayEnv::stepBatch(const std::int32_t* action_idx, std::uint8_t* done_out, float* reward_t0_out,
                               bool auto_reset) {
    auto worker = [&](int begin, int end) {
        for (int b = begin; b < end; ++b) {
            Round& r = deals_[std::size_t(b)];
            if (r.finished()) {
                done_out[b] = 1;
                reward_t0_out[b] = 0.0f;
                continue;
            }
            const int nl = cached_n_legal_[std::size_t(b)];
            int idx = action_idx[b];
            if (idx < 0 || idx >= nl) idx = 0;
            const int card = cached_legal_[std::size_t(b)][std::size_t(idx)];
            r.play(card, true);
            while (r.decision() == Round::Decision::kAnnounce) {
                r.declare(chooseDeclarations(r.offeredAnnounces()));
            }
            if (r.finished()) {
                done_out[b] = 1;
                const RoundResult& res = r.result();
                reward_t0_out[b] = float(res.sn - res.ew) / 40.0f;
                if (auto_reset) {
                    deals_[std::size_t(b)] = makeFreshDeal(rngs_[std::size_t(b)]);
                    cacheLegal(b);
                } else {
                    cached_n_legal_[std::size_t(b)] = 0;
                }
            } else {
                done_out[b] = 0;
                reward_t0_out[b] = 0.0f;
                cacheLegal(b);
            }
        }
    };

    const int nt = std::min(threads_, num_envs_);
    if (nt <= 1) {
        worker(0, num_envs_);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(nt));
    const int chunk = (num_envs_ + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int begin = t * chunk;
        const int end = std::min(num_envs_, begin + chunk);
        if (begin < end) pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
}

}  // namespace bg
