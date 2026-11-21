#include "search/fast_policy.h"

#include "belote/sampler.h"
#include "search/belote_game.h"
#include "search/bid_model.h"
#include "search/endgame.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace belote {

namespace {

// Sparse-input column-major GEMV: skips zero inputs (~88% of state[697] and
// ~50% of ReLU activations are exact 0.0f) and runs contiguous SIMD FMA over
// the output vector for every non-zero input.
void sparseLinearRelu(const float* in, int in_dim, const float* w_t, const float* b, int out_dim, float* out) {
    std::memcpy(out, b, sizeof(float) * std::size_t(out_dim));
    for (int i = 0; i < in_dim; ++i) {
        const float x = in[i];
        if (x == 0.0f) continue;
        const float* col = w_t + std::size_t(i) * std::size_t(out_dim);
        for (int o = 0; o < out_dim; ++o) out[o] += x * col[o];
    }
    for (int o = 0; o < out_dim; ++o)
        if (out[o] < 0.0f) out[o] = 0.0f;
}

void sparseLinearOut(const float* in, int in_dim, const float* w_t, const float* b, int out_dim, float* out) {
    std::memcpy(out, b, sizeof(float) * std::size_t(out_dim));
    for (int i = 0; i < in_dim; ++i) {
        const float x = in[i];
        if (x == 0.0f) continue;
        const float* col = w_t + std::size_t(i) * std::size_t(out_dim);
        for (int o = 0; o < out_dim; ++o) out[o] += x * col[o];
    }
}

void transposeRowMajor(const std::vector<float>& src, int rows, int cols, std::vector<float>& dst) {
    dst.resize(std::size_t(rows) * std::size_t(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[std::size_t(c) * std::size_t(rows) + std::size_t(r)] =
                src[std::size_t(r) * std::size_t(cols) + std::size_t(c)];
}

std::mt19937_64& hybridRng() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

}  // namespace

std::shared_ptr<const FastPolicyModel> FastPolicyModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("FastPolicyModel: cannot open " + path);
    char magic[4]{};
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "BFP1", 4) != 0)
        throw std::runtime_error("FastPolicyModel: bad header magic in " + path);

    std::int32_t dims[4]{};
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!in) throw std::runtime_error("FastPolicyModel: truncated header in " + path);
    if (dims[0] != kStateDim || dims[3] != kCards || dims[1] <= 0 || dims[2] <= 0)
        throw std::runtime_error("FastPolicyModel: incompatible dimensions in " + path);

    auto m = std::make_shared<FastPolicyModel>();
    m->state_dim_ = dims[0];
    m->h1_ = dims[1];
    m->h2_ = dims[2];
    m->out_dim_ = dims[3];

    auto readVec = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float)));
        if (!in) throw std::runtime_error("FastPolicyModel: truncated weights in " + path);
    };
    std::vector<float> tmp;
    readVec(tmp, std::size_t(m->h1_) * std::size_t(m->state_dim_));
    transposeRowMajor(tmp, m->h1_, m->state_dim_, m->w1_t_);
    readVec(m->b1_, std::size_t(m->h1_));

    readVec(tmp, std::size_t(m->h2_) * std::size_t(m->h1_));
    transposeRowMajor(tmp, m->h2_, m->h1_, m->w2_t_);
    readVec(m->b2_, std::size_t(m->h2_));

    readVec(tmp, std::size_t(m->out_dim_) * std::size_t(m->h2_));
    transposeRowMajor(tmp, m->out_dim_, m->h2_, m->w3_t_);
    readVec(m->b3_, std::size_t(m->out_dim_));
    return m;
}

void FastPolicyModel::forward(const float* state, float* logits_out) const {
    thread_local std::vector<float> buf1;
    thread_local std::vector<float> buf2;
    if (int(buf1.size()) < h1_) buf1.resize(std::size_t(h1_));
    if (int(buf2.size()) < h2_) buf2.resize(std::size_t(h2_));
    sparseLinearRelu(state, state_dim_, w1_t_.data(), b1_.data(), h1_, buf1.data());
    sparseLinearRelu(buf1.data(), h1_, w2_t_.data(), b2_.data(), h2_, buf2.data());
    sparseLinearOut(buf2.data(), h2_, w3_t_.data(), b3_.data(), out_dim_, logits_out);
}

void FastPolicyModel::evalObservation(const Observation& obs, float* logits_out) const {
    float state[kStateDim];
    encodeState(obs, state);
    forward(state, logits_out);
}

int FastPolicyModel::chooseCard(const Deal& deal) const {
    Mask legal = deal.legalMoves();
    if (popcount(legal) == 1) return lowestCard(legal);
    const int me = deal.currentPlayer();
    const int trump = deal.trump();
    if (deal.config().belot_queen_first_forfeits) {
        const Mask k_bit = bit(cardOf(trump, kK));
        const Mask q_bit = bit(cardOf(trump, kQ));
        if ((deal.hand(me) & (k_bit | q_bit)) == (k_bit | q_bit) && (legal & k_bit)) legal &= ~q_bit;
    }
    float logits[kCards];
    evalObservation(observe(deal, me), logits);
    int best_card = kNoCard;
    float best_val = -std::numeric_limits<float>::infinity();
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        const float v = logits[canonCard(c, trump)];
        if (v > best_val) {
            best_val = v;
            best_card = c;
        }
    }
    return best_card;
}

double FastPolicyModel::logProbOfPlay(const Observation& obs, int card, double temperature) const {
    Trick cur;
    for (const Play& pl : obs.plays) {
        cur.cards[cur.n] = pl.card;
        cur.players[cur.n] = pl.player;
        if (++cur.n == kPlayers) cur = Trick{};
    }
    const Mask legal = legalMask(obs.hand, cur, obs.trump, obs.observer, obs.cfg);
    if (popcount(legal) <= 1) return 0.0;
    float logits[kCards];
    evalObservation(obs, logits);
    const double inv_t = temperature > 0.0 ? (1.0 / temperature) : 1.0;
    double max_l = -std::numeric_limits<double>::infinity();
    for (Mask m = legal; m; m &= m - 1)
        max_l = std::max(max_l, double(logits[canonCard(lowestCard(m), obs.trump)]) * inv_t);
    double sum_exp = 0.0;
    for (Mask m = legal; m; m &= m - 1)
        sum_exp += std::exp(double(logits[canonCard(lowestCard(m), obs.trump)]) * inv_t - max_l);
    const double chosen_l = double(logits[canonCard(card, obs.trump)]) * inv_t - max_l;
    return chosen_l - std::log(sum_exp);
}

namespace {

// Honest FastPolicy log-likelihood of the observed hidden-seat plays in world w.
double playHistoryLogLik(const Observation& root_obs, const World& w, const FastPolicyModel& policy, double temp) {
    if (root_obs.plays.empty()) return 0.0;
    std::array<Mask, kPlayers> start{};
    for (int p = 0; p < kPlayers; ++p) start[p] = w.hands[p] | root_obs.playedBy(p);
    Deal d = Deal::fromPlay(root_obs.cfg, root_obs.dealer, root_obs.taker, root_obs.trump, start);
    for (int p = 0; p < kPlayers; ++p) {
        if (!d.canSetDeclare(p)) continue;
        if (p == root_obs.observer) d.setDeclare(p, root_obs.declare_self);
        else if (root_obs.playedBy(p)) d.setDeclare(p, root_obs.announced[p]);
    }
    double ll = 0.0;
    for (const Play& pl : root_obs.plays) {
        if (pl.player != root_obs.observer && popcount(d.legalMoves()) > 1) {
            ll += policy.logProbOfPlay(observe(d, pl.player), pl.card, temp);
        }
        d.play(pl.card);
    }
    return ll;
}

// M4 Auction-history log-likelihood for a sampled world with 5/3-split
// marginalization (PLAN.md §3.2):
// Each unseen seat p != observer acted 0, 1, or 2 times during the auction.
// We average P(seat p's joint auction bids | hand_5) over S random 5-card
// subsets of w.hands[p] (with blik excluded for taker since taker received blik
// on top of their initial 5 cards) and cache per (seat, hand8) across worlds.
double auctionHistoryLogLik(const Observation& obs, const World& w, const BidModel& bm,
                            const std::array<int, 2>& match_scores, int match_pot, double temp, int splits,
                            std::mt19937_64& rng, std::unordered_map<std::uint64_t, double>& cache) {
    if (obs.bids_made < 0 || obs.blik == kNoCard) return 0.0;
    const int n_pass = obs.bids_made;
    double total_ll = 0.0;

    for (int p = 0; p < kPlayers; ++p) {
        if (p == obs.observer) continue;
        // Collect this seat's (node_k, Bid) actions in chronological order (0, 1, or 2 actions).
        std::pair<int, Bid> acts[2];
        int n_acts = 0;
        for (int k = 0; k <= n_pass; ++k) {
            if (((obs.dealer + 1 + k) & 3) != p) continue;
            if (k < n_pass) {
                acts[n_acts++] = {k, Bid::pass()};
            } else {
                const Bid tb = (k < 4) ? Bid::take() : Bid::choose(obs.trump);
                acts[n_acts++] = {k, tb};
            }
        }
        if (n_acts == 0) continue;

        const Mask h8 = w.hands[p];
        const std::uint64_t key = (std::uint64_t(p) << 32) | std::uint64_t(h8);
        auto it = cache.find(key);
        if (it != cache.end()) {
            total_ll += it->second;
            continue;
        }

        // Pool of candidate cards for initial 5-card hand:
        // taker always received blik after the auction, so initial 5 came from h8 \ {blik} (7 cards).
        // non-takers' initial 5 came from h8 (8 cards).
        Mask pool_m = (p == obs.taker) ? (h8 & ~bit(obs.blik)) : h8;
        int pool[8];
        int pool_n = 0;
        for (Mask m = pool_m; m; m &= m - 1) pool[pool_n++] = lowestCard(m);
        if (pool_n < 5) continue;

        const int t_p = teamOf(p);
        const int us = match_scores[t_p];
        const int them = match_scores[1 - t_p];
        const int we_deal = (teamOf(obs.dealer) == t_p) ? 1 : 0;

        const int n_s = std::max(1, splits);
        double lls[16];
        const int eff_s = std::min(n_s, 16);
        double max_ll = -std::numeric_limits<double>::infinity();
        for (int s = 0; s < eff_s; ++s) {
            // Partial Fisher-Yates to draw 5 cards uniformly without replacement.
            int perm[8];
            for (int i = 0; i < pool_n; ++i) perm[i] = pool[i];
            Mask h5 = 0;
            for (int i = 0; i < 5; ++i) {
                const int j = i + int(rng() % std::uint64_t(pool_n - i));
                std::swap(perm[i], perm[j]);
                h5 |= bit(perm[i]);
            }
            double joint = 0.0;
            for (int ai = 0; ai < n_acts; ++ai) {
                joint += bm.bidLogProb(h5, obs.blik, acts[ai].first, obs.cfg, acts[ai].second, us, them, match_pot,
                                       we_deal, temp);
            }
            lls[s] = joint;
            max_ll = std::max(max_ll, joint);
        }
        double sum_e = 0.0;
        for (int s = 0; s < eff_s; ++s) sum_e += std::exp(lls[s] - max_ll);
        const double marg_ll = max_ll + std::log(sum_e / double(eff_s));
        cache.emplace(key, marg_ll);
        total_ll += marg_ll;
    }
    return total_ll;
}

}  // namespace

HybridResult hybridSearch(const Deal& deal, const FastPolicyModel& policy, const HybridOptions& options) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("hybridSearch: deal is not in kPlay");
    Mask legal = deal.legalMoves();
    HybridResult res;
    if (popcount(legal) == 1) {
        res.best_move = lowestCard(legal);
        res.move_values.push_back({res.best_move, 0.0});
        return res;
    }

    const int me = deal.currentPlayer();
    const int my_team = teamOf(me);
    const int trump = deal.trump();
    if (deal.config().belot_queen_first_forfeits) {
        const Mask k_bit = bit(cardOf(trump, kK));
        const Mask q_bit = bit(cardOf(trump, kQ));
        if ((deal.hand(me) & (k_bit | q_bit)) == (k_bit | q_bit) && (legal & k_bit)) legal &= ~q_bit;
    }

    if (popcount(legal) == 1) {
        res.best_move = lowestCard(legal);
        res.move_values.push_back({res.best_move, 0.0});
        return res;
    }

    const Observation root_obs = observe(deal, me);
    const WorldSampler sampler(root_obs);

    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    const std::size_t n_moves = moves.size();

    // Exact enumeration when total consistent worlds <= max budget (no racing needed).
    std::vector<World> enum_worlds;
    if (sampler.count() > 0 && sampler.count() <= std::uint64_t(options.worlds)) {
        enum_worlds = sampler.enumerate(std::uint64_t(options.worlds));
        res.exact_enumeration = !enum_worlds.empty();
    }

    std::mt19937_64 local_rng(options.seed);
    std::mt19937_64& rng = options.seed ? local_rng : hybridRng();

    const bool use_ab = options.ab_handoff_trick >= 0 && deal.tricksPlayed() >= options.ab_handoff_trick;
    const bool use_race = options.race && !res.exact_enumeration && n_moves > 1;
    const bool use_me = options.use_match_equity && options.bid_model != nullptr;
    const bool use_auct = options.weight_auction && options.bid_model != nullptr && root_obs.bids_made >= 0;
    const int max_w = res.exact_enumeration ? int(enum_worlds.size()) : options.worlds;
    const int step_w = use_race ? std::max(8, options.race_step) : max_w;
    std::unordered_map<std::uint64_t, double> auct_cache;

    // Track per-world unnormalized log-weights and per-move values so we can
    // compute exact normalized weighted means & paired-difference SEs on the fly.
    std::vector<double> logw;
    std::vector<std::vector<double>> vals(n_moves);
    std::vector<bool> alive(n_moves, true);
    int alive_count = int(n_moves);
    logw.reserve(std::size_t(max_w));
    for (auto& v : vals) v.reserve(std::size_t(max_w));

    int evaluated = 0;
    while (evaluated < max_w && alive_count > 1) {
        const int batch_n = std::min(step_w, max_w - evaluated);
        for (int bi = 0; bi < batch_n; ++bi) {
            World w;
            if (res.exact_enumeration) {
                w = enum_worlds[std::size_t(evaluated + bi)];
            } else {
                auto sw = sampler.sample(rng, 500);
                if (!sw) sw = sampler.sampleUnchecked(rng);
                w = *sw;
            }
            double lw = 0.0;
            if (options.weight_play_history && !root_obs.plays.empty() && max_w > 1) {
                lw += playHistoryLogLik(root_obs, w, policy, options.play_lik_temp);
            }
            if (use_auct && max_w > 1) {
                lw += auctionHistoryLogLik(root_obs, w, *options.bid_model, options.match_scores, options.match_pot,
                                           options.auction_lik_temp, options.auction_splits, rng, auct_cache);
            }
            logw.push_back(lw);

            const Deal base = sampler.replay(w);
            if (use_ab) {
                EndgameOptions eo;
                if (use_me) {
                    eo.objective = EndgameObjective::kMatchEquity;
                    eo.bid_model = options.bid_model;
                    eo.match_scores = options.match_scores;
                    eo.match_pot = options.match_pot;
                }
                const EndgameResult er = solveEndgame(base, eo);
                for (std::size_t i = 0; i < n_moves; ++i) {
                    double v = 0.0;
                    for (const auto& [c, ev] : er.move_values) {
                        if (c == moves[i]) {
                            v = use_me ? (0.5 + double(ev) / 20000.0) : double(ev);
                            break;
                        }
                    }
                    vals[i].push_back(v);
                }
            } else {
                for (std::size_t i = 0; i < n_moves; ++i) {
                    if (!alive[i]) {
                        vals[i].push_back(0.0);
                        continue;
                    }
                    Deal sim = base;
                    sim.play(moves[i]);
                    while (sim.phase() == Phase::kPlay &&
                           (options.rollout_ab_trick < 0 || sim.tricksPlayed() < options.rollout_ab_trick)) {
                        sim.play(policy.chooseCard(sim));
                    }
                    if (sim.phase() == Phase::kPlay) {
                        EndgameOptions eo;
                        eo.tt_log2 = 12;
                        if (use_me) {
                            eo.objective = EndgameObjective::kMatchEquity;
                            eo.bid_model = options.bid_model;
                            eo.match_scores = options.match_scores;
                            eo.match_pot = options.match_pot;
                        }
                        const int val0 = solveEndgameValue(sim, eo);
                        const int v_me = (my_team == 0) ? val0 : -val0;
                        vals[i].push_back(use_me ? (0.5 + double(v_me) / 20000.0) : double(v_me));
                    } else if (use_me) {
                        const int us = options.match_scores[my_team];
                        const int them = options.match_scores[1 - my_team];
                        vals[i].push_back(options.bid_model->evalDealEquity(sim.result(), deal.config(), deal.dealer(),
                                                                            my_team, us, them, options.match_pot));
                    } else {
                        vals[i].push_back(unitMargin(sim, my_team));
                    }
                }
            }
        }
        evaluated += batch_n;

        if (!use_race || evaluated < options.race_min_worlds) continue;

        // Compute normalized weights over the `evaluated` worlds seen so far.
        double max_lw = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < evaluated; ++k) max_lw = std::max(max_lw, logw[std::size_t(k)]);
        std::vector<double> wt(static_cast<std::size_t>(evaluated));
        double sum_w = 0.0;
        double sum_w2 = 0.0;
        for (int k = 0; k < evaluated; ++k) {
            wt[std::size_t(k)] = std::exp(std::max(-30.0, logw[std::size_t(k)] - max_lw));
            sum_w += wt[std::size_t(k)];
        }
        for (double& x : wt) {
            x /= sum_w;
            sum_w2 += x * x;
        }
        const double ess = sum_w2 > 0.0 ? (1.0 / sum_w2) : double(evaluated);
        if (ess < 12.0) continue;  // don't prune if effective sample size is too small

        // Find current leader among alive moves.
        std::size_t leader = 0;
        double leader_mean = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n_moves; ++i) {
            if (!alive[i]) continue;
            double mu = 0.0;
            for (int k = 0; k < evaluated; ++k) mu += wt[std::size_t(k)] * vals[i][std::size_t(k)];
            if (mu > leader_mean) {
                leader_mean = mu;
                leader = i;
            }
        }

        // CRN paired-difference z-test against leader: prune if mean(d) > z * se(d).
        const double denom = std::max(1e-9, 1.0 - sum_w2);
        for (std::size_t i = 0; i < n_moves; ++i) {
            if (!alive[i] || i == leader) continue;
            double mean_d = 0.0;
            for (int k = 0; k < evaluated; ++k) {
                const double d = vals[leader][std::size_t(k)] - vals[i][std::size_t(k)];
                mean_d += wt[std::size_t(k)] * d;
            }
            if (mean_d <= 0.0) continue;
            double var_d = 0.0;
            for (int k = 0; k < evaluated; ++k) {
                const double d = vals[leader][std::size_t(k)] - vals[i][std::size_t(k)];
                const double diff = d - mean_d;
                var_d += wt[std::size_t(k)] * diff * diff;
            }
            // Unbiased Kish ESS variance of weighted mean: (var_d / (1 - sum(wt^2))) * sum(wt^2).
            const double se_d = std::sqrt((var_d / denom) * sum_w2);
            if (mean_d > options.race_z * se_d + 1e-9) {
                alive[i] = false;
                --alive_count;
            }
        }
    }

    // Final weighted means over all evaluated worlds for alive moves (and prefix for pruned moves).
    double max_lw = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < evaluated; ++k) max_lw = std::max(max_lw, logw[std::size_t(k)]);
    std::vector<double> wt(static_cast<std::size_t>(evaluated));
    double sum_w = 0.0;
    for (int k = 0; k < evaluated; ++k) {
        wt[std::size_t(k)] = std::exp(std::max(-30.0, logw[std::size_t(k)] - max_lw));
        sum_w += wt[std::size_t(k)];
    }
    for (double& x : wt) x /= sum_w;

    res.worlds_used = evaluated;
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n_moves; ++i) {
        double mu = -1e9;
        if (alive[i]) {
            mu = 0.0;
            for (int k = 0; k < evaluated; ++k) mu += wt[std::size_t(k)] * vals[i][std::size_t(k)];
        }
        res.move_values.push_back({moves[i], mu});
        if (alive[i] && mu > best) {
            best = mu;
            res.best_move = moves[i];
        }
    }
    res.value = best;
    return res;
}

}  // namespace belote
