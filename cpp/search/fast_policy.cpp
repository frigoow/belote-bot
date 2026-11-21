#include "search/fast_policy.h"

#include "belote/sampler.h"
#include "search/belote_game.h"
#include "search/endgame.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace belote {

namespace {

void linearRelu(const float* in, int in_dim, const float* w, const float* b, int out_dim, float* out) {
    for (int o = 0; o < out_dim; ++o) {
        const float* row = w + std::size_t(o) * std::size_t(in_dim);
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) acc += row[i] * in[i];
        out[o] = acc > 0.0f ? acc : 0.0f;
    }
}

void linearOut(const float* in, int in_dim, const float* w, const float* b, int out_dim, float* out) {
    for (int o = 0; o < out_dim; ++o) {
        const float* row = w + std::size_t(o) * std::size_t(in_dim);
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) acc += row[i] * in[i];
        out[o] = acc;
    }
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
    readVec(m->w1_, std::size_t(m->h1_) * std::size_t(m->state_dim_));
    readVec(m->b1_, std::size_t(m->h1_));
    readVec(m->w2_, std::size_t(m->h2_) * std::size_t(m->h1_));
    readVec(m->b2_, std::size_t(m->h2_));
    readVec(m->w3_, std::size_t(m->out_dim_) * std::size_t(m->h2_));
    readVec(m->b3_, std::size_t(m->out_dim_));
    return m;
}

void FastPolicyModel::forward(const float* state, float* logits_out) const {
    thread_local std::vector<float> buf1;
    thread_local std::vector<float> buf2;
    if (int(buf1.size()) < h1_) buf1.resize(std::size_t(h1_));
    if (int(buf2.size()) < h2_) buf2.resize(std::size_t(h2_));
    linearRelu(state, state_dim_, w1_.data(), b1_.data(), h1_, buf1.data());
    linearRelu(buf1.data(), h1_, w2_.data(), b2_.data(), h2_, buf2.data());
    linearOut(buf2.data(), h2_, w3_.data(), b3_.data(), out_dim_, logits_out);
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

double FastPolicyModel::logProbOfPlay(const Observation& obs, int card) const {
    // Reconstruct current trick to find legal moves at obs.
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
    float max_l = -std::numeric_limits<float>::infinity();
    for (Mask m = legal; m; m &= m - 1) max_l = std::max(max_l, logits[canonCard(lowestCard(m), obs.trump)]);
    double sum_exp = 0.0;
    for (Mask m = legal; m; m &= m - 1)
        sum_exp += std::exp(double(logits[canonCard(lowestCard(m), obs.trump)] - max_l));
    const double chosen_l = double(logits[canonCard(card, obs.trump)] - max_l);
    return chosen_l - std::log(sum_exp);
}

namespace {

// Honest FastPolicy log-likelihood of the observed hidden-seat plays in world w.
double playHistoryLogLik(const Observation& root_obs, const World& w, const FastPolicyModel& policy) {
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
            ll += policy.logProbOfPlay(observe(d, pl.player), pl.card);
        }
        d.play(pl.card);
    }
    return ll;
}

}  // namespace

HybridResult hybridSearch(const Deal& deal, const FastPolicyModel& policy, const HybridOptions& options) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("hybridSearch: deal is not in kPlay");
    const Mask legal = deal.legalMoves();
    HybridResult res;
    if (popcount(legal) == 1) {
        res.best_move = lowestCard(legal);
        res.move_values.push_back({res.best_move, 0.0});
        return res;
    }

    const int me = deal.currentPlayer();
    const int my_team = teamOf(me);
    const Observation root_obs = observe(deal, me);
    const WorldSampler sampler(root_obs);

    std::vector<World> worlds;
    if (sampler.count() > 0 && sampler.count() <= std::uint64_t(options.worlds)) {
        worlds = sampler.enumerate(std::uint64_t(options.worlds));
        res.exact_enumeration = !worlds.empty();
    }
    if (worlds.empty()) {
        std::mt19937_64 local_rng(options.seed);
        std::mt19937_64& rng = options.seed ? local_rng : hybridRng();
        worlds.reserve(std::size_t(options.worlds));
        for (int i = 0; i < options.worlds; ++i) {
            auto w = sampler.sample(rng, 500);
            if (!w) w = sampler.sampleUnchecked(rng);
            worlds.push_back(*w);
        }
    }

    // Weight worlds by FastPolicy play-history log-likelihood (tempered slightly for robustness).
    std::vector<double> weights(worlds.size(), 1.0);
    if (options.weight_play_history && !root_obs.plays.empty() && worlds.size() > 1) {
        std::vector<double> logw(worlds.size(), 0.0);
        double max_lw = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < worlds.size(); ++i) {
            logw[i] = playHistoryLogLik(root_obs, worlds[i], policy);
            max_lw = std::max(max_lw, logw[i]);
        }
        double sum_w = 0.0;
        for (std::size_t i = 0; i < worlds.size(); ++i) {
            weights[i] = std::exp(std::max(-30.0, logw[i] - max_lw));
            sum_w += weights[i];
        }
        if (sum_w > 0.0)
            for (double& wt : weights) wt /= sum_w;
    } else if (!worlds.empty()) {
        const double inv = 1.0 / double(worlds.size());
        std::fill(weights.begin(), weights.end(), inv);
    }

    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    std::vector<double> sums(moves.size(), 0.0);
    const bool use_ab = options.ab_handoff_trick >= 0 && deal.tricksPlayed() >= options.ab_handoff_trick;

    for (std::size_t wi = 0; wi < worlds.size(); ++wi) {
        const double wt = weights[wi];
        const Deal base = sampler.replay(worlds[wi]);
        if (use_ab) {
            const EndgameResult er = solveEndgame(base);
            for (std::size_t i = 0; i < moves.size(); ++i) {
                for (const auto& [c, val] : er.move_values) {
                    if (c == moves[i]) {
                        sums[i] += wt * double(val);
                        break;
                    }
                }
            }
            continue;
        }
        for (std::size_t i = 0; i < moves.size(); ++i) {
            Deal sim = base;
            sim.play(moves[i]);
            while (sim.phase() == Phase::kPlay) sim.play(policy.chooseCard(sim));
            sums[i] += wt * unitMargin(sim, my_team);
        }
    }

    res.worlds_used = int(worlds.size());
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < moves.size(); ++i) {
        res.move_values.push_back({moves[i], sums[i]});
        if (sums[i] > best) {
            best = sums[i];
            res.best_move = moves[i];
        }
    }
    res.value = best;
    return res;
}

}  // namespace belote
