#include "bg/agent.h"
#include "bg/batch_env.h"
#include "bg/bg.h"
#include "bg/bid_model.h"
#include "bg/encoder.h"
#include "bg/neural.h"
#include "bg/rule_policy.h"
#include "bg/view.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace bg;

namespace {

SearchOptions parseOppOpts(const std::string& spec, bool& is_policy) {
    is_policy = (spec == "policy" || spec.empty());
    SearchOptions o;
    if (is_policy) return o;
    std::stringstream ss(spec);
    for (std::string kv; std::getline(ss, kv, ',');) {
        if (kv == "policy") {
            is_policy = true;
            continue;
        }
        const auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "w") o.worlds = std::stoi(v);
        else if (k == "exact") o.root_exact_trick = std::stoi(v);
        else if (k == "end") o.endgame_tricks = std::stoi(v);
        else if (k == "lik") o.lik_eps = std::stod(v);
        else if (k == "auction") o.auction_weight = std::stod(v);
        else if (k == "pool") o.pool = std::stoi(v);
        else if (k == "z") o.z = std::stod(v);
        else if (k == "m") o.margin = std::stod(v);
    }
    return o;
}

ContractMode parseMode(const std::string& mode) {
    if (mode == "all" || mode.empty()) return ContractMode::kAll;
    if (mode == "suit") return ContractMode::kSuit;
    if (mode == "nt") return ContractMode::kNoTrumps;
    if (mode == "at") return ContractMode::kAllTrumps;
    throw std::invalid_argument("unknown contract mode: " + mode);
}

// Mirrored evaluation batch: runs 2*N mirrored deals (our agent vs C++ opponent),
// stepping opponent decisions in parallel across C++ threads and pausing whenever
// any active game needs a card from our Python agent.
class BgEvalSession {
public:
    BgEvalSession(int num_deals, std::uint64_t seed, const std::string& opp_spec, int threads,
                  const std::string& mode_str = "all", bool allow_double = false)
        : threads_(std::max(1, threads)) {
        opp_opts_ = parseOppOpts(opp_spec, opp_is_policy_);
        const ContractMode mode = parseMode(mode_str);
        const int at_bonus = (mode == ContractMode::kAllTrumps) ? 10 : (mode == ContractMode::kNoTrumps ? -10 : 0);
        const int nt_bonus = (mode == ContractMode::kNoTrumps) ? 6 : 0;
        std::array<int, kCards> deck;
        std::iota(deck.begin(), deck.end(), 0);
        int produced = 0;
        for (int i = 0; produced < num_deals && i < num_deals * 10; ++i) {
            std::mt19937_64 rng(seed * 1000003ull + std::uint64_t(i));
            std::shuffle(deck.begin(), deck.end(), rng);
            const int first = i % kPlayers;
            Round probe(deck, first);
            while (probe.decision() == Round::Decision::kBid)
                probe.bid(naturalBidTuned(probe, at_bonus, nt_bonus, allow_double));
            if (probe.finished()) continue;
            const std::uint8_t c = probe.contract();
            if (mode == ContractMode::kSuit && !(c & 15u)) continue;
            if (mode == ContractMode::kNoTrumps && !(c & kNoTrumps)) continue;
            if (mode == ContractMode::kAllTrumps && !(c & kAllTrumps)) continue;
            while (probe.decision() == Round::Decision::kAnnounce)
                probe.declare(chooseDeclarations(probe.offeredAnnounces()));
            if (probe.finished()) continue;

            const std::uint64_t game_seed = rng();
            const int pair_idx = int(pair_scores_.size());
            pair_scores_.push_back(0.0);
            for (int side = 0; side < 2; ++side) {
                Game g;
                g.round = probe;
                g.our_team = side;
                g.pair_idx = pair_idx;
                g.rng.seed(game_seed);
                games_.push_back(std::move(g));
            }
            ++produced;
        }
        advanceOpponents();
    }

    int numPairs() const { return int(pair_scores_.size()); }
    int numActive() const { return int(active_.size()); }
    bool done() const { return active_.empty(); }

    py::dict observeActive(bool include_history) const {
        const std::size_t n = active_.size();
        py::array_t<float> state({py::ssize_t(n), py::ssize_t(kBgStateDim)});
        py::array_t<float> actions({py::ssize_t(n), py::ssize_t(kBgMaxLegal), py::ssize_t(kBgActionDim)});
        py::array_t<std::int32_t> legal_cards({py::ssize_t(n), py::ssize_t(kBgMaxLegal)});
        py::array_t<std::int32_t> n_legal({py::ssize_t(n)});
        py::array_t<std::int32_t> n_plies({py::ssize_t(n)});
        py::array_t<float> history;
        if (include_history) {
            history = py::array_t<float>({py::ssize_t(n), py::ssize_t(kBgMaxPlies), py::ssize_t(kBgPlyDim)});
        }

        float* s_ptr = state.mutable_data();
        float* a_ptr = actions.mutable_data();
        std::int32_t* lc_ptr = legal_cards.mutable_data();
        std::int32_t* nl_ptr = n_legal.mutable_data();
        std::int32_t* np_ptr = n_plies.mutable_data();
        float* h_ptr = include_history ? history.mutable_data() : nullptr;

        {
            py::gil_scoped_release release;
            auto worker = [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    const Round& r = games_[active_[i]].round;
                    const PublicState ps = publicState(r, r.toMove());
                    encodeState(ps, s_ptr + i * kBgStateDim);
                    nl_ptr[i] = encodeLegalActions(ps, a_ptr + i * kBgMaxLegal * kBgActionDim,
                                                   lc_ptr + i * kBgMaxLegal);
                    if (h_ptr) np_ptr[i] = encodeHistory(ps, h_ptr + i * kBgMaxPlies * kBgPlyDim);
                    else np_ptr[i] = std::int32_t(ps.plays.size());
                }
            };
            runParallel(n, worker);
        }

        py::dict out;
        out["state"] = std::move(state);
        out["legal_actions"] = std::move(actions);
        out["legal_cards"] = std::move(legal_cards);
        out["n_legal"] = std::move(n_legal);
        out["n_plies"] = std::move(n_plies);
        if (include_history) out["history"] = std::move(history);
        return out;
    }

    void stepActive(py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> action_idx) {
        if (std::size_t(action_idx.size()) != active_.size())
            throw std::invalid_argument("BgEvalSession::stepActive: size mismatch");
        const std::int32_t* ptr = action_idx.data();
        {
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < active_.size(); ++i) {
                Game& g = games_[active_[i]];
                Mask m = g.round.legalMoves();
                int idx = ptr[i];
                int card = lowestCard(m);
                for (int k = 0; m; m &= m - 1, ++k) {
                    if (k == idx) {
                        card = lowestCard(m);
                        break;
                    }
                }
                g.round.play(card, true);
            }
            advanceOpponents();
        }
    }

    py::dict summary() const {
        const std::size_t n = pair_scores_.size();
        double mean = 0.0, var = 0.0;
        for (double x : pair_scores_) mean += x / double(std::max<std::size_t>(1, n));
        if (n > 1) {
            for (double x : pair_scores_) var += (x - mean) * (x - mean) / double(n - 1);
        }
        const double ci = n > 1 ? 1.96 * std::sqrt(var / double(n)) : 0.0;
        py::dict out;
        out["mean"] = mean;
        out["ci"] = ci;
        out["pairs"] = int(n);
        out["pair_scores"] = pair_scores_;
        return out;
    }

private:
    struct Game {
        Round round{std::array<int, kCards>{}, 0};
        int our_team = 0;
        int pair_idx = 0;
        bool scored = false;
        std::mt19937_64 rng;
    };

    template <typename Fn>
    void runParallel(std::size_t n, Fn&& fn) const {
        const std::size_t nt = std::min(std::size_t(threads_), n);
        if (nt <= 1) {
            fn(0, n);
            return;
        }
        std::vector<std::thread> pool;
        pool.reserve(nt);
        const std::size_t chunk = (n + nt - 1) / nt;
        for (std::size_t t = 0; t < nt; ++t) {
            const std::size_t b = t * chunk;
            const std::size_t e = std::min(n, b + chunk);
            if (b < e) pool.emplace_back(fn, b, e);
        }
        for (auto& th : pool) th.join();
    }

    void advanceOpponents() {
        std::vector<std::size_t> pending;
        pending.reserve(games_.size());
        for (std::size_t i = 0; i < games_.size(); ++i)
            if (!games_[i].scored) pending.push_back(i);

        auto worker = [&](std::size_t begin, std::size_t end) {
            for (std::size_t idx = begin; idx < end; ++idx) {
                Game& g = games_[pending[idx]];
                while (!g.round.finished()) {
                    if (g.round.decision() == Round::Decision::kAnnounce) {
                        g.round.declare(chooseDeclarations(g.round.offeredAnnounces()));
                        continue;
                    }
                    if (teamOf(g.round.toMove()) == g.our_team) break;
                    int card;
                    if (opp_is_policy_ || popcount(g.round.legalMoves()) == 1) {
                        card = rulePolicy(g.round);
                    } else {
                        SearchOptions o = opp_opts_;
                        o.seed = g.rng() | 1;
                        card = searchCard(publicState(g.round, g.round.toMove()), o).card;
                    }
                    g.round.play(card, true);
                }
            }
        };
        runParallel(pending.size(), worker);

        active_.clear();
        for (std::size_t i : pending) {
            Game& g = games_[i];
            if (g.round.finished()) {
                g.scored = true;
                const int m = g.round.result().sn - g.round.result().ew;
                pair_scores_[std::size_t(g.pair_idx)] += 0.5 * double(g.our_team == 0 ? m : -m);
            } else {
                active_.push_back(i);
            }
        }
    }

    int threads_ = 1;
    bool opp_is_policy_ = true;
    SearchOptions opp_opts_;
    std::vector<Game> games_;
    std::vector<std::size_t> active_;
    std::vector<double> pair_scores_;
};

}  // namespace

PYBIND11_MODULE(bg_cpp, m) {
    m.doc() = "Bulgarian Bridge-Belot C++ engine, encoder, vectorized self-play env, and mirrored evaluator";

    m.attr("ENCODER_VERSION") = kBgEncoderVersion;
    m.attr("STATE_DIM") = kBgStateDim;
    m.attr("PLY_DIM") = kBgPlyDim;
    m.attr("MAX_PLIES") = kBgMaxPlies;
    m.attr("ACTION_DIM") = kBgActionDim;
    m.attr("MAX_LEGAL") = kBgMaxLegal;

    m.def("canon_suit", &canonSuit);
    m.def("raw_suit", &rawSuit);
    m.def("canon_card", &canonCard);
    m.def("raw_card", &rawCard);
    m.def("rel_seat", &relSeat);

    py::class_<BgBatchPlayEnv>(m, "BgBatchPlayEnv")
        .def(py::init([](int num_envs, std::uint64_t seed, double bid_noise, double hanging_prob, int threads,
                         const std::string& mode, bool allow_double) {
                 return new BgBatchPlayEnv(num_envs, seed, bid_noise, hanging_prob, threads, parseMode(mode),
                                           allow_double);
             }),
             py::arg("num_envs"), py::arg("seed") = 1, py::arg("bid_noise") = 0.0,
             py::arg("hanging_prob") = 0.0, py::arg("threads") = 8, py::arg("mode") = "all",
             py::arg("allow_double") = false)
        .def_property_readonly("num_envs", &BgBatchPlayEnv::numEnvs)
        .def("reset_all", &BgBatchPlayEnv::resetAll, py::arg("seed"))
        .def("observe_batch",
             [](const BgBatchPlayEnv& self, bool include_history, bool include_beliefs,
                bool include_belief_targets, bool include_belief_marginals) {
                 const bool want_targets = include_beliefs || include_belief_targets;
                 const bool want_marginals = include_beliefs || include_belief_marginals;
                 const int B = self.numEnvs();
                 py::array_t<float> state({B, kBgStateDim});
                 py::array_t<float> legal_actions({B, kBgMaxLegal, kBgActionDim});
                 py::array_t<std::int32_t> legal_cards({B, kBgMaxLegal});
                 py::array_t<std::int32_t> n_legal(B);
                 py::array_t<std::int32_t> n_plies(B);
                 py::array_t<std::int32_t> current_team(B);
                 py::array_t<std::uint8_t> contracts(B);
                 py::array_t<float> history;
                 py::array_t<std::int8_t> belief_targets;
                 py::array_t<float> belief_marginals;
                 py::array_t<std::uint8_t> belief_allowed;

                 float* h_ptr = nullptr;
                 if (include_history) {
                     history = py::array_t<float>({B, kBgMaxPlies, kBgPlyDim});
                     h_ptr = history.mutable_data();
                 }
                 std::int8_t* bt_ptr = nullptr;
                 float* bm_ptr = nullptr;
                 std::uint8_t* ba_ptr = nullptr;
                 if (want_targets) {
                     belief_targets = py::array_t<std::int8_t>({B, kCards});
                     bt_ptr = belief_targets.mutable_data();
                 }
                 if (want_marginals) {
                     belief_marginals = py::array_t<float>({B, kCards, 3});
                     belief_allowed = py::array_t<std::uint8_t>({B, kCards, 3});
                     bm_ptr = belief_marginals.mutable_data();
                     ba_ptr = belief_allowed.mutable_data();
                 }

                 float* s_ptr = state.mutable_data();
                 float* la_ptr = legal_actions.mutable_data();
                 std::int32_t* lc_ptr = legal_cards.mutable_data();
                 std::int32_t* nl_ptr = n_legal.mutable_data();
                 std::int32_t* np_ptr = n_plies.mutable_data();
                 std::int32_t* ct_ptr = current_team.mutable_data();
                 std::uint8_t* c_ptr = contracts.mutable_data();

                 {
                     py::gil_scoped_release release;
                     self.observeBatch(s_ptr, h_ptr, np_ptr, la_ptr, lc_ptr, nl_ptr, ct_ptr, c_ptr,
                                       bt_ptr, bm_ptr, ba_ptr);
                 }

                 py::dict out;
                 out["state"] = std::move(state);
                 out["legal_actions"] = std::move(legal_actions);
                 out["legal_cards"] = std::move(legal_cards);
                 out["n_legal"] = std::move(n_legal);
                 out["n_plies"] = std::move(n_plies);
                 out["current_team"] = std::move(current_team);
                 out["contract"] = std::move(contracts);
                 if (include_history) out["history"] = std::move(history);
                 if (want_targets) out["belief_targets"] = std::move(belief_targets);
                 if (want_marginals) {
                     out["belief_marginals"] = std::move(belief_marginals);
                     out["belief_allowed"] = std::move(belief_allowed);
                 }
                 return out;
             },
             py::arg("include_history") = true, py::arg("include_beliefs") = false,
             py::arg("include_belief_targets") = false, py::arg("include_belief_marginals") = false)
        .def("step_batch",
             [](BgBatchPlayEnv& self, py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> action_idx) {
                 const int B = self.numEnvs();
                 if (action_idx.size() != B) throw std::invalid_argument("step_batch: action_idx size mismatch");
                 py::array_t<std::uint8_t> done(B);
                 py::array_t<float> reward_t0(B);
                 const std::int32_t* act_ptr = action_idx.data();
                 std::uint8_t* d_ptr = done.mutable_data();
                 float* r_ptr = reward_t0.mutable_data();
                 {
                     py::gil_scoped_release release;
                     self.stepBatch(act_ptr, d_ptr, r_ptr);
                 }
                 return py::make_tuple(std::move(done), std::move(reward_t0));
             });

    py::class_<BgEvalSession>(m, "BgEvalSession")
        .def(py::init<int, std::uint64_t, const std::string&, int, const std::string&, bool>(),
             py::arg("num_deals"), py::arg("seed") = 12345, py::arg("opp") = "policy", py::arg("threads") = 12,
             py::arg("mode") = "all", py::arg("allow_double") = false)
        .def_property_readonly("num_pairs", &BgEvalSession::numPairs)
        .def_property_readonly("num_active", &BgEvalSession::numActive)
        .def_property_readonly("done", &BgEvalSession::done)
        .def("observe_active", &BgEvalSession::observeActive, py::arg("include_history") = true)
        .def("step_active", &BgEvalSession::stepActive)
        .def("summary", &BgEvalSession::summary);

    py::class_<BgFastPolicyModel>(m, "BgFastPolicyModel")
        .def(py::init<>())
        .def("load", &BgFastPolicyModel::load)
        .def("forward_state", [](const BgFastPolicyModel& self, py::array_t<float, py::array::c_style | py::array::forcecast> st) {
            if (st.size() != kBgStateDim) throw std::invalid_argument("state dim mismatch");
            py::array_t<float> out({32});
            self.forwardState(st.data(), out.mutable_data());
            return out;
        });

    py::class_<BgBeliefModel>(m, "BgBeliefModel")
        .def(py::init<>())
        .def("load", &BgBeliefModel::load)
        .def("forward_logits", [](const BgBeliefModel& self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> st,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> hist,
                                  int n_plies,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> marg) {
            py::array_t<float> out({32, 3});
            self.forwardLogits(st.data(), hist.data(), n_plies, marg.data(), out.mutable_data());
            return out;
        });

    py::class_<BgQGRUModel>(m, "BgQGRUModel")
        .def(py::init<>())
        .def("load", &BgQGRUModel::load)
        .def("forward_q", [](const BgQGRUModel& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> st,
                             py::array_t<float, py::array::c_style | py::array::forcecast> hist,
                             int n_plies,
                             py::array_t<float, py::array::c_style | py::array::forcecast> acts,
                             int n_legal) {
            py::array_t<float> out({n_legal});
            self.forwardQ(st.data(), hist.data(), n_plies, acts.data(), n_legal, out.mutable_data());
            return out;
        });

    m.attr("BID_VERSION") = kBgBidVersion;
    m.attr("BID_VERSION_UNMASKED") = kBgBidVersionUnmasked;
    m.attr("NUM_BIDS") = kBgNumBids;
    m.attr("BID_SUIT_DIM") = kBgBidSuitDim;
    m.attr("BID_GLOBAL_DIM") = kBgBidGlobalDim;
    m.attr("BID_STEP_DIM") = kBgBidStepDim;
    m.attr("MAX_BIDS") = kBgMaxBids;

    py::class_<BgBidModel>(m, "BgBidModel")
        .def(py::init<>())
        .def("load", &BgBidModel::load)
        .def("forward_q", [](const BgBidModel& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> sf,
                             py::array_t<float, py::array::c_style | py::array::forcecast> gf,
                             py::array_t<float, py::array::c_style | py::array::forcecast> bh,
                             int n_bids) {
            py::array_t<float> out({kBgNumBids});
            self.forwardQ(sf.data(), gf.data(), bh.data(), n_bids, out.mutable_data());
            return out;
        });

    py::class_<BgBatchBidEnv>(m, "BgBatchBidEnv")
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("fast_at"), py::arg("fast_suit"), py::arg("fast_nt"), py::arg("threads") = 12)
        .def("load_bid_model", &BgBatchBidEnv::loadBidModel)
        .def("generate_batch",
             [](const BgBatchBidEnv& self, int num_deals, std::uint64_t seed, double explore_eps, bool allow_double,
                bool use_natural_mask) {
                 CounterfactualBidBatch b;
                 {
                     py::gil_scoped_release release;
                     b = self.generateBatch(num_deals, seed, explore_eps, allow_double, use_natural_mask);
                 }
                 const py::ssize_t N = b.n_samples;
                 py::array_t<float> sf({N, py::ssize_t(4), py::ssize_t(kBgBidSuitDim)});
                 py::array_t<float> gf({N, py::ssize_t(kBgBidGlobalDim)});
                 py::array_t<float> bh({N, py::ssize_t(kBgMaxBids), py::ssize_t(kBgBidStepDim)});
                 py::array_t<std::int32_t> nb({N});
                 py::array_t<float> lm({N, py::ssize_t(kBgNumBids)});
                 py::array_t<float> tgt({N, py::ssize_t(kBgNumBids)});
                 if (N > 0) {
                     std::memcpy(sf.mutable_data(), b.suit_feats.data(), b.suit_feats.size() * sizeof(float));
                     std::memcpy(gf.mutable_data(), b.global_feats.data(), b.global_feats.size() * sizeof(float));
                     std::memcpy(bh.mutable_data(), b.bid_hist.data(), b.bid_hist.size() * sizeof(float));
                     std::memcpy(nb.mutable_data(), b.n_bids.data(), b.n_bids.size() * sizeof(std::int32_t));
                     std::memcpy(lm.mutable_data(), b.legal_mask.data(), b.legal_mask.size() * sizeof(float));
                     std::memcpy(tgt.mutable_data(), b.targets.data(), b.targets.size() * sizeof(float));
                 }
                 py::dict out;
                 out["n_samples"] = b.n_samples;
                 out["suit_feats"] = std::move(sf);
                 out["global_feats"] = std::move(gf);
                 out["bid_hist"] = std::move(bh);
                 out["n_bids"] = std::move(nb);
                 out["legal_mask"] = std::move(lm);
                 out["targets"] = std::move(tgt);
                 return out;
             },
             py::arg("num_deals"), py::arg("seed") = 1, py::arg("explore_eps") = 0.08,
             py::arg("allow_double") = false, py::arg("use_natural_mask") = true)
        .def("eval_vs_natural",
             [](const BgBatchBidEnv& self, int num_deals, std::uint64_t seed, int opp_at_bonus, bool mixed_partner,
                bool allow_double) {
                 BgBatchBidEnv::DuelSummary s;
                 {
                     py::gil_scoped_release release;
                     s = self.evalVsNatural(num_deals, seed, opp_at_bonus, mixed_partner, allow_double);
                 }
                 py::dict out;
                 out["mean"] = s.mean;
                 out["ci"] = s.ci;
                 out["deals"] = s.deals;
                 out["pct_suit"] = s.pct_suit;
                 out["pct_nt"] = s.pct_nt;
                 out["pct_at"] = s.pct_at;
                 out["pct_pass"] = s.pct_pass;
                 out["pct_dbl"] = s.pct_dbl;
                 out["pct_rdbl"] = s.pct_rdbl;
                 out["suit_has_j9"] = s.suit_has_j9;
                 out["nt_has_ace"] = s.nt_has_ace;
                 out["at_has_j"] = s.at_has_j;
                 return out;
             },
             py::arg("num_deals") = 2000, py::arg("seed") = 54321, py::arg("opp_at_bonus") = 0,
             py::arg("mixed_partner") = false, py::arg("allow_double") = false);
}
