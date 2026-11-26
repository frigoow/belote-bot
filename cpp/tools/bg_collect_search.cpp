// Multithreaded C++ Search-Distillation Collector & FastPolicy Mirrored Dueler
// for Bulgarian Bridge-Belot (Expert Iteration #1).
//
// Mode 1 (collect): plays deals matching --mode suit|at|nt using full searchCard()
//   (no racing, full-support CRN search Q-values for every legal card) and writes
//   binary SearchSample records for `python/bg/train_search_fast.py`.
// Mode 2 (duel): plays mirrored pairs comparing two BgFastPolicyModel binaries
//   head-to-head (pure 1-ply) in < 2 seconds across 14 threads.

#include "bg/agent.h"
#include "bg/bg.h"
#include "bg/bid_model.h"
#include "bg/encoder.h"
#include "bg/neural.h"
#include "bg/rule_policy.h"
#include "bg/view.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace bg;

struct Sample {
    float state[kBgStateDim];
    std::uint32_t legal_mask;
    std::uint8_t trick;
    std::uint8_t contract_fam;  // 0=suit, 1=nt, 2=at
    std::uint8_t mult;          // 1, 2, 4
    std::uint8_t pad;
    float q_canon[kCards];
};
static_assert(sizeof(Sample) == kBgStateDim * 4 + 4 + 4 + kCards * 4, "Sample layout mismatch");

enum class Mode { kAll, kSuit, kNoTrumps, kAllTrumps };

int contractFamily(std::uint8_t c) {
    if (c & kNoTrumps) return 1;
    if (c & kAllTrumps) return 2;
    return 0;
}

int contractMult(std::uint8_t c) {
    if (c & kReDouble) return 4;
    if (c & kDouble) return 2;
    return 1;
}

Round makeDeal(std::mt19937_64& rng, Mode mode, const BgBidModel* bid_model, double bid_model_prob, double bid_noise,
               bool allow_double) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    for (int attempt = 0; attempt < 500; ++attempt) {
        std::shuffle(deck.begin(), deck.end(), rng);
        const int first = int(rng() % kPlayers);
        Round r(deck, first, 0);
        const bool use_bid_net = bid_model && bid_model->loaded() && (u(rng) < bid_model_prob);
        int at_bonus = 0, nt_bonus = 0;
        if (!use_bid_net) {
            if (mode == Mode::kAll) {
                at_bonus = (rng() & 1u) ? 8 : 0;
            } else if (mode == Mode::kAllTrumps) {
                at_bonus = 10;
            } else if (mode == Mode::kNoTrumps) {
                at_bonus = -10;
                nt_bonus = 6;
            }
        }
        while (r.decision() == Round::Decision::kBid) {
            std::uint8_t bid = use_bid_net ? bid_model->chooseBid(r, allow_double, 0.0)
                                           : naturalBidTuned(r, at_bonus, nt_bonus, allow_double);
            if (bid_noise > 0.0 && u(rng) < bid_noise) {
                const std::uint8_t avail = r.availableBids();
                std::uint8_t opts[9];
                int n = 0;
                opts[n++] = kPass;
                for (int b = 0; b < 8; ++b) {
                    const std::uint8_t f = std::uint8_t(1u << b);
                    if (!allow_double && (f == kDouble || f == kReDouble)) continue;
                    if (avail & f) opts[n++] = f;
                }
                if (n > 1) bid = opts[rng() % std::uint64_t(n)];
            }
            r.bid(bid);
        }
        if (r.finished()) continue;
        const std::uint8_t c = r.contract();
        if (mode == Mode::kSuit && !(c & 15u)) continue;
        if (mode == Mode::kNoTrumps && !(c & kNoTrumps)) continue;
        if (mode == Mode::kAllTrumps && !(c & kAllTrumps)) continue;
        while (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
        }
        if (r.decision() == Round::Decision::kPlay) return r;
    }
    throw std::runtime_error("bg_collect_search: failed to generate matching deal");
}

int pickMove(const SearchResult& sr, std::mt19937_64& rng, double explore_prob, double explore_temp, double mult) {
    if (explore_prob <= 0.0 || sr.values.size() <= 1) return sr.card;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) >= explore_prob) return sr.card;
    double mx = -1e30;
    for (const auto& [c, v] : sr.values) mx = std::max(mx, v / mult);
    double sum = 0.0;
    std::vector<double> w(sr.values.size());
    for (std::size_t i = 0; i < sr.values.size(); ++i) {
        w[i] = std::exp((sr.values[i].second / mult - mx) / explore_temp);
        sum += w[i];
    }
    double r = u(rng) * sum;
    for (std::size_t i = 0; i < sr.values.size(); ++i) {
        r -= w[i];
        if (r <= 0.0) return sr.values[i].first;
    }
    return sr.values.back().first;
}

int playDealDuel(Round r, const BgFastPolicyModel& t0_model, const BgFastPolicyModel& t1_model) {
    while (!r.finished()) {
        if (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
            continue;
        }
        const int seat = r.toMove();
        const int c = (teamOf(seat) == 0 ? t0_model : t1_model).chooseCard(r);
        r.play(c, true);
    }
    return r.result().sn - r.result().ew;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode_str = "suit";
    std::string out_path;
    std::string bid_path = "checkpoints/bg_models_bid_unmasked_dbl/bg_bid.bin";
    std::string fast_at_path, belief_at_path, qgru_at_path;
    std::string fast_suit_path, belief_suit_path, qgru_suit_path;
    std::string fast_nt_path, belief_nt_path, qgru_nt_path;
    std::string duel_a_path, duel_b_path;

    int deals = 20000;
    int threads = 14;
    int worlds = 64;
    int exact_trick = 4;
    int endgame_tricks = 3;
    int max_trick = 7;
    double bid_model_prob = 0.65;
    double bid_noise = 0.02;
    double explore = 0.03;
    double explore_temp = 2.0;
    bool allow_double = true;
    bool raw_tiebreak = true;
    bool ply0_only = false;
    std::uint64_t seed = 9001;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mode") mode_str = need("--mode");
        else if (a == "--out") out_path = need("--out");
        else if (a == "--bid") bid_path = need("--bid");
        else if (a == "--fast-at") fast_at_path = need("--fast-at");
        else if (a == "--belief-at") belief_at_path = need("--belief-at");
        else if (a == "--qgru-at") qgru_at_path = need("--qgru-at");
        else if (a == "--fast-suit") fast_suit_path = need("--fast-suit");
        else if (a == "--belief-suit") belief_suit_path = need("--belief-suit");
        else if (a == "--qgru-suit") qgru_suit_path = need("--qgru-suit");
        else if (a == "--fast-nt") fast_nt_path = need("--fast-nt");
        else if (a == "--belief-nt") belief_nt_path = need("--belief-nt");
        else if (a == "--qgru-nt") qgru_nt_path = need("--qgru-nt");
        else if (a == "--duel-a") duel_a_path = need("--duel-a");
        else if (a == "--duel-b") duel_b_path = need("--duel-b");
        else if (a == "--deals") deals = std::stoi(need("--deals"));
        else if (a == "--threads") threads = std::stoi(need("--threads"));
        else if (a == "--worlds") worlds = std::stoi(need("--worlds"));
        else if (a == "--exact-trick") exact_trick = std::stoi(need("--exact-trick"));
        else if (a == "--endgame") endgame_tricks = std::stoi(need("--endgame"));
        else if (a == "--max-trick") max_trick = std::stoi(need("--max-trick"));
        else if (a == "--bid-prob") bid_model_prob = std::stod(need("--bid-prob"));
        else if (a == "--bid-noise") bid_noise = std::stod(need("--bid-noise"));
        else if (a == "--explore") explore = std::stod(need("--explore"));
        else if (a == "--seed") seed = std::stoull(need("--seed"));
        else if (a == "--no-raw-tiebreak") raw_tiebreak = false;
        else if (a == "--no-double") allow_double = false;
        else if (a == "--ply0-only") ply0_only = true;
        else {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 2;
        }
    }

    Mode mode = Mode::kAll;
    if (mode_str == "suit") mode = Mode::kSuit;
    else if (mode_str == "nt") mode = Mode::kNoTrumps;
    else if (mode_str == "at") mode = Mode::kAllTrumps;
    else if (mode_str != "all") {
        std::fprintf(stderr, "bad --mode %s\n", mode_str.c_str());
        return 2;
    }

    BgBidModel bid_model;
    if (!bid_path.empty() && !bid_model.load(bid_path)) {
        std::fprintf(stderr, "failed to load --bid %s\n", bid_path.c_str());
        return 1;
    }

    // ---- FastPolicy head-to-head mirrored duel mode ----
    if (!duel_a_path.empty() && !duel_b_path.empty()) {
        BgFastPolicyModel fa, fb;
        if (!fa.load(duel_a_path) || !fb.load(duel_b_path)) {
            std::fprintf(stderr, "failed to load duel fast policies\n");
            return 1;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<double> diffs(std::size_t(deals), 0.0);
        std::atomic<int> next_i{0};
        auto worker = [&]() {
            while (true) {
                const int i = next_i.fetch_add(1);
                if (i >= deals) break;
                std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(i + 1) * 0xBF58476D1CE4E5B9ull);
                const Round base = makeDeal(rng, mode, &bid_model, bid_model_prob, 0.0, allow_double);
                const int m1 = playDealDuel(base, fa, fb);   // A is team 0
                const int m2 = -playDealDuel(base, fb, fa);  // A is team 1
                diffs[std::size_t(i)] = 0.5 * double(m1 + m2);
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();

        double sum = 0.0, sum2 = 0.0;
        int wins = 0, ties = 0;
        for (double d : diffs) {
            sum += d;
            sum2 += d * d;
            if (d > 0.0) ++wins;
            else if (d == 0.0) ++ties;
        }
        const double mean = sum / double(deals);
        const double var = std::max(0.0, sum2 / double(deals) - mean * mean);
        const double ci95 = 1.96 * std::sqrt(var / double(deals));
        const double wr = 100.0 * (double(wins) + 0.5 * double(ties)) / double(deals);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("FastPolicy Duel [%s] A vs B (%d mirrored pairs, %.2fs): %+.3f +- %.3f game pts/deal | pair-win %.2f%%\n",
                    mode_str.c_str(), deals, secs, mean, ci95, wr);
        return 0;
    }

    if (out_path.empty()) {
        std::fprintf(stderr, "must specify --out <file.bin> (or --duel-a / --duel-b)\n");
        return 2;
    }

    BgFastPolicyModel fast_at, fast_suit, fast_nt;
    BgBeliefModel belief_at, belief_suit, belief_nt;
    BgQGRUModel qgru_at, qgru_suit, qgru_nt;

    auto loadOpt = [](auto& m, const std::string& p, const char* name) {
        if (p.empty()) return true;
        if (!m.load(p)) {
            std::fprintf(stderr, "failed to load %s: %s\n", name, p.c_str());
            return false;
        }
        return true;
    };
    if (!loadOpt(fast_at, fast_at_path, "--fast-at") || !loadOpt(belief_at, belief_at_path, "--belief-at") ||
        !loadOpt(qgru_at, qgru_at_path, "--qgru-at") || !loadOpt(fast_suit, fast_suit_path, "--fast-suit") ||
        !loadOpt(belief_suit, belief_suit_path, "--belief-suit") || !loadOpt(qgru_suit, qgru_suit_path, "--qgru-suit") ||
        !loadOpt(fast_nt, fast_nt_path, "--fast-nt") || !loadOpt(belief_nt, belief_nt_path, "--belief-nt") ||
        !loadOpt(qgru_nt, qgru_nt_path, "--qgru-nt")) {
        return 1;
    }

    SearchOptions base_opts;
    base_opts.worlds = worlds;
    base_opts.endgame_tricks = endgame_tricks;
    base_opts.root_exact_trick = exact_trick;
    base_opts.z = -1.0;
    base_opts.lik_eps = 1.0;
    base_opts.auction_weight = 1.0;
    base_opts.fast_lik_weight = 0.3;
    base_opts.fast_lik_temp = 1.0;
    base_opts.q_blend = 0.25;
    base_opts.q_taper_trick = 5;
    base_opts.race = false;  // full-support CRN values for EVERY legal card!
    base_opts.raw_tiebreak = raw_tiebreak;
    if (fast_at.loaded()) base_opts.at_models.fast = &fast_at;
    if (belief_at.loaded()) base_opts.at_models.belief = &belief_at;
    if (qgru_at.loaded()) base_opts.at_models.q = &qgru_at;
    if (fast_suit.loaded()) base_opts.suit_models.fast = &fast_suit;
    if (belief_suit.loaded()) base_opts.suit_models.belief = &belief_suit;
    if (qgru_suit.loaded()) base_opts.suit_models.q = &qgru_suit;
    if (fast_nt.loaded()) base_opts.nt_models.fast = &fast_nt;
    if (belief_nt.loaded()) base_opts.nt_models.belief = &belief_nt;
    if (qgru_nt.loaded()) base_opts.nt_models.q = &qgru_nt;

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "failed to open output %s\n", out_path.c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> next_deal{0};
    std::atomic<int> done_deals{0};
    std::atomic<std::uint64_t> total_samples{0};
    std::mutex io_mtx;

    auto worker = [&](int tid) {
        std::vector<Sample> buf;
        buf.reserve(512);
        SearchOptions opts = base_opts;
        while (true) {
            const int di = next_deal.fetch_add(1);
            if (di >= deals) break;
            std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(di + 1) * 0xBF58476D1CE4E5B9ull +
                                std::uint64_t(tid) * 0x94D049BB133111EBull);
            Round r = makeDeal(rng, mode, &bid_model, bid_model_prob, bid_noise, allow_double);
            const std::uint8_t c = r.contract();
            const std::uint8_t fam = std::uint8_t(contractFamily(c));
            const int mult = contractMult(c);
            const double inv_mult = 1.0 / double(mult);

            while (!r.finished()) {
                if (r.decision() == Round::Decision::kAnnounce) {
                    r.declare(chooseDeclarations(r.offeredAnnounces()));
                    continue;
                }
                const Mask legal = r.legalMoves();
                if (popcount(legal) <= 1) {
                    r.play(lowestCard(legal), true);
                    continue;
                }
                const int seat = r.toMove();
                const int trick = r.trickNumber();
                const PublicState ps = publicState(r, seat);
                opts.seed = rng() | 1ull;
                const SearchResult sr = searchCard(ps, opts);

                if (trick <= max_trick && sr.values.size() > 1) {
                    Sample s{};
                    encodeState(ps, s.state);
                    s.trick = std::uint8_t(trick);
                    s.contract_fam = fam;
                    s.mult = std::uint8_t(mult);
                    s.pad = 0;
                    s.legal_mask = 0;
                    for (int k = 0; k < kCards; ++k) s.q_canon[k] = -1e9f;
                    for (const auto& [raw_c, val] : sr.values) {
                        const int cc = canonCard(raw_c, c);
                        s.legal_mask |= (1u << cc);
                        s.q_canon[cc] = float(val * inv_mult);
                    }
                    buf.push_back(s);
                }
                if (ply0_only) break;

                const int chosen = pickMove(sr, rng, explore, explore_temp, double(mult));
                r.play(chosen, true);
            }

            const int finished = done_deals.fetch_add(1) + 1;
            if (buf.size() >= 256 || finished == deals) {
                std::lock_guard<std::mutex> lk(io_mtx);
                if (!buf.empty()) {
                    out.write(reinterpret_cast<const char*>(buf.data()),
                              std::streamsize(buf.size() * sizeof(Sample)));
                    total_samples.fetch_add(buf.size());
                    buf.clear();
                }
            }
            if (finished % 500 == 0 || finished == deals) {
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double dps = double(finished) / std::max(secs, 1e-6);
                const double eta = double(deals - finished) / std::max(dps, 1e-6);
                std::lock_guard<std::mutex> lk(io_mtx);
                std::printf("[%6d/%d] %.1f deals/s | samples=%llu | elapsed=%.0fs | eta=%.0fs\n", finished, deals, dps,
                            static_cast<unsigned long long>(total_samples.load()), secs, eta);
                std::fflush(stdout);
            }
        }
        if (!buf.empty()) {
            std::lock_guard<std::mutex> lk(io_mtx);
            out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size() * sizeof(Sample)));
            total_samples.fetch_add(buf.size());
        }
    };

    std::printf("Collecting %d %s deals (worlds=%d, exact_trick=%d, endgame=%d, raw_tb=%d, threads=%d) -> %s\n", deals,
                mode_str.c_str(), worlds, exact_trick, endgame_tricks, raw_tiebreak ? 1 : 0, threads, out_path.c_str());
    std::fflush(stdout);

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    out.flush();

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Done! %d deals -> %llu search samples in %.1fs (%.1f samples/s) saved to %s\n", deals,
                static_cast<unsigned long long>(total_samples.load()), secs,
                double(total_samples.load()) / std::max(secs, 1e-6), out_path.c_str());
    return 0;
}
