// Mirrored-deal duels between two configurations of our Bulgarian card player
// (bg/agent.h), for fast tuning without the C# bridge. The default is natural
// bidding. The margin is A's game points per deal.
//
// Usage: bg_duel --a "<opts>" --b "<opts>" [--deals N] [--seed S] [--threads T]
// opts: "policy" or "w=24,exact=5,end=3,lik=1,auction=0,z=1".
// Append ",bid=simple" to either side to compare bidders on the same decks.

#include "bg/agent.h"
#include "bg/rule_policy.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace bg;

namespace {

struct Config {
    bool policy = false;
    bool qonly = false;
    bool fonly = false;
    bool natural = true;
    SearchOptions o;
    std::shared_ptr<BgFastPolicyModel> fast_model;
    std::shared_ptr<BgBeliefModel> belief_model;
    std::shared_ptr<BgQGRUModel> q_model;
};

Config parse(const std::string& s) {
    Config c;
    std::stringstream ss(s);
    for (std::string kv; std::getline(ss, kv, ',');) {
        if (kv == "policy") { c.policy = true; continue; }
        if (kv == "qonly") { c.qonly = true; continue; }
        if (kv == "fonly") { c.fonly = true; continue; }
        if (kv == "race") { c.o.race = true; continue; }
        const auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "w") c.o.worlds = std::stoi(v);
        else if (k == "exact") c.o.root_exact_trick = std::stoi(v);
        else if (k == "end") c.o.endgame_tricks = std::stoi(v);
        else if (k == "lik") c.o.lik_eps = std::stod(v);
        else if (k == "auction") c.o.auction_weight = std::stod(v);
        else if (k == "pool") c.o.pool = std::stoi(v);
        else if (k == "bid") {
            if (v != "natural" && v != "simple") throw std::invalid_argument("unknown bidder");
            c.natural = v == "natural";
        } else if (k == "z") c.o.z = std::stod(v);
        else if (k == "m") c.o.margin = std::stod(v);
        else if (k == "fast") {
            c.fast_model = std::make_shared<BgFastPolicyModel>();
            if (!c.fast_model->load(v)) throw std::runtime_error("failed to load fast model: " + v);
            c.o.fast_policy = c.fast_model.get();
        } else if (k == "belief") {
            c.belief_model = std::make_shared<BgBeliefModel>();
            if (!c.belief_model->load(v)) throw std::runtime_error("failed to load belief model: " + v);
            c.o.belief_model = c.belief_model.get();
        } else if (k == "btemp") c.o.belief_temp = std::stod(v);
        else if (k == "qgru") {
            c.q_model = std::make_shared<BgQGRUModel>();
            if (!c.q_model->load(v)) throw std::runtime_error("failed to load qgru model: " + v);
            c.o.q_model = c.q_model.get();
        } else if (k == "qblend") c.o.q_blend = std::stod(v);
        else if (k == "qtaper") c.o.q_taper_trick = std::stoi(v);
        else if (k == "flik") c.o.fast_lik_weight = std::stod(v);
        else if (k == "ftemp") c.o.fast_lik_temp = std::stod(v);
        else if (k == "race") c.o.race = (v != "0");
        else if (k == "rmin") c.o.race_min = std::stoi(v);
        else if (k == "rstep") c.o.race_step = std::stoi(v);
        else if (k == "rz") c.o.race_z = std::stod(v);
    }
    return c;
}

std::uint8_t simpleBid(const Round& r) {
    const Mask h = r.hand(r.toMove());
    const std::uint8_t avail = r.availableBids();
    double best_v = 0;
    std::uint8_t best = kPass;
    for (int s = 0; s < 4; ++s) {
        const std::uint8_t flag = std::uint8_t(1u << s);
        if (!(avail & flag)) continue;
        const Mask t = h & suitMask(s);
        double v = popcount(t) * 0.8;
        if (t & bit(cardOf(s, belote::kJ))) v += 1.6;
        if (t & bit(cardOf(s, belote::k9))) v += 1.0;
        for (int o = 0; o < 4; ++o)
            if (o != s && (h & bit(cardOf(o, belote::kA)))) v += 0.6;
        if (v >= 4.2 && v > best_v) best_v = v, best = flag;
    }
    if (avail & kAllTrumps) {
        double v = 0;
        for (int s = 0; s < 4; ++s) {
            if (h & bit(cardOf(s, belote::kJ))) v += 1.5;
            if (h & bit(cardOf(s, belote::k9))) v += 0.8;
        }
        if (v >= 4.0 && v + 0.5 > best_v) best_v = v + 0.5, best = kAllTrumps;
    }
    if (avail & kNoTrumps) {
        double v = 0;
        for (int s = 0; s < 4; ++s) {
            if (h & bit(cardOf(s, belote::kA))) v += 1.4;
            if (h & bit(cardOf(s, belote::k10))) v += 0.5;
        }
        if (v >= 4.4 && v > best_v) best_v = v, best = kNoTrumps;
    }
    return best;
}

int play(const std::array<int, kCards>& deck, int first, const Config& a, const Config& b, bool a_team0,
         std::uint64_t seed, double* ms_a, long* cards_a) {
    Round r(deck, first);
    while (r.decision() == Round::Decision::kBid) {
        const Config& bidder = ((teamOf(r.toMove()) == 0) == a_team0) ? a : b;
        r.bid(bidder.natural ? naturalBid(r) : simpleBid(r));
    }
    if (r.finished()) return 0;
    std::mt19937_64 rng(seed);
    while (!r.finished()) {
        if (r.decision() == Round::Decision::kAnnounce) {
            r.declare(chooseDeclarations(r.offeredAnnounces()));
            continue;
        }
        const int seat = r.toMove();
        const bool is_a = (teamOf(seat) == 0) == a_team0;
        const Config& c = is_a ? a : b;
        int card;
        const auto t0 = std::chrono::steady_clock::now();
        if (popcount(r.legalMoves()) == 1) {
            card = lowestCard(r.legalMoves());
        } else if (c.qonly && c.q_model) {
            card = c.q_model->chooseCard(publicState(r, seat));
        } else if (c.fonly && c.fast_model) {
            card = c.fast_model->chooseCard(publicState(r, seat));
        } else if (c.policy) {
            card = rulePolicy(r);
        } else {
            SearchOptions o = c.o;
            o.seed = rng() | 1;
            card = searchCard(publicState(r, seat), o).card;
        }
        if (is_a) {
            *ms_a += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++*cards_a;
        }
        r.play(card, true);
    }
    const int m = r.result().sn - r.result().ew;
    return a_team0 ? m : -m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string sa = "w=24", sb = "policy";
    int deals = 1000, threads = 12;
    std::uint64_t seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--a")) sa = argv[i + 1];
        else if (!std::strcmp(argv[i], "--b")) sb = argv[i + 1];
        else if (!std::strcmp(argv[i], "--deals")) deals = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--seed")) seed = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
    }
    const Config a = parse(sa), b = parse(sb);
    std::vector<double> pairs(std::size_t(deals), 0.0);
    std::vector<char> played(std::size_t(deals), 0);
    std::atomic<int> next{0};
    std::mutex mu;
    double ms = 0;
    long cards = 0;
    auto worker = [&]() {
        double my_ms = 0;
        long my_cards = 0;
        for (int i; (i = next.fetch_add(1)) < deals;) {
            std::mt19937_64 rng(seed * 1000003ull + std::uint64_t(i));
            std::array<int, kCards> deck;
            std::iota(deck.begin(), deck.end(), 0);
            std::shuffle(deck.begin(), deck.end(), rng);
            if (a.natural == b.natural) {
                Round probe(deck, i % 4);
                while (probe.decision() == Round::Decision::kBid)
                    probe.bid(a.natural ? naturalBid(probe) : simpleBid(probe));
                if (probe.finished()) continue;
            }
            const std::uint64_t s = rng();
            const int x = play(deck, i % 4, a, b, true, s, &my_ms, &my_cards);
            const int y = play(deck, i % 4, a, b, false, s, &my_ms, &my_cards);
            pairs[std::size_t(i)] = 0.5 * (x + y);
            played[std::size_t(i)] = 1;
        }
        std::lock_guard<std::mutex> lock(mu);
        ms += my_ms;
        cards += my_cards;
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    std::vector<double> v;
    for (int i = 0; i < deals; ++i)
        if (played[std::size_t(i)]) v.push_back(pairs[std::size_t(i)]);
    double mean = 0, var = 0;
    for (double x : v) mean += x / double(v.size());
    for (double x : v) var += (x - mean) * (x - mean) / double(v.size() - 1);
    std::printf("A[%s] vs B[%s]: %+.3f +- %.3f game points/deal over %zu mirrored deals, A %.5f ms/card\n", sa.c_str(),
                sb.c_str(), mean, 1.96 * std::sqrt(var / double(v.size())), v.size(), ms / std::max(1L, cards));
    return 0;
}
