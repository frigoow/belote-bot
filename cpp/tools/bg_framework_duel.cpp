// Bulgarian Belot through cardbot-framework: duplicate-deal duels between the
// framework's agents (random, ISMCTS) and Belot's own (rule, search), one deal
// per game, payoff = the deal's game-point margin.
//
//   bg_framework_duel --a ismcts:1000 --b random --pairs 200 --threads 4
// Agents: random | ismcts:N | rule | search:W  (rule and search bid naturally
// and declare everything; search plays searchCard on W worlds, no nets).
#include "bg/framework.h"
#include "bg/rule_policy.h"
#include "cardbot/agents.h"
#include "cardbot/eval.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace bg;
using G = FrameworkGame;

namespace {

int belotBid(const G& g) { return G::kBid + bidToIndex(naturalBid(g.round())); }

int declareAll(const G& g) { return G::kDeclare + (1 << int(g.round().offeredAnnounces().size())) - 1; }

cardbot::Agent<G> make(const std::string& spec) {
    if (spec == "random") return cardbot::randomAgent<G>();
    if (spec.rfind("ismcts:", 0) == 0) {
        cardbot::IsmctsOptions o;
        o.iterations = std::size_t(std::atoi(spec.c_str() + 7));
        o.scale = 1.0 / 30.0;
        return cardbot::ismctsAgent<G>(o);
    }
    if (spec == "rule") {
        return [](const G& g, std::mt19937_64& rng) -> int {
            const Round& r = g.round();
            if (r.decision() == Round::Decision::kBid) return belotBid(g);
            if (r.decision() == Round::Decision::kAnnounce) return declareAll(g);
            return rulePolicy(g.determinize(r.toMove(), rng).round());
        };
    }
    if (spec.rfind("search:", 0) == 0) {
        const int worlds = std::atoi(spec.c_str() + 7);
        return [worlds](const G& g, std::mt19937_64& rng) -> int {
            const Round& r = g.round();
            if (r.decision() == Round::Decision::kBid) return belotBid(g);
            if (r.decision() == Round::Decision::kAnnounce) return declareAll(g);
            SearchOptions o;
            o.worlds = worlds;
            o.seed = rng() | 1;
            return searchCard(publicState(r, r.toMove()), o).card;
        };
    }
    std::fprintf(stderr, "unknown agent %s\n", spec.c_str());
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::string a = "ismcts:1000", b = "random";
    int pairs = 200;
    unsigned threads = 4;
    std::uint64_t seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--a") a = v;
        else if (k == "--b") b = v;
        else if (k == "--pairs") pairs = std::atoi(v.c_str());
        else if (k == "--threads") threads = unsigned(std::atoi(v.c_str()));
        else if (k == "--seed") seed = std::strtoull(v.c_str(), nullptr, 10);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::function<G(std::uint64_t)> deal = [](std::uint64_t s) { return G::deal(s); };
    const auto r = cardbot::duel<G>(deal, make(a), make(b), pairs, seed, threads);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s vs %s: %d pairs, %+.2f +- %.2f game points per deal (A better on %.1f%% of pairs), %.0fs\n",
                a.c_str(), b.c_str(), r.pairs, r.mean, r.ci95, 100 * r.a_better, s);
    return 0;
}
