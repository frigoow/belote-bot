// Reviews the rule policy (agents/rule_agent.h) the way the Manille rule
// agent was tuned: plays deals with the rule bidder and rule2 at every seat,
// scores every legal card at every real decision with a shared-world
// play-out (rule2 continuation, exact AB late), and reports where the rule
// card loses value -- by reason, role, seat in the trick and trick -- plus
// the worst decisions for reading.
//
// Usage: rule_review [--deals N] [--worlds W] [--ab T] [--seed S] [--threads K] [--show N]

#include "agents/agent.h"
#include "agents/rule_agent.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace belote;

namespace {

struct Decision {
    RuleReason reason;
    int role;  // 0 taker, 1 taker's partner, 2 defender
    int seat;  // position in the trick 0..3
    int trick;
    double regret;
    std::string text;
};

const char* kRoles[3] = {"taker", "taker's partner", "defender"};

std::string describe(const Deal& d, const RulePlayResult& r) {
    const int me = d.currentPlayer();
    std::string s;
    char buf[256];
    std::snprintf(buf, sizeof buf, "trick %d, seat %d (%s), trump %s, taker %d, points %d-%d\n", d.tricksPlayed() + 1,
                  me, teamOf(me) == d.takers() ? (me == d.taker() ? "taker" : "taker's partner") : "defender",
                  cardName(cardOf(d.trump(), kJ)).substr(1).c_str(), d.taker(), d.cardPoints()[teamOf(me)],
                  d.cardPoints()[1 - teamOf(me)]);
    s += buf;
    for (int i = 0; i < d.tricksPlayed(); ++i) {
        const Trick& t = d.completedTrick(i);
        s += "    ";
        for (int k = 0; k < t.n; ++k) s += std::to_string(t.players[k]) + ":" + cardName(t.cards[k]) + " ";
        s += "-> " + std::to_string(d.trickWinner(i)) + "\n";
    }
    s += "    trick: ";
    for (int k = 0; k < d.trick().n; ++k) s += std::to_string(d.trick().players[k]) + ":" + cardName(d.trick().cards[k]) + " ";
    s += "\n    hand: " + maskString(d.hand(me)) + "\n    values:";
    for (const auto& [c, v] : r.move_values) {
        std::snprintf(buf, sizeof buf, " %s%s=%+.2f", cardName(c).c_str(), c == r.rule_move ? "*" : "", v);
        s += buf;
    }
    s += std::string("\n    rule: ") + ruleReasonName(r.reason) + "\n";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int deals = 400, worlds = 96, ab = 3, threads = 8, show = 12;
    std::uint64_t seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--deals")) deals = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--worlds")) worlds = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--ab")) ab = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--seed")) seed = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--show")) show = std::atoi(argv[i + 1]);
    }

    std::vector<Decision> all;
    std::mutex mu;
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<Decision> mine;
        for (int i; (i = next.fetch_add(1)) < deals;) {
            std::mt19937_64 rng(seed * 1000003ull + std::uint64_t(i));
            std::array<int, kCards> deck;
            std::iota(deck.begin(), deck.end(), 0);
            std::shuffle(deck.begin(), deck.end(), rng);
            Deal d(Config::belgian(), i % kPlayers, deck);
            while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) d.bid(ruleBid(d));
            while (d.phase() == Phase::kPlay) {
                if (popcount(d.legalMoves()) > 1) {
                    RulePlayOptions o;
                    o.worlds = worlds;
                    o.ab_handoff_trick = ab;
                    o.z = -1;
                    o.seed = rng() | 1;
                    const RulePlayResult r = rulePlaySearch(d, o);
                    double best = -1e9, rule_v = 0;
                    for (const auto& [c, v] : r.move_values) {
                        best = std::max(best, v);
                        if (c == r.rule_move) rule_v = v;
                    }
                    const int me = d.currentPlayer();
                    Decision x;
                    x.reason = r.reason;
                    x.role = teamOf(me) != d.takers() ? 2 : (me == d.taker() ? 0 : 1);
                    x.seat = d.trick().n;
                    x.trick = d.tricksPlayed();
                    x.regret = best - rule_v;
                    if (x.regret > 0.5) x.text = describe(d, r);
                    mine.push_back(std::move(x));
                }
                d.play(rule2ChooseCard(d));
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        all.insert(all.end(), mine.begin(), mine.end());
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    double total = 0;
    for (const auto& x : all) total += x.regret;
    std::printf("%zu decisions over %d deals, mean regret %.3f units/decision, %.3f units/deal (all seats)\n",
                all.size(), deals, total / double(all.size()), total / deals);

    auto table = [&](const char* title, auto key, auto name, int n) {
        std::printf("\n%-36s %7s %8s %8s %7s\n", title, "n", "mean", "units/dl", "share");
        for (int k = 0; k < n; ++k) {
            double s = 0;
            int cnt = 0;
            for (const auto& x : all)
                if (key(x) == k) {
                    s += x.regret;
                    ++cnt;
                }
            if (!cnt) continue;
            std::printf("%-36s %7d %8.3f %8.3f %6.1f%%\n", name(k).c_str(), cnt, s / cnt, s / deals, 100 * s / total);
        }
    };
    table("reason", [](const Decision& x) { return int(x.reason); },
          [](int k) { return std::string(ruleReasonName(RuleReason(k))); }, int(RuleReason::kCount));
    table("role", [](const Decision& x) { return x.role; }, [](int k) { return std::string(kRoles[k]); }, 3);
    table("seat in trick", [](const Decision& x) { return x.seat; }, [](int k) { return "seat " + std::to_string(k); }, 4);
    table("trick", [](const Decision& x) { return x.trick; }, [](int k) { return "trick " + std::to_string(k + 1); }, 8);

    std::sort(all.begin(), all.end(), [](const Decision& a, const Decision& b) { return a.regret > b.regret; });
    std::printf("\nworst decisions:\n");
    for (int i = 0; i < show && i < int(all.size()); ++i)
        std::printf("[%d] regret %.2f, %s\n", i + 1, all[std::size_t(i)].regret, all[std::size_t(i)].text.c_str());
    return 0;
}
