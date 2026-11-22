// Duplicate-deal & full-match (101) evaluation harness (agents/duel.h).
//
//   belote_duel --a rollout:32 --b rule --deals 500 --threads 8
//   belote_duel --mode bid --a rule --b random --player rule --deals 500
//   belote_duel --mode mixed --a rollout:32 --b rule --partner rule --deals 300
//   belote_duel --mode match --a rollout:32 --b rule --matches 200 --threads 8
//
// Agents: random, rule, ismcts:<iterations>[:<threads>], rollout:<worlds>[:ab<trick>].

#include "agents/duel.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace belote;

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: belote_duel [--mode play|bid|deal|mixed|match] [--a SPEC] [--b SPEC]\n"
                 "                   [--bidder SPEC|self] [--player SPEC] [--partner SPEC]\n"
                 "                   [--deals N] [--matches N] [--seed S] [--threads T]\n"
                 "                   [--rules belgian|wikibooks|tanguy]\n"
                 "SPEC: random | rule | ismcts:<iterations>[:<threads>] | rollout:<worlds>[:ab<trick>]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = "play";
    std::string a = "ismcts:1000";
    std::string b = "rule";
    std::string bidder = "";
    std::string player = "";
    std::string partner = "";
    int deals = 1000;
    int matches = 200;
    std::uint64_t seed = 1;
    int threads = int(std::thread::hardware_concurrency());
    Config config = Config::belgian();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--mode") mode = value();
        else if (arg == "--a") a = value();
        else if (arg == "--b") b = value();
        else if (arg == "--bidder") bidder = value();
        else if (arg == "--player") player = value();
        else if (arg == "--partner") partner = value();
        else if (arg == "--deals") deals = std::stoi(value());
        else if (arg == "--matches") matches = std::stoi(value());
        else if (arg == "--seed") seed = std::stoull(value());
        else if (arg == "--threads") threads = std::stoi(value());
        else if (arg == "--rules") {
            const std::string r = value();
            if (r == "belgian") config = Config::belgian();
            else if (r == "wikibooks") config = Config::wikibooks();
            else if (r == "tanguy") config = Config::tanguyCompat();
            else { usage(); return 2; }
        } else {
            usage();
            return arg == "--help" || arg == "-h" ? 0 : 2;
        }
    }

    if (mode == "match") {
        MatchDuelOptions mo;
        mo.a = a;
        mo.b = b;
        mo.bidder = bidder.empty() ? "self" : bidder;
        mo.player = player;
        mo.partner = partner;
        mo.matches = matches;
        mo.seed = seed;
        mo.threads = threads;
        mo.config = config;
        const MatchDuelResult r = runMatchDuel(mo);
        std::printf("[config: %s]\n", r.config_summary.c_str());
        std::printf("MATCH %s vs %s (bidder=%s%s%s, %d pairs x 2, seed %llu)\n", mo.a.c_str(), mo.b.c_str(),
                    mo.bidder.c_str(), mo.player.empty() ? "" : (", player=" + mo.player).c_str(),
                    mo.partner.empty() ? "" : (", partner=" + mo.partner).c_str(), r.pairs,
                    (unsigned long long)mo.seed);
        std::printf("  A win rate: %.2f%% +- %.2f pp (95%% CI), score margin: %+.2f +- %.2f units/match\n",
                    100.0 * r.win_rate, 196.0 * r.win_rate_se, r.score_margin, 1.96 * r.score_margin_se);
        std::printf("  deals/match: %.1f, time: A %.2f ms/card, B %.2f ms/card, %.1f s total\n", r.mean_deals,
                    r.ms_per_card_a, r.ms_per_card_b, r.seconds);
        return 0;
    }

    DuelOptions o;
    o.a = a;
    o.b = b;
    o.deals = deals;
    o.seed = seed;
    o.threads = threads;
    o.config = config;
    if (mode == "play") {
        o.bidder = bidder.empty() ? "rule" : bidder;
        o.player = player;
        o.partner = partner;
    } else if (mode == "bid") {
        o.bidder = bidder.empty() ? "self" : bidder;
        o.player = player.empty() ? "rule" : player;
        o.partner = partner;
    } else if (mode == "deal") {
        o.bidder = bidder.empty() ? "self" : bidder;
        o.player = player;
        o.partner = partner;
    } else if (mode == "mixed") {
        o.bidder = bidder.empty() ? "rule" : bidder;
        o.player = player;
        o.partner = partner.empty() ? "rule" : partner;
    } else {
        usage();
        return 2;
    }

    const DuelResult r = runDuel(o);
    std::printf("[config: %s]\n", r.config_summary.c_str());
    std::printf("%s vs %s (mode=%s, bidder=%s%s%s, %d decks x 2, %d both-pass skipped, seed %llu)\n", o.a.c_str(),
                o.b.c_str(), mode.c_str(), o.bidder.c_str(),
                o.player.empty() ? "" : (", player=" + o.player).c_str(),
                o.partner.empty() ? "" : (", partner=" + o.partner).c_str(), r.decks, r.skipped,
                (unsigned long long)o.seed);
    std::printf("  A margin: %+.3f +- %.3f units/deal (95%% CI), %+.2f +- %.2f raw points/deal\n", r.units,
                1.96 * r.units_se, r.raw, 1.96 * r.raw_se);
    std::printf("  time: A %.2f ms/card, B %.2f ms/card, %.1f s total\n", r.ms_per_card_a, r.ms_per_card_b,
                r.seconds);
    return 0;
}
