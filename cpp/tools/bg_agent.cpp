// Our Bulgarian Belot card player as a process, for the bridge into Kostov's
// engine (bg/bridge). Stateless: every request carries the public record.
//
// Request, one line per card decision:
//   PLAY <first> <me> <hand mask> <contract BidType> <declarer>
//        <n> (<seat> <bid>)*          every bid, forced passes included
//        <n> (<seat> <type> <card>)*  non-belot declarations (card -1 unless mine)
//        <n> (<seat> <card> <0|1>)*   every card played, belote flag
//        <sn> <ew> <hanging>
// Reply: <card> <belote 0|1>
// "QUIT" ends the process.
//
// Options: --worlds N  --exact-trick T (root solved exactly from trick T)
//          --endgame K (play-outs exact for the last K tricks)  --lik EPS
//          --z Z  --policy (the rule policy alone)  --seed S

#include "bg/agent.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace bg;

int main(int argc, char** argv) {
    SearchOptions o;
    bool policy_only = false;
    bool qgru_only = false;
    bool fast_only = false;
    std::uint64_t seed = 0;
    std::string fast_path, belief_path, qgru_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--worlds") o.worlds = std::stoi(next());
        else if (a == "--exact-trick") o.root_exact_trick = std::stoi(next());
        else if (a == "--endgame") o.endgame_tricks = std::stoi(next());
        else if (a == "--lik") o.lik_eps = std::stod(next());
        else if (a == "--auction") o.auction_weight = std::stod(next());
        else if (a == "--pool") o.pool = std::stoi(next());
        else if (a == "--z") o.z = std::stod(next());
        else if (a == "--margin") o.margin = std::stod(next());
        else if (a == "--fast") fast_path = next();
        else if (a == "--belief") belief_path = next();
        else if (a == "--belief-temp") o.belief_temp = std::stod(next());
        else if (a == "--qgru") qgru_path = next();
        else if (a == "--q-blend") o.q_blend = std::stod(next());
        else if (a == "--q-taper") o.q_taper_trick = std::stoi(next());
        else if (a == "--fast-lik") o.fast_lik_weight = std::stod(next());
        else if (a == "--fast-lik-temp") o.fast_lik_temp = std::stod(next());
        else if (a == "--race") o.race = true;
        else if (a == "--race-min") o.race_min = std::stoi(next());
        else if (a == "--race-step") o.race_step = std::stoi(next());
        else if (a == "--race-z") o.race_z = std::stod(next());
        else if (a == "--match-equity") o.match_equity = true;
        else if (a == "--policy") policy_only = true;
        else if (a == "--qgru-only") qgru_only = true;
        else if (a == "--fast-only") fast_only = true;
        else if (a == "--seed") seed = std::stoull(next());
    }
    BgFastPolicyModel fast_model;
    BgBeliefModel belief_model;
    BgQGRUModel q_model;
    if (!fast_path.empty()) {
        if (!fast_model.load(fast_path)) {
            std::cerr << "bg_agent: failed to load --fast " << fast_path << std::endl;
            return 2;
        }
        o.fast_policy = &fast_model;
    }
    if (!belief_path.empty()) {
        if (!belief_model.load(belief_path)) {
            std::cerr << "bg_agent: failed to load --belief " << belief_path << std::endl;
            return 2;
        }
        o.belief_model = &belief_model;
    }
    if (!qgru_path.empty()) {
        if (!q_model.load(qgru_path)) {
            std::cerr << "bg_agent: failed to load --qgru " << qgru_path << std::endl;
            return 2;
        }
        o.q_model = &q_model;
    }
    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "QUIT") break;
        if (cmd != "PLAY") {
            std::cout << "ERR unknown command" << std::endl;
            continue;
        }
        PublicState ps;
        long long hand = 0;
        int contract = 0, n = 0;
        in >> ps.first >> ps.me >> hand >> contract >> ps.declarer;
        ps.hand = Mask(hand);
        ps.contract = std::uint8_t(contract);
        in >> n;
        for (int i = 0, s, b; i < n && in >> s >> b; ++i) ps.bids.push_back({s, std::uint8_t(b)});
        in >> n;
        for (int i = 0; i < n; ++i) {
            Declared d;
            in >> d.seat >> d.type >> d.card;
            ps.declared.push_back(d);
        }
        in >> n;
        for (int i = 0; i < n; ++i) {
            PublicPlay p;
            int b = 0;
            in >> p.seat >> p.card >> b;
            p.belote = b != 0;
            ps.plays.push_back(p);
        }
        in >> ps.sn >> ps.ew >> ps.hanging;
        int card = kNoCard;
        try {
            if (qgru_only && q_model.loaded()) {
                card = q_model.chooseCard(ps);
            } else if (fast_only && fast_model.loaded()) {
                card = fast_model.chooseCard(ps);
            } else if (policy_only) {
                DealSampler s(ps);
                auto h = s.sample(rng);
                auto r = h ? s.replay(*h) : std::nullopt;
                if (!r) throw std::runtime_error("no consistent world");
                card = rulePolicy(*r);
            } else {
                o.seed = rng() | 1;
                card = searchCard(ps, o).card;
            }
        } catch (const std::exception& e) {
            std::cerr << "bg_agent: " << e.what() << " on: " << line << std::endl;
            card = kNoCard;
        }
        std::cout << card << " 1" << std::endl;
    }
    return 0;
}
