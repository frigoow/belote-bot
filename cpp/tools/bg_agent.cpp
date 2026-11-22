// Our Bulgarian Belot card player as a process, for the bridge into Kostov's
// engine (tools/kostov/bridge). Stateless: every request carries the public record.
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
#include "bg/bid_model.h"

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
    std::string fast_path, belief_path, qgru_path, bid_path;
    std::string fast_suit, belief_suit, qgru_suit;
    std::string fast_nt, belief_nt, qgru_nt;
    std::string fast_at, belief_at, qgru_at;
    bool bid_allow_double = false;
    double bid_double_margin = 3.0;
    BidMatchEquityOptions bid_me;
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
        else if (a == "--bid") bid_path = next();
        else if (a == "--bid-double") bid_allow_double = true;
        else if (a == "--bid-double-margin") bid_double_margin = std::stod(next());
        else if (a == "--bid-me") bid_me.enabled = true;
        else if (a == "--bid-me-score") bid_me.min_score = std::stoi(next());
        else if (a == "--bid-me-worlds") bid_me.worlds = std::stoi(next());
        else if (a == "--bid-me-top") bid_me.top_k = std::stoi(next());
        else if (a == "--bid-me-temp") bid_me.prior_temp = std::stod(next());
        else if (a == "--bid-me-dbl-edge") bid_me.dbl_eq_edge = std::stod(next());
        else if (a == "--neural-auction") o.neural_auction_weight = std::stod(next());
        else if (a == "--neural-auction-temp") o.neural_auction_temp = std::stod(next());
        else if (a == "--neural-auction-subsets") o.neural_auction_subsets = std::stoi(next());
        else if (a == "--raw-tiebreak") o.raw_tiebreak = true;
        else if (a == "--exact-worlds") o.exact_worlds = std::stoi(next());
        else if (a == "--fast") fast_path = next();
        else if (a == "--belief") belief_path = next();
        else if (a == "--qgru") qgru_path = next();
        else if (a == "--fast-suit") fast_suit = next();
        else if (a == "--belief-suit") belief_suit = next();
        else if (a == "--qgru-suit") qgru_suit = next();
        else if (a == "--fast-nt") fast_nt = next();
        else if (a == "--belief-nt") belief_nt = next();
        else if (a == "--qgru-nt") qgru_nt = next();
        else if (a == "--fast-at") fast_at = next();
        else if (a == "--belief-at") belief_at = next();
        else if (a == "--qgru-at") qgru_at = next();
        else if (a == "--belief-temp") o.belief_temp = std::stod(next());
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
    BgFastPolicyModel fast_model, fast_m_suit, fast_m_nt, fast_m_at;
    BgBeliefModel belief_model, belief_m_suit, belief_m_nt, belief_m_at;
    BgQGRUModel q_model, q_m_suit, q_m_nt, q_m_at;
    BgBidModel bid_model;
    if (!bid_path.empty()) {
        if (!bid_model.load(bid_path)) {
            std::cerr << "bg_agent: failed to load --bid " << bid_path << std::endl;
            return 2;
        }
        o.bid_model = &bid_model;
    }
    auto loadOne = [](auto& m, const std::string& p, const auto*& dst, const char* tag) -> bool {
        if (p.empty()) return true;
        if (!m.load(p)) {
            std::cerr << "bg_agent: failed to load " << tag << " " << p << std::endl;
            return false;
        }
        dst = &m;
        return true;
    };
    if (!loadOne(fast_model, fast_path, o.fast_policy, "--fast") ||
        !loadOne(belief_model, belief_path, o.belief_model, "--belief") ||
        !loadOne(q_model, qgru_path, o.q_model, "--qgru") ||
        !loadOne(fast_m_suit, fast_suit, o.suit_models.fast, "--fast-suit") ||
        !loadOne(belief_m_suit, belief_suit, o.suit_models.belief, "--belief-suit") ||
        !loadOne(q_m_suit, qgru_suit, o.suit_models.q, "--qgru-suit") ||
        !loadOne(fast_m_nt, fast_nt, o.nt_models.fast, "--fast-nt") ||
        !loadOne(belief_m_nt, belief_nt, o.nt_models.belief, "--belief-nt") ||
        !loadOne(q_m_nt, qgru_nt, o.nt_models.q, "--qgru-nt") ||
        !loadOne(fast_m_at, fast_at, o.at_models.fast, "--fast-at") ||
        !loadOne(belief_m_at, belief_at, o.at_models.belief, "--belief-at") ||
        !loadOne(q_m_at, qgru_at, o.at_models.q, "--qgru-at")) {
        return 2;
    }
    bid_me.fast_at = o.at_models.fast;
    bid_me.fast_suit = o.suit_models.fast;
    bid_me.fast_nt = o.nt_models.fast;
    bid_me.fast_fallback = o.fast_policy;
    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "QUIT") break;
        if (cmd == "BID") {
            BidPublicState bs;
            long long hand = 0;
            int avail = 0, n = 0;
            in >> bs.first >> bs.me >> hand >> avail >> n;
            bs.hand5 = Mask(hand);
            bs.avail = std::uint8_t(avail);
            for (int i = 0, s, b; i < n && in >> s >> b; ++i) bs.bids.push_back({s, std::uint8_t(b)});
            in >> bs.sn >> bs.ew >> bs.hanging;
            int chosen = int(kPass);
            try {
                if (bid_model.loaded()) {
                    chosen = int(bid_me.enabled
                                     ? bid_model.chooseBidMatchEquity(bs, bid_allow_double, bid_double_margin, bid_me, rng)
                                     : bid_model.chooseBid(bs, bid_allow_double, bid_double_margin));
                } else {
                    chosen = int(kPass);
                }
            } catch (const std::exception& e) {
                std::cerr << "bg_agent BID: " << e.what() << std::endl;
                chosen = int(kPass);
            }
            std::cout << chosen << std::endl;
            continue;
        }
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
            if (qgru_only) {
                const NeuralSuite& sp = (ps.contract & kAllTrumps) ? o.at_models
                                        : (ps.contract & kNoTrumps) ? o.nt_models
                                                                    : o.suit_models;
                const BgQGRUModel* qp = (sp.q && sp.q->loaded()) ? sp.q : o.q_model;
                if (!qp || !qp->loaded()) throw std::runtime_error("no qgru model loaded");
                card = qp->chooseCard(ps);
            } else if (fast_only) {
                const NeuralSuite& sp = (ps.contract & kAllTrumps) ? o.at_models
                                        : (ps.contract & kNoTrumps) ? o.nt_models
                                                                    : o.suit_models;
                const BgFastPolicyModel* fp = (sp.fast && sp.fast->loaded()) ? sp.fast : o.fast_policy;
                if (!fp || !fp->loaded()) throw std::runtime_error("no fast model loaded");
                card = fp->chooseCard(ps);
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
