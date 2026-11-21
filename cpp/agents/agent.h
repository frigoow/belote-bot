#pragma once

// Players for evaluation: a common interface, four baselines and a factory.
//
// Contract: an agent is handed the whole Deal but may use only what its seat
// knows -- the public play, its own hand, the blik. Search agents go through
// observe() (belote/observation.h), which enforces this; the rule agents
// below read only their own hand and public state.

#include "belote/deal.h"
#include "search/ismcts.h"
#include "search/rollout.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace belote {

class Agent {
public:
    virtual ~Agent() = default;
    virtual std::string name() const = 0;
    // Phase kBid1/kBid2, the agent's turn.
    virtual Bid chooseBid(const Deal& d) = 0;
    // Before the agent's first card (only asked when the choice is open).
    virtual bool chooseDeclare(const Deal&) { return true; }
    // Phase kPlay, the agent's turn.
    virtual int chooseCard(const Deal& d) = 0;
};

// The rule bidder on its own: how much a hand of 5 + the blik is worth with
// `trump` as trump, and the bid it makes from that.
double ruleHandValue(Mask cards, int trump);
Bid ruleBid(const Deal& d, double threshold = 5.0);

// Fast heuristic card play (used by RuleAgent and shared-world rollouts).
int ruleChooseCard(const Deal& d);

// Uniformly random legal bids (taking with probability p_take) and cards.
class RandomAgent : public Agent {
public:
    explicit RandomAgent(std::uint64_t seed, double p_take = 0.3) : rng_(seed), p_take_(p_take) {}
    std::string name() const override { return "random"; }
    Bid chooseBid(const Deal& d) override;
    int chooseCard(const Deal& d) override;

private:
    std::mt19937_64 rng_;
    double p_take_;
};

// A plain human-style player: the rule bidder, and simple card play (win
// cheaply when it matters, give partner points, pull trumps as takers).
class RuleAgent : public Agent {
public:
    std::string name() const override { return "rule"; }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override { return ruleChooseCard(d); }
};

// SO-ISMCTS card play over sampled worlds; bids with the rule bidder.
class IsmctsAgent : public Agent {
public:
    explicit IsmctsAgent(IsmctsOptions options) : options_(options) {}
    std::string name() const override { return "ismcts:" + std::to_string(options_.iterations); }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override { return ismctsSearch(d, options_).best_move; }

private:
    IsmctsOptions options_;
};

// Shared-world heuristic rollout card play (with optional exact AB endgame);
// bids with the rule bidder.
class RolloutAgent : public Agent {
public:
    explicit RolloutAgent(RolloutOptions options) : options_(options) {}
    std::string name() const override {
        std::string n = "rollout:" + std::to_string(options_.worlds);
        if (options_.ab_handoff_trick >= 0) n += ":ab" + std::to_string(options_.ab_handoff_trick);
        return n;
    }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override { return rolloutSearch(d, options_).best_move; }

private:
    RolloutOptions options_;
};

// "random", "rule", "ismcts:<iterations>[:<threads>]",
// "rollout:<worlds>[:ab<trick>]".
std::unique_ptr<Agent> makeAgent(const std::string& spec, std::uint64_t seed);

}  // namespace belote
