#pragma once

// Players for evaluation: a common interface, four baselines and a factory.
//
// Contract: an agent is handed the whole Deal but may use only what its seat
// knows -- the public play, its own hand, the blik. Search agents go through
// observe() (belote/observation.h), which enforces this; the rule agents
// below read only their own hand and public state.

#include "belote/deal.h"
#include "search/bid_model.h"
#include "search/fast_policy.h"
#include "search/ismcts.h"
#include "search/rollout.h"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace belote {

// Current match scoreboard passed to score-aware agents before each deal.
struct MatchState {
    std::array<int, 2> scores = {0, 0};
    int pot = 0;
    bool valid = false;
};

class Agent {
public:
    virtual ~Agent() = default;
    virtual std::string name() const = 0;
    // Reset any internal RNG for reproducible evaluations (no-op for stateless agents).
    virtual void reseed(std::uint64_t) {}
    // Optional match scoreboard context for score-aware bidding and play.
    virtual void setMatchState(const MatchState&) {}
    // Optional BidModel attachment (for M4 match-equity leaves & auction likelihood).
    virtual void setBidModel(std::shared_ptr<const BidModel>) {}
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

// Phase 3 Toon conceal gate (PLAN.md §4.3): conceals at trick 1 when an
// opponent has already announced a strictly superior toon (kind / carré rank)
// and partner has already played without matching or beating it.
bool smartChooseDeclare(const Deal& d);

// Uniformly random legal bids (taking with probability p_take) and cards.
class RandomAgent : public Agent {
public:
    explicit RandomAgent(std::uint64_t seed, double p_take = 0.3) : rng_(seed), p_take_(p_take) {}
    std::string name() const override { return "random"; }
    void reseed(std::uint64_t seed) override { rng_.seed(seed); }
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
    explicit IsmctsAgent(IsmctsOptions options, std::uint64_t seed = 0)
        : options_(options), rng_(seed), seeded_(seed != 0) {}
    std::string name() const override { return "ismcts:" + std::to_string(options_.iterations); }
    void reseed(std::uint64_t seed) override {
        rng_.seed(seed);
        seeded_ = seed != 0;
    }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override {
        IsmctsOptions o = options_;
        if (seeded_) {
            o.seed = rng_();
            if (o.seed == 0) o.seed = 1;
        }
        return ismctsSearch(d, o).best_move;
    }

private:
    IsmctsOptions options_;
    std::mt19937_64 rng_;
    bool seeded_ = false;
};

// Shared-world heuristic rollout card play (with optional exact AB endgame);
// bids with the rule bidder.
class RolloutAgent : public Agent {
public:
    explicit RolloutAgent(RolloutOptions options, std::uint64_t seed = 0)
        : options_(options), rng_(seed), seeded_(seed != 0) {}
    std::string name() const override {
        std::string n = "rollout:" + std::to_string(options_.worlds);
        if (options_.ab_handoff_trick >= 0) n += ":ab" + std::to_string(options_.ab_handoff_trick);
        return n;
    }
    void reseed(std::uint64_t seed) override {
        rng_.seed(seed);
        seeded_ = seed != 0;
    }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override {
        RolloutOptions o = options_;
        if (seeded_) {
            o.seed = rng_();
            if (o.seed == 0) o.seed = 1;
        }
        return rolloutSearch(d, o).best_move;
    }

private:
    RolloutOptions options_;
    std::mt19937_64 rng_;
    bool seeded_ = false;
};

// Direct greedy FastPolicy card play; bids with the rule bidder.
class FastPolicyAgent : public Agent {
public:
    FastPolicyAgent(std::shared_ptr<const FastPolicyModel> model, std::string label)
        : model_(std::move(model)), name_(std::move(label)) {}
    std::string name() const override { return name_; }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    int chooseCard(const Deal& d) override { return model_->chooseCard(d); }

private:
    std::shared_ptr<const FastPolicyModel> model_;
    std::string name_;
};

// FastPolicy + WorldSampler + play-history weighting + exact AB endgame hybrid.
class HybridAgent : public Agent {
public:
    HybridAgent(std::shared_ptr<const FastPolicyModel> model, HybridOptions options, std::string label,
                std::uint64_t seed = 0)
        : model_(std::move(model)), options_(options), name_(std::move(label)), rng_(seed), seeded_(seed != 0) {}
    std::string name() const override { return name_; }
    void reseed(std::uint64_t seed) override {
        rng_.seed(seed);
        seeded_ = seed != 0;
    }
    void setMatchState(const MatchState& ms) override { ms_ = ms; }
    void setBidModel(std::shared_ptr<const BidModel> bm) override { bid_model_ = std::move(bm); }
    Bid chooseBid(const Deal& d) override { return ruleBid(d); }
    bool chooseDeclare(const Deal& d) override { return smartChooseDeclare(d); }
    int chooseCard(const Deal& d) override {
        HybridOptions o = options_;
        if (seeded_) {
            o.seed = rng_();
            if (o.seed == 0) o.seed = 1;
        }
        if (bid_model_) o.bid_model = bid_model_.get();
        if (ms_.valid) {
            o.match_scores = ms_.scores;
            o.match_pot = ms_.pot;
        } else {
            o.use_match_equity = false;
        }
        return hybridSearch(d, *model_, o).best_move;
    }

private:
    std::shared_ptr<const FastPolicyModel> model_;
    std::shared_ptr<const BidModel> bid_model_;
    MatchState ms_{};
    HybridOptions options_;
    std::string name_;
    std::mt19937_64 rng_;
    bool seeded_ = false;
};

// Score-aware neural bidder (DealOutcomeNet + BidNet + optional Stage 3 search)
// combined with any card-play Agent.
class BidWrapperAgent : public Agent {
public:
    BidWrapperAgent(std::shared_ptr<const BidModel> bid_model, std::unique_ptr<Agent> player,
                    std::shared_ptr<const FastPolicyModel> search_policy, int search_worlds, double search_margin,
                    std::string label, std::uint64_t seed = 0)
        : bid_model_(std::move(bid_model)),
          player_(std::move(player)),
          search_policy_(std::move(search_policy)),
          search_worlds_(search_worlds),
          search_margin_(search_margin),
          name_(std::move(label)),
          rng_(seed),
          seeded_(seed != 0) {
        if (player_ && bid_model_) player_->setBidModel(bid_model_);
    }

    std::string name() const override { return name_; }
    void reseed(std::uint64_t seed) override {
        rng_.seed(seed);
        seeded_ = seed != 0;
        player_->reseed(seed ^ 0xA5A5A5A55A5A5A5Aull);
    }
    void setMatchState(const MatchState& ms) override {
        ms_ = ms;
        player_->setMatchState(ms);
    }
    Bid chooseBid(const Deal& d) override {
        const int t = teamOf(d.currentPlayer());
        const int us = ms_.valid ? ms_.scores[t] : 0;
        const int them = ms_.valid ? ms_.scores[1 - t] : 0;
        const int pot = ms_.valid ? ms_.pot : 0;
        if (search_worlds_ > 0 && search_policy_) {
            std::uint64_t s = 0;
            if (seeded_) {
                s = rng_();
                if (s == 0) s = 1;
            }
            return bid_model_->chooseBidSearch(d, *search_policy_, us, them, pot, search_worlds_, search_margin_, s);
        }
        return bid_model_->chooseBid(d, us, them, pot);
    }
    bool chooseDeclare(const Deal& d) override { return smartChooseDeclare(d); }
    int chooseCard(const Deal& d) override { return player_->chooseCard(d); }

private:
    std::shared_ptr<const BidModel> bid_model_;
    std::unique_ptr<Agent> player_;
    std::shared_ptr<const FastPolicyModel> search_policy_;
    int search_worlds_ = 0;
    double search_margin_ = 0.04;
    std::string name_;
    MatchState ms_{};
    std::mt19937_64 rng_;
    bool seeded_ = false;
};

// "random", "rule", "ismcts:<iterations>[:<threads>]",
// "rollout:<worlds>[:ab<trick>]", "fast:<weights.bin>",
// "hybrid:<weights.bin>:<worlds>[:ab<trick>][:race][:z<val>][:t<temp>][:nowt]",
// "bid:<bundle.bin>[:s3=<worlds>[:m=<margin>][:<fast.bin>]][+<play_spec>]".
std::unique_ptr<Agent> makeAgent(const std::string& spec, std::uint64_t seed);

}  // namespace belote
