#pragma once

// Our Bulgarian Belot card player: the approach that worked for Manille and
// Belgian Belote, not a port of anyone's bot.
//
//  1. rulePolicy: a fast, honest card policy (own hand + public play) for
//     suit, no-trumps and all-trumps contracts. It is the first choice and the
//     play-out policy for every seat.
//  2. solveEndgame: exact double-dummy negamax over the last tricks, scored
//     with the engine's own deal score (contract, rounding, capot, belot).
//  3. searchCard: draws worlds consistent with the public play (DealSampler),
//     plays every legal card out on the same worlds (exact from the last
//     `endgame_tricks`), optionally weights worlds by how well the others'
//     past plays match the policy, and overrules the rule card only on a
//     significant paired gain.
//
// Values are the deal's game-point margin for the player's team.

#include "bg/neural.h"
#include "bg/rule_policy.h"
#include "bg/view.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace bg {

// Optional pre-deal match score context for exact 151-point Match-Equity leaves.
struct MatchContext {
    bool enabled = false;
    int sn_before = 0;
    int ew_before = 0;
    int first_before = 0;
};

// Plays the deal to the end: every declaration offered is declared, belot is
// claimed whenever allowed, cards come from fast_policy (if provided) or
// rulePolicy, and from `endgame_tricks` tricks before the end (0: never) the
// rest is solved exactly. Returns deal margin or match equity * 100.
double playOut(Round r, int team, int endgame_tricks, const BgFastPolicyModel* fast_policy = nullptr,
               const MatchContext& match_ctx = {});

// Exact value for `team` with all hands visible from the current position
// (kPlay, trick 2 or later), and per legal move of the seat to play.
// When match_ctx.enabled is true, scores are integer basis points round(10000 * equity).
struct EndgameResult {
    int value = 0;
    std::vector<std::pair<int, int>> move_values;
    long nodes = 0;
};
EndgameResult solveEndgame(const Round& r, int team, const MatchContext& match_ctx = {});
int solveEndgameValue(const Round& r, int team, const MatchContext& match_ctx = {});

struct NeuralSuite {
    const BgFastPolicyModel* fast = nullptr;
    const BgBeliefModel* belief = nullptr;
    const BgQGRUModel* q = nullptr;
};

struct SearchOptions {
    int worlds = 48;
    int endgame_tricks = 3;      // exact for the last tricks of every play-out (0: never)
    int root_exact_trick = 5;    // from this trick (1-based) solve the root position exactly per world
    double z = 1.0;              // overrule the prior card if gain > z * se and > margin; z < 0: argmax
    double margin = 0.1;
    double lik_eps = 1.0;        // < 1: weight worlds by eps^(policy mismatches of the others' past plays)
    int pool = 3;                // weighted: sample pool * worlds, resample
    std::uint64_t seed = 0;      // 0: random_device
    double auction_weight = 1.0; // 0 disables the natural bidder likelihood

    // Manille v1.2.3 neural + Belief-DP + CRN racing + Match-Equity extensions:
    // Default (unified) models plus optional per-contract specialist overrides (Suit / NT / AT).
    const BgFastPolicyModel* fast_policy = nullptr;
    const BgBeliefModel* belief_model = nullptr;
    const BgQGRUModel* q_model = nullptr;
    NeuralSuite suit_models;
    NeuralSuite nt_models;
    NeuralSuite at_models;
    double belief_temp = 1.0;
    double fast_lik_weight = 0.0;
    double fast_lik_temp = 1.0;
    double q_blend = 0.0;
    int q_taper_trick = 6;
    bool race = false;
    int race_min = 32;
    int race_step = 32;
    double race_z = 2.0;
    bool match_equity = false;
};

struct SearchResult {
    int card = kNoCard;
    int rule_card = kNoCard;
    bool overruled = false;
    int worlds = 0;
    std::vector<std::pair<int, double>> values;
};

SearchResult searchCard(const PublicState& ps, const SearchOptions& options = {});

// The declarations we make when offered: all of them (the engine keeps the
// first of overlapping ones; carrés come first).
std::vector<Announce> chooseDeclarations(const std::vector<Announce>& offered);

}  // namespace bg
