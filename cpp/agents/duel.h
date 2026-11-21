#pragma once

// Duplicate-deal and full-match (101) evaluation harness (PLAN.md M0/M1).
//
// Supports all four evaluation modes with mirrored team swaps and paired SE:
//  1. Card-play duplicate: fixed `bidder` (default "rule"), A vs B play.
//  2. Bidding-only duplicate: `bidder = "self"`, fixed `player` (e.g. "rule").
//  3. Full-deal duplicate: `bidder = "self"`, `player = ""` (A vs B bid+play).
//  4. Mixed-partner duplicate: `partner != ""` (e.g. "rule"), (A, partner) vs
//     (B, partner), mirroring which team has A and rotating partner seats.
//  5. Full 101-point match duplicate (`runMatchDuel`): mirrored pairs of full
//     matches using the exact same deck sequence, with optional `partner`,
//     fixed `bidder`, or fixed `player`.

#include "belote/rules.h"

#include <cstdint>
#include <string>

namespace belote {

// Canonical config-version tag logged with every benchmark result.
std::string configSummary(const Config& cfg);

struct DuelOptions {
    std::string a = "ismcts:1000";
    std::string b = "rule";
    std::string bidder = "rule";   // fixed bidder spec, or "self" (each seat bids)
    std::string player = "";       // empty: each seat plays; non-empty: fixed card player (bidding-only mode)
    std::string partner = "";      // empty: pure A vs B; non-empty: mixed-partner (A+partner vs B+partner)
    int deals = 1000;              // decks; each is played twice (teams swapped)
    std::uint64_t seed = 1;
    int threads = 1;               // decks played in parallel
    Config config = Config::belgian();
};

struct DuelResult {
    int decks = 0;                   // paired decks scored (played twice each)
    int skipped = 0;                 // decks where both runs all-passed
    double units = 0, units_se = 0;  // A's margin per deal, recorded units
    double raw = 0, raw_se = 0;      // A's margin per deal, raw credited points
    double ms_per_card_a = 0, ms_per_card_b = 0;
    double seconds = 0;
    std::string config_summary;
};

DuelResult runDuel(const DuelOptions& options);

struct MatchDuelOptions {
    std::string a = "rollout:32";
    std::string b = "rule";
    std::string bidder = "self";   // "self" (each seat bids) or fixed bidder spec
    std::string player = "";       // empty (each seat plays) or fixed card player
    std::string partner = "";      // empty (pure teams) or conventional partner spec (mixed-partner)
    int matches = 200;             // mirrored pairs of full matches (total matches = 2 * matches)
    std::uint64_t seed = 1;
    int threads = 1;
    Config config = Config::belgian();
};

struct MatchDuelResult {
    int pairs = 0;                           // mirrored match pairs played
    double win_rate = 0, win_rate_se = 0;    // A's match win rate in [0, 1]
    double score_margin = 0, score_margin_se = 0;  // A's final match score margin
    double mean_deals = 0;                   // average deals per match (incl. redeals)
    double ms_per_card_a = 0, ms_per_card_b = 0;
    double seconds = 0;
    std::string config_summary;
};

MatchDuelResult runMatchDuel(const MatchDuelOptions& options);

}  // namespace belote
