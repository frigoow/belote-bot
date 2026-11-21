#pragma once

// Rule agent v2, in the structure of the Manille bot's rule agent (manil-bot
// learn/rule_agent.py). Three layers:
//
//  1. rule2ChooseCard: table heuristics using the player's hand and public
//     play. From trick five, if at most 48 layouts remain before declaration
//     checks, an alternative must match or beat the heuristic on every
//     compatible layout under best play and improve on at least one.
//  2. rulePlaySearch ("ruleplay"): the rule card is the first choice; every
//     legal card is then played out on the same sampled worlds with the rule
//     heuristics at all four seats (exact AB for late roots and the last two
//     tricks of early continuations by default), and the rule card is
//     overruled only on a significant paired gain. Worlds are weighted by how
//     likely the auction was under the rule bidder and by how often the
//     others' plays match the rule policy, then resampled from a 4x pool.
//  3. ruleBidSearch ("simbid"): plays every legal bid out on sampled deals
//     consistent with the passes so far.
//
// Strategy (Belgian rules, RULES.md §4: must follow, no obligation to beat a
// plain suit, must trump/overtrump/undertrump only when an opponent is
// winning, free when partner is winning):
//
//  Leading
//   - Takers draw trumps: lead the master trump whenever an opponent may
//     still hold a trump; the taker, long in trumps (3+), leads a low trump
//     without the master (never the 9 into an outstanding jack). Drawing
//     stops once both opponents are known to be out of trumps.
//   - Cash side masters (an ace, a 10 once the ace has gone) unless an
//     opponent may ruff them.
//   - Lead a suit partner is known to be void in while partner may still
//     ruff.
//   - Otherwise lead low from length: never a point card away from an
//     unguarded 10 or a bare honour, a singleton when holding trumps (to ruff
//     later), defenders avoid leading trumps.
//   - Ronde in sight (one team has every trick so far, from trick 3): every
//     trick is fought for.
//  Following
//   - On partner's safe trick, bank points, preserve entries, and prefer a
//     void in a valuable suit partner has shown and can return.
//   - Partner's card may be beaten: overtake with a sure winner when the
//     trick is worth it, else play cheap.
//   - An opponent is winning: last to play, win with the least valuable
//     winner; otherwise win with a sure winner (second hand takes with the
//     master, ducks with a 10 under an outstanding ace), try a cheap winner
//     third hand, else lose the cheapest card (fewest points, keeps masters,
//     trumps and the guard of a 10).
//
// "Sure" uses only public facts (TableView): cards played, cards shown or
// pinned (blik, toon, belot), and the exclusions the legal-move rule implies
// (voids, trump ceilings) -- the same facts as inferConstraints().

#include "agents/agent.h"
#include "belote/observation.h"
#include "belote/sampler.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace belote {

// What seat `me` knows about a deal in play, read straight from the Deal's
// public history (no allocation; matches inferConstraints(observe(d, me))).
struct TableView {
    int me = -1;
    int trump = kNoSuit;
    int taker = -1;
    int takers = -1;  // team
    Mask hand = 0;     // me's cards
    Mask played = 0;   // every card played so far, current trick included
    Mask unseen = 0;   // neither played nor in me's hand
    std::array<Mask, kPlayers> excluded{};  // cards p cannot hold
    std::array<Mask, kPlayers> pinned{};    // cards p must hold (unplayed)
    std::array<int, kPlayers> cards_left{};
    std::array<int, kCards> owner{};        // pinned holder of an unseen card, or -1

    bool mayHold(int p, int card) const;
    bool mayHoldAny(int p, Mask cards) const;
    bool knownVoid(int p, int suit) const { return !mayHoldAny(p, unseen & suitMask(suit)); }
    int unseenIn(int suit) const { return popcount(unseen & suitMask(suit)); }
};

TableView tableView(const Deal& d, int me);

// Why the rule policy chose its card (for diagnostics / reviews).
enum class RuleReason : int {
    kForced,
    kLeadDrawMaster,
    kLeadDrawLow,
    kLeadCashMaster,
    kLeadPartnerRuff,
    kLeadLow,
    kSmear,
    kSecurePartner,
    kCheapUnderPartner,
    kWinLast,
    kWinSure,
    kWinCheapTry,
    kRuffLow,
    kFightRonde,
    kDump,
    kCountedEndgame,
    kCount
};
const char* ruleReasonName(RuleReason r);

// Switches for A/B tests of the rule policy; the defaults are the current
// best. The numbers are the switch's effect in head-to-head rule2 duels
// (units/deal, 95% CI, 3,600-6,000 mirrored decks).
struct RuleParams {
    bool ronde = true;             // fight for / against a ronde (-0.02 +- 0.05 off: neutral, kept)
    bool keep_v2 = false;          // side master keeps 7 * P(not ruffed) instead of 8 + 0.6 * points (-0.24 +- 0.09)
    bool def_master_trump = false; // defenders cash the master trump from trick 5 on (-0.06 +- 0.05)
    bool partner_draw = true;      // the taker's partner draws with the master trump too (-0.03 +- 0.02 off)
    int low_draw_min = 3;          // taker draws with a low trump holding this many: 2 -0.01, 4 -0.06, never -0.21
    bool ten_on_ace = false;       // smear a short master 10 on partner's trick (-0.03 +- 0.04)
    bool void_dump = false;        // discard a side singleton to make a void (0.00 +- 0.01)
    bool force_taker = false;      // defenders lead the taker's void to make the taker ruff (0.00 +- 0.03)
    // From the manual table review (reports/manual-table-review/strategy.md):
    // Both together: +0.028 +- 0.029 (8,000 decks), +0.030 +- 0.020 (16,000, fresh seed).
    bool eff_rank = true;          // a trump's keep value counts the unseen trumps that beat it (§3; +0.017 alone)
    bool partner_void = true;      // on partner's trick, discard a side singleton partner can return (§2; +0.008 alone)
    bool keep_v3 = false;          // a side master keeps 6 + half the unseen points of its suit + its own (§1, pair C; -0.044 +- 0.016)
    bool spare_partner = true;     // no trump lead when only partner can still hold trumps (§4, pair F; +0.004 +- 0.005, rare)
    int count_layouts = 48;        // confirm late plays across all remaining layouts; 0 uses the heuristics alone
};

// The honest rule policy for the current player (phase kPlay).
int rule2ChooseCard(const Deal& d, RuleReason* why = nullptr, const RuleParams& params = {});

struct RulePlayOptions {
    int worlds = 64;
    int ab_handoff_trick = 4;  // >= 0: exact AB per world from this trick on
    double z = 1.0;            // overrule the rule card only if mean gain > z * se ...
    double margin = 0.05;      // ... and > margin (units/deal); z < 0: plain argmax
    double lik_eps = 0.5;      // < 1: weight worlds by eps^(rule mismatches of the others' plays)
    double bid_tau = 1.0;      // > 0: weight worlds by the auction's likelihood under the
    double bid_theta = 5.0;    //      rule bidder, P(take) = sigmoid((value - theta) / tau)
    int pool = 4;              // weighted: sample pool * worlds, resample worlds by weight
    std::uint64_t seed = 0;    // 0: thread-local random_device RNG
    RuleParams rules;          // the policy for the first choice and the play-outs
    MatchState match;
};

struct RulePlayResult {
    int best_move = kNoCard;
    int rule_move = kNoCard;
    RuleReason reason = RuleReason::kForced;
    bool overruled = false;
    int worlds_used = 0;
    std::vector<std::pair<int, double>> move_values;  // weighted mean unit margin per legal card
};

// Log-likelihood of the observed auction in world w, every seat but the
// observer bidding like the rule bidder with a soft threshold. 0 for deals
// without an auction (Deal::fromPlay).
double auctionLogLik(const Observation& obs, const World& w, double theta = 5.0, double tau = 1.0);

// Rule choice + shared-world play-out of every legal card with the rule
// policy at all four seats (exact AB from ab_handoff_trick), overruling the
// rule choice only on a significant paired gain.
RulePlayResult rulePlaySearch(const Deal& d, const RulePlayOptions& options = {});

struct RuleBidOptions {
    int worlds = 0;          // 0: evaluate the known six cards using the rule2 bidder
    int pool = 4;            // sample pool * worlds deals, resample by the auction's likelihood
    double theta = 5.0;      // the other seats' bidder model: P(take) = sigmoid((value - theta) / tau)
    double tau = 1.0;
    std::uint64_t seed = 0;  // 0: thread-local random_device RNG
    RuleParams rules;        // the play-out policy
    MatchState match;
};

struct RuleBidResult {
    Bid bid;
    std::vector<std::pair<Bid, double>> values;  // mean unit margin of each legal bid
};

// Simulation bidder: draws the hidden cards (the other hands' first five,
// the stock) given the passes so far, and plays every legal bid out on the
// same deals -- the rest of the auction by the rule bidder (an all-pass
// redeal scores 0), the cards by the rule policy at all four seats. Bids the
// option with the best mean unit margin for its team.
RuleBidResult ruleBidSearch(const Deal& d, const RuleBidOptions& options);

// "rule2[:simbid<worlds>][:<switch>...]": deterministic table play and bidding.
// nocount disables the bounded endgame check; simbid enables simulation bidding. Other switches: noronde, keepv2,
// deftrump, nopartnerdraw, lowdraw<n>, tenonace, voiddump, force, noeffrank, nopartnervoid, keepv3, nosparepartner.
class Rule2Agent : public Agent {
public:
    Rule2Agent(RuleParams params, RuleBidOptions bidding, std::string label, std::uint64_t seed = 0)
        : params_(params), bidding_(bidding), match_(bidding.match), name_(std::move(label)), rng_(seed), seeded_(seed != 0) {
        bidding_.rules = params_;
    }
    std::string name() const override { return name_; }
    void reseed(std::uint64_t seed) override {
        rng_.seed(seed);
        seeded_ = seed != 0;
    }
    Bid chooseBid(const Deal& d) override;
    void setMatchState(const MatchState& ms) override { match_ = ms; }
    int chooseCard(const Deal& d) override;

protected:
    RuleParams params_;
    RuleBidOptions bidding_;
    MatchState match_;
    std::string name_;
    std::mt19937_64 rng_;
    bool seeded_ = false;
};

// "ruleplay:<worlds>[:ab<trick>|:noab][:z<z>|:argmax][:m<margin>][:bid<tau>|:nobid]
// [:lik<eps>|:nolik][:theta<x>][:pool<k>]" plus the rule2 options.
class RulePlayAgent : public Rule2Agent {
public:
    RulePlayAgent(RulePlayOptions options, RuleBidOptions bidding, std::string label, std::uint64_t seed = 0)
        : Rule2Agent(options.rules, bidding, std::move(label), seed), options_(options) {
        if (options.match.valid) match_ = options.match;
    }
    int chooseCard(const Deal& d) override {
        RulePlayOptions o = options_;
        o.match = match_;
        if (seeded_) {
            o.seed = rng_();
            if (o.seed == 0) o.seed = 1;
        }
        return rulePlaySearch(d, o).best_move;
    }

private:
    RulePlayOptions options_;
};

// Parses a "rule2" / "ruleplay:..." spec; nullptr for any other spec.
std::unique_ptr<Agent> makeRuleAgent(const std::string& spec, std::uint64_t seed);

}  // namespace belote
