#include "agents/rule_agent.h"

#include "belote/match.h"
#include "belote/sampler.h"
#include "search/belote_game.h"
#include "search/endgame.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace belote {

// ---------------------------------------------------------------- the view

namespace {

// stronger[trump][card]: the cards of card's suit that beat it.
struct StrongerTable {
    std::array<std::array<Mask, kCards>, kSuits> m{};
    StrongerTable() {
        for (int t = 0; t < kSuits; ++t)
            for (int c = 0; c < kCards; ++c)
                for (int r = 0; r < kRanks; ++r) {
                    const int x = cardOf(suitOf(c), r);
                    if (cardStrength(x, t) > cardStrength(c, t)) m[t][c] |= bit(x);
                }
    }
};
const StrongerTable kStronger;

inline Mask stronger(int card, int trump) { return kStronger.m[trump][card]; }

}  // namespace

bool TableView::mayHold(int p, int card) const {
    if (p == me || !(unseen & bit(card)) || (excluded[p] & bit(card)) || cards_left[p] == 0) return false;
    if (owner[card] >= 0) return owner[card] == p;
    // A player whose remaining cards are all pinned holds nothing else.
    return popcount(pinned[p]) < cards_left[p];
}

bool TableView::mayHoldAny(int p, Mask cards) const {
    for (Mask m = cards; m; m &= m - 1)
        if (mayHold(p, lowestCard(m))) return true;
    return false;
}

TableView tableView(const Deal& d, int me) {
    TableView v;
    v.me = me;
    v.trump = d.trump();
    v.taker = d.taker();
    v.takers = d.takers();
    v.hand = d.hand(me);
    const Config& cfg = d.config();
    const int trump = v.trump;
    const int belot_announcer = d.belotAnnouncer();
    const int belot_card = d.belotCard();
    Mask played = 0;
    std::array<int, kPlayers> n_played{};

    // The legal-move rule read backwards (observation.cpp inferConstraints).
    auto process = [&](Trick& t, int p, int card) {
        if (t.n > 0) {
            const int led = suitOf(t.cards[0]);
            const int w = trickWinnerIndex(t, trump);
            const int winning = t.cards[w];
            const bool partner_winning = t.players[w] == partnerOf(p);
            if (suitOf(card) != led) {
                v.excluded[p] |= suitMask(led);
                if (!partner_winning) {
                    if (suitOf(card) != trump) {
                        if (suitOf(winning) != trump || cfg.must_undertrump)
                            v.excluded[p] |= suitMask(trump);
                        else
                            v.excluded[p] |= stronger(winning, trump);
                    } else if (suitOf(winning) == trump &&
                               cardStrength(card, trump) < cardStrength(winning, trump)) {
                        v.excluded[p] |= stronger(winning, trump);
                    }
                }
            } else if (led == trump && (!partner_winning || cfg.trump_lead_partner_must_overtrump) &&
                       cardStrength(card, trump) < cardStrength(winning, trump)) {
                v.excluded[p] |= stronger(winning, trump);
            }
        }
        if (suitOf(card) == trump && (rankOf(card) == kK || rankOf(card) == kQ)) {
            const int other = cardOf(trump, rankOf(card) == kK ? kQ : kK);
            if (!(played & bit(other))) {
                if (belot_announcer == p && belot_card == card)
                    v.pinned[p] |= bit(other);
                else if (rankOf(card) == kK || !cfg.belot_queen_first_forfeits)
                    v.excluded[p] |= bit(other);
            }
        }
        played |= bit(card);
        ++n_played[p];
        t.cards[t.n] = card;
        t.players[t.n] = p;
        ++t.n;
    };
    for (int i = 0; i < d.tricksPlayed(); ++i) {
        const Trick& h = d.completedTrick(i);
        Trick t;
        for (int k = 0; k < h.n; ++k) process(t, h.players[k], h.cards[k]);
    }
    {
        const Trick& h = d.trick();
        Trick t;
        for (int k = 0; k < h.n; ++k) process(t, h.players[k], h.cards[k]);
    }

    if (d.blik() != kNoCard && d.taker() >= 0) v.pinned[d.taker()] |= bit(d.blik());
    for (int p = 0; p < kPlayers; ++p) {
        if (d.shown(p))
            for (const Toon& toon : d.toons(p)) v.pinned[p] |= toon.cards;
        if (d.announced(p) && d.announcement(p).kind == ToonKind::kCarre)
            for (int s = 0; s < kSuits; ++s) v.pinned[p] |= bit(cardOf(s, d.announcement(p).carre_rank));
    }
    for (int p = 0; p < kPlayers; ++p) {
        v.pinned[p] &= ~played;
        v.cards_left[p] = kTricks - n_played[p];
    }
    v.played = played;
    v.unseen = ~(played | v.hand);
    v.owner.fill(-1);
    for (int p = 0; p < kPlayers; ++p) {
        if (p == me) continue;
        for (Mask m = v.pinned[p] & v.unseen; m; m &= m - 1) v.owner[lowestCard(m)] = p;
    }
    return v;
}

const char* ruleReasonName(RuleReason r) {
    switch (r) {
        case RuleReason::kForced: return "forced";
        case RuleReason::kLeadDrawMaster: return "lead: draw with master trump";
        case RuleReason::kLeadDrawLow: return "lead: draw with low trump";
        case RuleReason::kLeadCashMaster: return "lead: cash side master";
        case RuleReason::kLeadPartnerRuff: return "lead: partner's ruff";
        case RuleReason::kLeadLow: return "lead: low from length";
        case RuleReason::kSmear: return "follow: smear on partner";
        case RuleReason::kSecurePartner: return "follow: secure partner's trick";
        case RuleReason::kCheapUnderPartner: return "follow: cheap under partner";
        case RuleReason::kWinLast: return "follow: win last";
        case RuleReason::kWinSure: return "follow: win sure";
        case RuleReason::kWinCheapTry: return "follow: cheap try";
        case RuleReason::kRuffLow: return "follow: ruff low";
        case RuleReason::kFightRonde: return "follow: ronde at stake";
        case RuleReason::kDump: return "follow: dump cheapest";
        case RuleReason::kCountedEndgame: return "finish: counted remaining cards";
        case RuleReason::kCount: break;
    }
    return "?";
}

// ---------------------------------------------------------------- the policy

namespace {

// Future value of a trump by rank (7 8 9 10 J Q K A): what it is likely to
// capture or protect later, in card points.
constexpr double kTrumpKeep[kRanks] = {4, 5, 18, 12, 25, 6, 7, 14};

struct Policy {
    const Deal& d;
    const RuleParams& P;
    TableView v;
    int me, partner, lho, trump;
    bool attacker;
    Mask legal;
    bool opp_trumps = false;  // an opponent may still hold a trump
    bool we_all = false;      // my team won every trick so far (2+): ronde in sight
    bool they_all = false;    // the opponents did: stop their ronde

    Policy(const Deal& deal, const RuleParams& params, Mask choices)
        : d(deal), P(params), v(tableView(deal, deal.currentPlayer())), me(deal.currentPlayer()), partner(partnerOf(me)),
          lho(nextPlayer(me)), trump(deal.trump()), attacker(teamOf(me) == deal.takers()), legal(choices) {
        const Mask tu = v.unseen & suitMask(trump);
        opp_trumps = v.mayHoldAny(lho, tu) || v.mayHoldAny((me + 3) & 3, tu);
        const int tp = deal.tricksPlayed();
        if (tp >= 2) {
            int ours = 0;
            for (int i = 0; i < tp; ++i) ours += teamOf(deal.trickWinner(i)) == teamOf(me);
            we_all = ours == tp;
            they_all = ours == 0;
        }
    }
    bool rondeLive() const { return P.ronde && (we_all || they_all); }

    int pts(int c) const { return cardPoints(c, trump); }
    bool isTrump(int c) const { return suitOf(c) == trump; }
    // No unseen card of its suit beats it.
    bool master(int c) const { return !(stronger(c, trump) & v.unseen); }
    Mask trumpsUnseen() const { return v.unseen & suitMask(trump); }
    bool isOpp(int p) const { return teamOf(p) != teamOf(me); }

    // p is void in `suit`, or very likely so (at most one card of it is out).
    bool likelyVoid(int p, int suit) const { return v.knownVoid(p, suit) || v.unseenIn(suit) <= 1; }
    // An opponent p may ruff a lead of plain suit `suit`.
    bool ruffThreat(int p, int suit) const {
        return suit != trump && v.mayHoldAny(p, trumpsUnseen()) && likelyVoid(p, suit);
    }

    // What keeping c is worth beyond its own points (which are banked
    // whichever trick it lands in): the cards it may capture later.
    double keepValue(int c) const {
        if (isTrump(c)) {
            if (P.eff_rank) {
                // Trumps no unseen trump separates are interchangeable: value
                // a trump by how many unseen trumps still beat it.
                static constexpr double kByBeaters[8] = {31, 18, 12, 8, 6, 5, 4, 4};
                return kByBeaters[popcount(stronger(c, trump) & v.unseen)];
            }
            return kTrumpKeep[rankOf(c)] + (master(c) ? 6.0 : 0.0);
        }
        if (P.keep_v3 && master(c)) {
            // Its own points are banked whichever trick it lands in; keeping it
            // is worth what it can still capture in its suit, plus the lead.
            int out = 0;
            for (Mask m = v.unseen & suitMask(suitOf(c)); m; m &= m - 1) out += pts(lowestCard(m));
            return 6.0 + 0.5 * out + pts(c);
        }
        if (!P.keep_v2) return master(c) ? 8.0 + 0.6 * pts(c) : 0.3 * pts(c);
        if (master(c)) {
            // A side master wins a later trick unless it is ruffed first.
            const double p_safe = !opp_trumps ? 1.0 : (v.unseenIn(suitOf(c)) >= 4 ? 0.8 : 0.45);
            return 7.0 * p_safe;
        }
        return 0.15 * pts(c);
    }
    // Playing the small card that guards a 10 still under an outstanding ace.
    double guardPenalty(int c) const {
        if (isTrump(c) || rankOf(c) == k10) return 0;
        const int s = suitOf(c);
        const int ten = cardOf(s, k10);
        if (!(v.hand & bit(ten)) || master(ten)) return 0;
        return popcount(v.hand & suitMask(s)) == 2 ? 5.0 : 1.5;
    }
    bool singletonSide(int c) const {
        return !isTrump(c) && popcount(v.hand & suitMask(suitOf(c))) == 1 && (v.hand & suitMask(trump));
    }
    double dumpCost(int c) const {
        return pts(c) + keepValue(c) + guardPenalty(c) - (P.void_dump && singletonSide(c) ? 2.0 : 0.0);
    }
    double smearValue(int c) const {
        double keep = keepValue(c);
        // A side winner is an entry only if we can cash it. Bank a vulnerable
        // winner when partner has already secured the trick.
        if (!isTrump(c) && master(c) && (ruffThreat(lho, suitOf(c)) || ruffThreat((me + 3) & 3, suitOf(c))))
            keep -= 8.0;
        // "10 op de aas": a master 10 left short is smeared rather than kept to be ruffed.
        if (P.ten_on_ace && !isTrump(c) && rankOf(c) == k10 && master(c) && popcount(v.hand & suitMask(suitOf(c))) <= 2)
            keep = 0;
        return pts(c) - keep - guardPenalty(c) - (isTrump(c) ? 4.0 : 0.0);
    }

    template <class F>
    int argBest(Mask m, F score) const {
        int best = kNoCard;
        double best_s = -std::numeric_limits<double>::infinity();
        for (; m; m &= m - 1) {
            const int c = lowestCard(m);
            const double s = score(c);
            if (s > best_s) {
                best_s = s;
                best = c;
            }
        }
        return best;
    }
    int strongestOf(Mask m) const {
        return argBest(m, [&](int c) { return double(cardStrength(c, trump)); });
    }
    int weakestOf(Mask m) const {
        return argBest(m, [&](int c) { return -double(cardStrength(c, trump)) - 0.01 * pts(c); });
    }

    // Would `card`, winning the trick for team `team`, still win after the
    // players `later` (in order) have played?
    bool holds(int card, int team, int led, const int* later, int n_later) const {
        for (int i = 0; i < n_later; ++i) {
            const int q = later[i];
            if (teamOf(q) == team) continue;
            if (isTrump(card)) {
                if (!v.mayHoldAny(q, stronger(card, trump) & v.unseen)) continue;
                if (led == trump || likelyVoid(q, led)) return false;
            } else {
                if (v.mayHoldAny(q, stronger(card, trump) & v.unseen)) return false;
                if (ruffThreat(q, led)) return false;
            }
        }
        return true;
    }

    int lead(RuleReason& why) const {
        const Mask hand = legal;
        const Mask my_trumps = hand & suitMask(trump);
        const int rho = (me + 3) & 3;
        const int my_top_trump = my_trumps ? strongestOf(my_trumps) : kNoCard;
        const bool master_trump = my_top_trump != kNoCard && master(my_top_trump);

        // 1. Takers draw trumps.
        if (attacker && my_trumps && opp_trumps) {
            const int top = strongestOf(my_trumps);
            if (master(top) && (P.partner_draw || me == v.taker)) {
                why = RuleReason::kLeadDrawMaster;
                return top;
            }
            const int n = popcount(my_trumps);
            if (P.partner_draw && partner == v.taker &&
                ((v.pinned[partner] & bit(strongestOf(trumpsUnseen()))) ||
                 (d.tricksPlayed() == 0 && rankOf(weakestOf(my_trumps)) != k9))) {
                why = RuleReason::kLeadDrawLow;
                return weakestOf(my_trumps);
            }
            if (me == v.taker && (n >= P.low_draw_min || (n >= 2 && popcount(trumpsUnseen()) <= 2 && P.low_draw_min < 9))) {
                why = RuleReason::kLeadDrawLow;
                return weakestOf(my_trumps);
            }
        }

        // 2. Cash a side master nobody can ruff.
        {
            int best = kNoCard;
            double best_s = -1;
            for (Mask m = hand & ~suitMask(trump); m; m &= m - 1) {
                const int c = lowestCard(m);
                if (!master(c) || ruffThreat(lho, suitOf(c)) || ruffThreat(rho, suitOf(c))) continue;
                // Nothing left of the suit outside: it is a sure trick but
                // only pulls our own cards -- still worth it unless trumps remain.
                const double s = pts(c) + 1.0;
                if (s > best_s) {
                    best_s = s;
                    best = c;
                }
            }
            if (best != kNoCard) {
                why = RuleReason::kLeadCashMaster;
                return best;
            }
        }

        // 2b. Defenders cash the master trump late, or to stop a ronde.
        if (P.def_master_trump && !attacker && master_trump && (d.tricksPlayed() >= 4 || rondeLive())) {
            why = RuleReason::kLeadDrawMaster;
            return my_top_trump;
        }

        // 3. Partner's ruff.
        if (v.mayHoldAny(partner, trumpsUnseen())) {
            for (int s = 0; s < kSuits; ++s) {
                if (s == trump || !(hand & suitMask(s)) || !v.knownVoid(partner, s)) continue;
                if (attacker == false && ruffThreat(rho, s)) continue;  // RHO overruffs partner
                why = RuleReason::kLeadPartnerRuff;
                return weakestOf(hand & suitMask(s));
            }
        }

        // 4. Low from length.
        why = RuleReason::kLeadLow;
        return argBest(hand, [&](int c) {
            const int s = suitOf(c);
            const Mask in_suit = hand & suitMask(s);
            if (c != weakestOf(in_suit)) return -1e9;  // the suit's lowest card only
            const int n = popcount(in_suit);
            double score = -1.5 * pts(c) + 0.6 * n;
            if (s == trump) {
                score -= attacker ? 4.0 : 14.0;
                // Nobody but partner can hold a trump: leading one only pulls partner's.
                if (P.spare_partner && !opp_trumps && v.mayHoldAny(partner, trumpsUnseen())) score -= 12.0;
            } else {
                const int ten = cardOf(s, k10);
                if ((in_suit & bit(ten)) && c != ten && !master(ten)) score -= (n == 2 ? 8.0 : 4.0);
                const int king = cardOf(s, kK);
                if ((in_suit & bit(king)) && c != king && !master(king) && n == 2) score -= 2.0;
                if (ruffThreat(lho, s) || ruffThreat(rho, s)) {
                    // Defenders may force the taker to ruff and shorten the trumps.
                    const bool taker_ruffs = v.taker >= 0 && isOpp(v.taker) && ruffThreat(v.taker, s);
                    score += (P.force_taker && !attacker && taker_ruffs) ? 2.0 : -3.0;
                }
                if (n == 1 && my_trumps && (hand & ~suitMask(trump)) != in_suit) score += 3.0;
                if (master(c)) score += 2.0;
            }
            return score;
        });
    }

    int follow(RuleReason& why) const {
        const Trick& t = d.trick();
        const int n = t.n;
        const int led = suitOf(t.cards[0]);
        const int w = trickWinnerIndex(t, trump);
        const int win_card = t.cards[w];
        const bool partner_winning = t.players[w] == partner;
        int later[3];
        const int n_later = kPlayers - 1 - n;
        for (int k = 0; k < n_later; ++k) later[k] = (me + 1 + k) & 3;
        int trick_pts = 0;
        for (int i = 0; i < n; ++i) trick_pts += pts(t.cards[i]);
        const int team = teamOf(me);

        Mask winners = 0, sure = 0;
        for (Mask m = legal; m; m &= m - 1) {
            const int c = lowestCard(m);
            const bool beats = suitOf(c) == suitOf(win_card) ? cardStrength(c, trump) > cardStrength(win_card, trump)
                                                             : isTrump(c);
            if (!beats) continue;
            winners |= bit(c);
            if (holds(c, team, led, later, n_later)) sure |= bit(c);
        }
        auto cheapestKeep = [&](Mask m) { return argBest(m, [&](int c) { return -keepValue(c) + 0.01 * pts(c); }); };
        auto dump = [&]() {
            why = RuleReason::kDump;
            return argBest(legal, [&](int c) { return -dumpCost(c); });
        };

        if (partner_winning) {
            const bool safe = n_later == 0 || holds(win_card, team, led, later, n_later);
            if (safe) {
                why = RuleReason::kSmear;
                const bool discarding = !(v.hand & suitMask(led));
                const bool have_trumps = (v.hand & suitMask(trump)) != 0;
                return argBest(legal, [&](int c) {
                    double sv = smearValue(c);
                    // A side singleton thrown on partner's trick makes a void
                    // partner can lead into for a ruff.
                    if (P.partner_void && discarding && have_trumps && singletonSide(c) &&
                        !v.knownVoid(partner, suitOf(c)) && v.unseenIn(suitOf(c)) >= 2) {
                        sv += (v.pinned[partner] & suitMask(suitOf(c))) ? 8.0 : 3.0;
                        // For two cheap singletons, make the void where the
                        // outstanding honors give a later ruff more to win.
                        for (Mask out = v.unseen & suitMask(suitOf(c)); out; out &= out - 1)
                            sv += 0.05 * pts(lowestCard(out));
                    }
                    return sv;
                });
            }
            if (sure && (trick_pts >= 10 || rondeLive())) {
                // Overtake partner with a sure winner when there is something to protect.
                why = RuleReason::kSecurePartner;
                return cheapestKeep(sure);
            }
            why = RuleReason::kCheapUnderPartner;
            return argBest(legal, [&](int c) { return -dumpCost(c); });
        }

        // An opponent is winning.
        if (n_later == 0) {
            if (winners) {
                why = RuleReason::kWinLast;
                // Bank the points when two winners have comparable future
                // value. In particular, spend the richer of equivalent trumps.
                return argBest(winners, [&](int c) { return pts(c) - keepValue(c); });
            }
            return dump();
        }
        const bool forced_trumps = winners && !(legal & ~suitMask(trump)) && led != trump;
        if (sure) {
            const int c = cheapestKeep(sure);
            // A ruff that holds is worth a big trump only on a rich trick.
            if (!isTrump(c) || led == trump || trick_pts >= 10 || keepValue(c) <= 12 || rondeLive()) {
                why = RuleReason::kWinSure;
                return c;
            }
        }
        if (winners && rondeLive()) {
            // Every trick counts: play the winner most likely to hold.
            why = RuleReason::kFightRonde;
            return strongestOf(winners);
        }
        if (forced_trumps) {
            why = RuleReason::kRuffLow;
            return cheapestKeep(legal);
        }
        if (winners && n == 2) {
            // Third hand: a cheap winner may get through.
            const int c = argBest(winners, [&](int x) { return -(pts(x) + keepValue(x)); });
            if (pts(c) <= 4 && keepValue(c) <= 8) {
                why = RuleReason::kWinCheapTry;
                return c;
            }
        }
        return dump();
    }

};

Mask ruleChoices(const Deal& d) {
    Mask legal = d.legalMoves();
    if (!legal) throw std::logic_error("rule card requested outside play");
    if (d.config().belot_queen_first_forfeits) {
        const Mask k_bit = bit(cardOf(d.trump(), kK));
        const Mask q_bit = bit(cardOf(d.trump(), kQ));
        if (legal & k_bit) legal &= ~q_bit;
    }
    return legal;
}

int heuristicCard(const Deal& d, RuleReason* why, const RuleParams& params) {
    const Mask legal = ruleChoices(d);
    if ((legal & (legal - 1)) == 0) {
        if (why) *why = RuleReason::kForced;
        return lowestCard(legal);
    }
    const Policy pol(d, params, legal);
    RuleReason r = RuleReason::kForced;
    const int c = d.trick().n == 0 ? pol.lead(r) : pol.follow(r);
    if (why) *why = r;
    return c;
}

EndgameOptions finishOptions(const MatchState& match) {
    EndgameOptions options;
    if (match.valid) {
        options.objective = EndgameObjective::kMatchFinish;
        options.match_scores = match.scores;
        options.match_pot = match.pot;
    }
    return options;
}

double finishedValue(const Deal& d, int team, const MatchState& match) {
    if (!match.valid) return unitMargin(d, team);
    const double value = matchFinishValue(d.result(), d.config(), match.scores, match.pot);
    return (team == 0 ? value : -value) * recordedPoints(kDealPoints, d.config());
}

// Change a heuristic play only if it loses nothing in *every* compatible
// layout under best play, and improves at least one. The size cap applies
// before enumeration. A partial sample cannot establish this condition.
int countedCard(const Deal& d, RuleReason* why, const RuleParams& params, const MatchState& match) {
    const int initial = heuristicCard(d, why, params);
    if (params.count_layouts <= 0 || d.tricksPlayed() < 4) return initial;
    Mask candidates = ruleChoices(d);
    if (candidates == bit(initial)) return initial;
    WorldSampler sampler(observe(d, d.currentPlayer()));
    const auto layouts = sampler.enumerate(static_cast<std::uint64_t>(params.count_layouts));
    if (layouts.empty()) return initial;

    std::array<std::int64_t, kCards> gain{};
    EndgameOptions options = finishOptions(match);
    options.tt_log2 = 8;
    for (const World& layout : layouts) {
        const auto result = solveEndgame(sampler.replay(layout), options);
        const auto& moves = result.move_values;
        const int base = std::find_if(moves.begin(), moves.end(),
            [&](const auto& move) { return move.first == initial; })->second;
        for (const auto& [card, value] : moves) {
            if (value < base) candidates &= ~bit(card);
            gain[card] += value - base;
        }
        if (candidates == bit(initial)) return initial;
    }
    int best = initial;
    for (Mask m = candidates; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (gain[c] > gain[best]) best = c;
    }
    if (best != initial && why) *why = RuleReason::kCountedEndgame;
    return best;
}

}  // namespace

int rule2ChooseCard(const Deal& d, RuleReason* why, const RuleParams& params) {
    return countedCard(d, why, params, {});
}

int Rule2Agent::chooseCard(const Deal& d) {
    return countedCard(d, nullptr, params_, match_);
}

// ---------------------------------------------------------------- play-out evaluation

namespace {

std::mt19937_64& rulePlayRng() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

// How many of the other seats' observed plays the rule policy would not have
// made in world w (seats with a single legal card are skipped).
int ruleMismatches(const Observation& obs, const World& w, const RuleParams& params) {
    std::array<Mask, kPlayers> start{};
    for (int p = 0; p < kPlayers; ++p) start[p] = w.hands[p] | obs.playedBy(p);
    Deal r = Deal::fromPlay(obs.cfg, obs.dealer, obs.taker, obs.trump, start, obs.blik, obs.bids_made);
    for (int p = 0; p < kPlayers; ++p) {
        if (!r.canSetDeclare(p)) continue;
        if (p == obs.observer) r.setDeclare(p, obs.declare_self);
        else if (obs.playedBy(p)) r.setDeclare(p, obs.announced[p]);
    }
    int mismatches = 0;
    for (const Play& pl : obs.plays) {
        if (pl.player != obs.observer && popcount(r.legalMoves()) > 1 && heuristicCard(r, nullptr, params) != pl.card) ++mismatches;
        r.play(pl.card);
    }
    return mismatches;
}

}  // namespace

// Log-likelihood of the observed auction in world w, every other seat bidding
// like the rule bidder with a soft threshold: P(take | value) =
// sigmoid((value - theta) / tau). A seat bid on its first five cards and the
// blik; which five of its eight (the taker: of its seven besides the blik)
// came first is unknown, so the likelihood is averaged over every choice.
double auctionLogLik(const Observation& obs, const World& w, double theta, double tau) {
    if (obs.blik == kNoCard || obs.bids_made < 0 || obs.taker < 0) return 0.0;
    const int blik_suit = suitOf(obs.blik);
    const Mask blik_bit = bit(obs.blik);
    auto p_take = [&](double value) { return 1.0 / (1.0 + std::exp(-(value - theta) / tau)); };
    double ll = 0;
    for (int p = 0; p < kPlayers; ++p) {
        if (p == obs.observer) continue;
        int nodes[2];
        int nn = 0;
        for (int k = 0; k <= obs.bids_made && nn < 2; ++k)
            if (((obs.dealer + 1 + k) & 3) == p) nodes[nn++] = k;
        if (nn == 0) continue;
        const Mask pool = (w.hands[p] | obs.playedBy(p)) & ~blik_bit;
        int cards[kTricks];
        int n = 0;
        for (Mask m = pool; m; m &= m - 1) cards[n++] = lowestCard(m);
        double sum = 0;
        int count = 0;
        // Every 5-subset of the n cards (Gosper's hack over n bits).
        for (unsigned x = 0x1Fu; x < (1u << n);) {
            Mask five = blik_bit;
            for (unsigned b = x; b; b &= b - 1) five |= bit(cards[__builtin_ctz(b)]);
            double pr = 1;
            for (int i = 0; i < nn; ++i) {
                const int k = nodes[i];
                const bool took = k == obs.bids_made;
                if (k < kPlayers) {
                    const double t = p_take(ruleHandValue(five, blik_suit));
                    pr *= took ? t : 1 - t;
                } else if (took) {
                    pr *= p_take(ruleHandValue(five, obs.trump));
                } else {
                    for (int s = 0; s < kSuits; ++s)
                        if (s != blik_suit || obs.cfg.round2_allow_blik_suit) pr *= 1 - p_take(ruleHandValue(five, s));
                }
            }
            sum += pr;
            ++count;
            const unsigned c = x & (~x + 1);
            const unsigned r = x + c;
            x = (((r ^ x) >> 2) / c) | r;
        }
        ll += std::log(std::max(sum / count, 1e-300));
    }
    return ll;
}

RulePlayResult rulePlaySearch(const Deal& d, const RulePlayOptions& o) {
    if (d.phase() != Phase::kPlay) throw std::logic_error("rulePlaySearch: deal is not in kPlay");
    RulePlayResult res;
    res.rule_move = o.ab_handoff_trick < 0 ? heuristicCard(d, &res.reason, o.rules) :
                                            countedCard(d, &res.reason, o.rules, o.match);
    res.best_move = res.rule_move;
    const Mask legal = d.legalMoves();
    if (popcount(legal) == 1 || o.worlds <= 0) {
        res.move_values.push_back({res.best_move, 0.0});
        return res;
    }

    const int me = d.currentPlayer();
    const int my_team = teamOf(me);
    const Observation obs = observe(d, me);
    const WorldSampler sampler(obs);

    const bool weighted = o.lik_eps < 1.0 || o.bid_tau > 0;
    std::vector<World> worlds;
    if (sampler.count() > 0 && sampler.count() <= std::uint64_t(o.worlds)) worlds = sampler.enumerate(std::uint64_t(o.worlds));
    const bool enumerated = !worlds.empty();
    std::mt19937_64 local_rng(o.seed);
    std::mt19937_64& rng = o.seed ? local_rng : rulePlayRng();
    if (!enumerated) {
        // With world weights, draw a larger pool and resample from it below,
        // so the play-outs still run on o.worlds equally weighted worlds.
        const int n = weighted ? o.worlds * std::max(1, o.pool) : o.worlds;
        worlds.reserve(std::size_t(n));
        for (int i = 0; i < n; ++i) {
            auto w = sampler.sample(rng, 500);
            if (!w) w = sampler.sampleUnchecked(rng);
            worlds.push_back(*w);
        }
    }

    std::vector<double> wt(worlds.size(), 1.0);
    if (o.lik_eps < 1.0 && !obs.plays.empty()) {
        for (std::size_t i = 0; i < worlds.size(); ++i) wt[i] = std::pow(o.lik_eps, ruleMismatches(obs, worlds[i], o.rules));
    }
    if (o.bid_tau > 0) {
        std::vector<double> ll(worlds.size());
        double top = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < worlds.size(); ++i)
            top = std::max(top, ll[i] = auctionLogLik(obs, worlds[i], o.bid_theta, o.bid_tau));
        for (std::size_t i = 0; i < worlds.size(); ++i) wt[i] *= std::exp(ll[i] - top);
    }
    double wsum = 0;
    for (double x : wt) wsum += x;
    for (double& x : wt) x /= wsum;
    if (weighted && !enumerated && worlds.size() > std::size_t(o.worlds)) {
        // Systematic resampling down to o.worlds worlds.
        std::vector<World> picked;
        picked.reserve(std::size_t(o.worlds));
        const double step = 1.0 / o.worlds;
        double u = std::uniform_real_distribution<double>(0.0, step)(rng);
        double cum = wt[0];
        std::size_t j = 0;
        for (int k = 0; k < o.worlds; ++k, u += step) {
            while (u > cum && j + 1 < worlds.size()) cum += wt[++j];
            picked.push_back(worlds[j]);
        }
        worlds.swap(picked);
        wt.assign(worlds.size(), 1.0 / double(worlds.size()));
    }
    const std::size_t nw = worlds.size();
    res.worlds_used = int(nw);

    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    const std::size_t nm = moves.size();
    std::vector<double> val(nm * nw, 0.0);  // [move][world]

    const bool use_ab = o.ab_handoff_trick >= 0 && d.tricksPlayed() >= o.ab_handoff_trick;
    const EndgameOptions end = finishOptions(o.match);
    const double scale = o.match.valid ? recordedPoints(kDealPoints, d.config()) / 10000.0 : 1.0;
    // Early roots evaluate many moves on each layout. Count the last two
    // tricks in those continuations; reserve the deeper solve for late roots.
    const int rollout_handoff = o.ab_handoff_trick < 0 ? kTricks : std::max(kTricks - 2, o.ab_handoff_trick);
    for (std::size_t wi = 0; wi < nw; ++wi) {
        const Deal base = sampler.replay(worlds[wi]);
        if (use_ab) {
            const EndgameResult er = solveEndgame(base, end);
            for (std::size_t i = 0; i < nm; ++i)
                for (const auto& [c, value] : er.move_values)
                    if (c == moves[i]) val[i * nw + wi] = scale * value;
            continue;
        }
        for (std::size_t i = 0; i < nm; ++i) {
            Deal sim = base;
            sim.play(moves[i]);
            while (sim.phase() == Phase::kPlay && sim.tricksPlayed() < rollout_handoff)
                sim.play(heuristicCard(sim, nullptr, o.rules));
            val[i * nw + wi] = sim.phase() == Phase::kDone ? finishedValue(sim, my_team, o.match) :
                (my_team == 0 ? scale : -scale) * solveEndgameValue(sim, end);
        }
    }

    std::size_t best = 0, rule_i = 0;
    std::vector<double> mean(nm, 0.0);
    for (std::size_t i = 0; i < nm; ++i) {
        for (std::size_t wi = 0; wi < nw; ++wi) mean[i] += wt[wi] * val[i * nw + wi];
        res.move_values.push_back({moves[i], mean[i]});
        if (mean[i] > mean[best]) best = i;
        if (moves[i] == res.rule_move) rule_i = i;
    }
    if (best == rule_i) return res;
    if (o.z < 0) {
        res.best_move = moves[best];
        res.overruled = true;
        return res;
    }
    // Paired test of the play-out's favourite against the rule card.
    const double m = mean[best] - mean[rule_i];
    double var = 0, w2 = 0;
    for (std::size_t wi = 0; wi < nw; ++wi) {
        const double diff = val[best * nw + wi] - val[rule_i * nw + wi] - m;
        var += wt[wi] * diff * diff;
        w2 += wt[wi] * wt[wi];
    }
    const double n_eff = 1.0 / w2;
    const double se = std::sqrt(std::max(var, 1e-12) / n_eff);
    if (m > o.margin && m > o.z * se) {
        res.best_move = moves[best];
        res.overruled = true;
    }
    return res;
}

// ---------------------------------------------------------------- bidding

namespace {

Bid tableBid(const Deal& d) {
    const Mask six = d.hand(d.currentPlayer()) | bit(d.blik());
    // A run already completed by the blik can move the contract threshold
    // considerably. Discount it because an opponent can beat the declaration.
    const int toon = d.config().toon_enabled ? totalPoints(findToons(six, d.config())) : 0;
    auto strength = [&](int suit) {
        const Mask trumps = six & suitMask(suit);
        double value = ruleHandValue(six, suit) + std::min(4.5, 0.043 * toon) + 0.55 * (popcount(trumps) - 2);
        if (trumps & bit(cardOf(suit, kJ))) {
            value += 0.45;
            if (trumps & bit(cardOf(suit, k9))) value += 0.8;
        }
        return value;
    };
    const int turned = suitOf(d.blik());
    const bool on_lead = d.currentPlayer() == nextPlayer(d.dealer());
    if (d.phase() == Phase::kBid1) {
        double alternative = 0.0;
        for (int suit = 0; suit < kSuits; ++suit)
            if (suit != turned) alternative = std::max(alternative, strength(suit));
        double threshold = 5.55 + 0.32 * std::max(0.0, alternative - 5.0);
        if (rankOf(d.blik()) == kJ || rankOf(d.blik()) == k9) threshold -= 0.65;
        return strength(turned) >= threshold ? Bid::take() : Bid::pass();
    }
    Bid best = Bid::pass();
    double best_value = on_lead ? 5.3 : 5.75;
    for (Bid bid : d.legalBids()) {
        if (bid.kind != Bid::kChoose) continue;
        const double value = strength(bid.suit);
        if (value >= best_value) {
            best_value = value;
            best = bid;
        }
    }
    return best;
}

}  // namespace

RuleBidResult ruleBidSearch(const Deal& d, const RuleBidOptions& o) {
    if (d.phase() != Phase::kBid1 && d.phase() != Phase::kBid2) throw std::logic_error("ruleBidSearch: not bidding");
    if (o.worlds <= 0) return {tableBid(d), {}};
    const int me = d.currentPlayer();
    const int dealer = d.dealer();
    const int blik = d.blik();
    const int blik_suit = suitOf(blik);
    const int passes = d.bidsMade();
    const Mask hand = d.hand(me);
    const Config& cfg = d.config();
    RuleBidResult res;
    const std::vector<Bid> options = d.legalBids();

    // Deck slots: seat i after the dealer gets [3i, 3i+3) and [12+2i, 12+2i+2).
    auto slotsOf = [](int i, int* out) {
        out[0] = 3 * i, out[1] = 3 * i + 1, out[2] = 3 * i + 2, out[3] = 12 + 2 * i, out[4] = 13 + 2 * i;
    };
    const int my_i = (me - dealer - 1 + 2 * kPlayers) % kPlayers;
    int my_slots[5];
    slotsOf(my_i, my_slots);
    std::array<bool, kCards> mine_slot{};
    for (int k : my_slots) mine_slot[std::size_t(k)] = true;
    std::vector<int> hidden;
    for (int c = 0; c < kCards; ++c)
        if (!(hand & bit(c)) && c != blik) hidden.push_back(c);
    std::vector<int> my_cards;
    for (Mask m = hand; m; m &= m - 1) my_cards.push_back(lowestCard(m));

    std::mt19937_64 local_rng(o.seed);
    thread_local std::mt19937_64 tl_rng{std::random_device{}()};
    std::mt19937_64& rng = o.seed ? local_rng : tl_rng;
    auto p_take = [&](double value) { return 1.0 / (1.0 + std::exp(-(value - o.theta) / o.tau)); };

    // Pool of deals, weighted by the passes made so far by the other seats.
    const int pool_n = passes > 0 ? o.worlds * std::max(1, o.pool) : o.worlds;
    std::vector<std::array<int, kCards>> decks(static_cast<std::size_t>(pool_n));
    std::vector<double> wt(static_cast<std::size_t>(pool_n), 1.0);
    for (int w = 0; w < pool_n; ++w) {
        std::shuffle(hidden.begin(), hidden.end(), rng);
        auto& deck = decks[std::size_t(w)];
        std::size_t h = 0, m = 0;
        for (int k = 0; k < kCards; ++k) {
            if (k == 20) deck[std::size_t(k)] = blik;
            else if (mine_slot[std::size_t(k)]) deck[std::size_t(k)] = my_cards[m++];
            else deck[std::size_t(k)] = hidden[h++];
        }
        double lik = 1;
        for (int k = 0; k < passes; ++k) {
            const int i = k % kPlayers;
            if (i == my_i) continue;
            int sl[5];
            slotsOf(i, sl);
            Mask five = bit(blik);
            for (int x : sl) five |= bit(deck[std::size_t(x)]);
            if (k < kPlayers) {
                lik *= 1 - p_take(ruleHandValue(five, blik_suit));
            } else {
                for (int s = 0; s < kSuits; ++s)
                    if (s != blik_suit || cfg.round2_allow_blik_suit) lik *= 1 - p_take(ruleHandValue(five, s));
            }
        }
        wt[std::size_t(w)] = lik;
    }
    std::vector<std::size_t> use;
    if (pool_n > o.worlds) {
        double sum = 0;
        for (double x : wt) sum += x;
        const double step = sum / o.worlds;
        double u = std::uniform_real_distribution<double>(0.0, step)(rng), cum = wt[0];
        std::size_t j = 0;
        for (int k = 0; k < o.worlds; ++k, u += step) {
            while (u > cum && j + 1 < wt.size()) cum += wt[++j];
            use.push_back(j);
        }
    } else {
        for (int w = 0; w < pool_n; ++w) use.push_back(std::size_t(w));
    }

    std::vector<double> sums(options.size(), 0.0);
    for (std::size_t w : use) {
        for (std::size_t i = 0; i < options.size(); ++i) {
            Deal sim(cfg, dealer, decks[w]);
            for (int k = 0; k < passes; ++k) sim.bid(Bid::pass());
            sim.bid(options[i]);
            while (sim.phase() == Phase::kBid1 || sim.phase() == Phase::kBid2) sim.bid(ruleBid(sim, o.theta));
            while (sim.phase() == Phase::kPlay) sim.play(heuristicCard(sim, nullptr, o.rules));
            sums[i] += finishedValue(sim, teamOf(me), o.match);
        }
    }
    std::size_t best = 0;
    for (std::size_t i = 0; i < options.size(); ++i) {
        res.values.push_back({options[i], sums[i] / double(use.size())});
        if (sums[i] > sums[best]) best = i;
    }
    res.bid = options[best];
    return res;
}

Bid Rule2Agent::chooseBid(const Deal& d) {
    if (bidding_.worlds <= 0) return tableBid(d);
    RuleBidOptions o = bidding_;
    o.match = match_;
    if (seeded_) {
        o.seed = rng_();
        if (o.seed == 0) o.seed = 1;
    }
    return ruleBidSearch(d, o).bid;
}

// ---------------------------------------------------------------- factory

std::unique_ptr<Agent> makeRuleAgent(const std::string& spec, std::uint64_t seed) {
    const bool play = spec.rfind("ruleplay:", 0) == 0;
    if (!play && spec != "rule2" && spec.rfind("rule2:", 0) != 0) return nullptr;
    std::vector<std::string> parts;
    const std::string rest = play ? spec.substr(9) : (spec.size() > 6 ? spec.substr(6) : std::string());
    for (std::size_t start = 0; !rest.empty();) {
        const auto pos = rest.find(':', start);
        parts.push_back(rest.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    RulePlayOptions o;
    RuleBidOptions b;
    std::size_t first = 0;
    if (play) {
        if (parts.empty()) throw std::invalid_argument("ruleplay needs ruleplay:<worlds>");
        o.worlds = std::stoi(parts[0]);
        first = 1;
    }
    for (std::size_t i = first; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "noronde") o.rules.ronde = false;
        else if (p == "keepv1") o.rules.keep_v2 = false;
        else if (p == "keepv2") o.rules.keep_v2 = true;
        else if (p == "deftrump") o.rules.def_master_trump = true;
        else if (p == "nodeftrump") o.rules.def_master_trump = false;
        else if (p.rfind("simbid", 0) == 0) b.worlds = std::stoi(p.substr(6));
        else if (p == "nopartnerdraw") o.rules.partner_draw = false;
        else if (p.rfind("lowdraw", 0) == 0) o.rules.low_draw_min = std::stoi(p.substr(7));
        else if (p == "tenonace") o.rules.ten_on_ace = true;
        else if (p == "voiddump") o.rules.void_dump = true;
        else if (p == "force") o.rules.force_taker = true;
        else if (p == "effrank") o.rules.eff_rank = true;
        else if (p == "noeffrank") o.rules.eff_rank = false;
        else if (p == "keepv3") o.rules.keep_v3 = true;
        else if (p == "sparepartner") o.rules.spare_partner = true;
        else if (p == "nosparepartner") o.rules.spare_partner = false;
        else if (p == "partnervoid") o.rules.partner_void = true;
        else if (p == "nopartnervoid") o.rules.partner_void = false;
        else if (p == "nocount") o.rules.count_layouts = 0;
        else if (!play) throw std::invalid_argument("unknown rule2 option: " + p);
        else if (p == "noab") o.ab_handoff_trick = -1;
        else if (p.rfind("ab", 0) == 0) o.ab_handoff_trick = std::stoi(p.substr(2));
        else if (p == "argmax") o.z = -1;
        else if (p.rfind("z", 0) == 0) o.z = std::stod(p.substr(1));
        else if (p.rfind("m", 0) == 0) o.margin = std::stod(p.substr(1));
        else if (p == "nolik") o.lik_eps = 1.0;
        else if (p == "nobid") o.bid_tau = 0;
        else if (p.rfind("lik", 0) == 0) o.lik_eps = std::stod(p.substr(3));
        else if (p.rfind("bid", 0) == 0) o.bid_tau = std::stod(p.substr(3));
        else if (p.rfind("theta", 0) == 0) o.bid_theta = std::stod(p.substr(5));
        else if (p.rfind("pool", 0) == 0) o.pool = std::stoi(p.substr(4));
        else throw std::invalid_argument("unknown ruleplay option: " + p);
    }
    if (!play) return std::make_unique<Rule2Agent>(o.rules, b, spec, seed);
    return std::make_unique<RulePlayAgent>(o, b, spec, seed);
}

}  // namespace belote
