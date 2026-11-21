#include "bg/agent.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace bg {

// ================================================================== the rule policy

namespace {

struct Ctx {
    const Round& r;
    int me, team, declarer, trump;
    std::uint8_t contract;
    bool all, none;
    Mask hand, legal, played = 0, unseen;
    const Trick& t;

    explicit Ctx(const Round& rr)
        : r(rr), me(rr.toMove()), team(teamOf(me)), declarer(rr.contractPlayer()), trump(trumpSuit(rr.contract())),
          contract(rr.contract()), all(rr.contract() & kAllTrumps), none(rr.contract() & kNoTrumps),
          hand(rr.hand(me)), legal(rr.legalMoves()), unseen(0), t(rr.trick()) {
        for (const auto& pl : rr.plays()) played |= bit(pl.second.first);
        unseen = ~(hand | played);
    }

    bool trumpLike(int c) const { return all || (!none && suitOf(c) == trump); }
    int order(int c) const { return trumpLike(c) ? trumpOrder(c) : plainOrder(c); }
    int pts(int c) const { return cardValue(c, contract); }
    bool isTrump(int c) const { return !all && !none && suitOf(c) == trump; }
    int unseenIn(int s) const { return popcount(unseen & suitMask(s)); }
    bool master(int c) const {
        for (Mask m = unseen & suitMask(suitOf(c)); m; m &= m - 1)
            if (order(lowestCard(m)) > order(c)) return false;
        return true;
    }
    bool wouldWin(int c) const {
        Trick u = t;
        u.cards[u.n] = c;
        u.players[u.n] = me;
        ++u.n;
        return trickWinnerIndex(contract, u) == t.n;
    }
    // Could a later opponent beat `winning` (currently winning the trick)?
    bool threatened(int winning, int n_later, bool partner_last) const {
        if (n_later == 0 || (n_later == 1 && partner_last)) return false;
        const int led = suitOf(t.cards[0]);
        for (Mask m = unseen & suitMask(suitOf(winning)); m; m &= m - 1)
            if (order(lowestCard(m)) > order(winning)) return true;
        // A ruff (or overruff) from a seat out of the led suit.
        if (!all && !none && suitOf(winning) != trump && (unseen & suitMask(trump)) && unseenIn(led) <= 1) return true;
        if (isTrump(winning) && led != trump && unseenIn(led) <= 1) {
            for (Mask m = unseen & suitMask(trump); m; m &= m - 1)
                if (trumpOrder(lowestCard(m)) > trumpOrder(winning)) return true;
        }
        return false;
    }
    double keep(int c) const {
        if (trumpLike(c)) return 2.0 + order(c) + (master(c) ? 6.0 : 0.0);
        return master(c) ? 8.0 + 0.6 * pts(c) : 0.3 * pts(c);
    }
    double dumpCost(int c) const { return pts(c) + keep(c); }

    template <class F>
    int best(Mask m, F score) const {
        int b = kNoCard;
        double bs = -std::numeric_limits<double>::infinity();
        for (; m; m &= m - 1) {
            const int c = lowestCard(m);
            const double s = score(c);
            if (s > bs) {
                bs = s;
                b = c;
            }
        }
        return b;
    }
    int strongest(Mask m) const { return best(m, [&](int c) { return double(order(c)); }); }
    int weakest(Mask m) const { return best(m, [&](int c) { return -double(order(c)) - 0.01 * pts(c); }); }

    int lead() const {
        const bool attacking = team == teamOf(declarer);
        if (isTrumpContract()) {
            const Mask my_trumps = hand & suitMask(trump), out_trumps = unseen & suitMask(trump);
            if (attacking && my_trumps && out_trumps) {
                const int top = strongest(my_trumps);
                if (master(top)) return top;
                if (me == declarer && popcount(my_trumps) >= 3) return weakest(my_trumps);
            }
            // Side masters that should not be ruffed.
            const int cash = best(hand & ~suitMask(trump), [&](int c) {
                if (!master(c) || (out_trumps && unseenIn(suitOf(c)) < 3)) return -1e9;
                return double(pts(c));
            });
            if (cash != kNoCard && master(cash) && (!out_trumps || unseenIn(suitOf(cash)) >= 3)) return cash;
        } else {
            // No trumps / all trumps: cash the masters, richest first.
            const int cash = best(hand, [&](int c) { return master(c) ? double(pts(c)) + 1.0 : -1e9; });
            if (cash != kNoCard && master(cash)) return cash;
        }
        // Low from length, keeping guards of an exposed honour.
        return best(hand, [&](int c) {
            const int s = suitOf(c);
            const Mask in_suit = hand & suitMask(s);
            if (c != weakest(in_suit)) return -1e9;
            const int n = popcount(in_suit);
            double score = -1.5 * pts(c) + 0.6 * n;
            if (isTrump(c)) score -= attacking ? 3.0 : 12.0;
            // The second-best card of the suit still out over it: leading away leaves it bare.
            for (Mask m = in_suit & ~bit(c); m; m &= m - 1) {
                const int h = lowestCard(m);
                if (!master(h) && pts(h) >= 10 && n == 2) score -= 6.0;
            }
            return score;
        });
    }

    bool isTrumpContract() const { return !all && !none; }

    int follow() const {
        const int w = trickWinnerIndex(contract, t);
        const int win_card = t.cards[w];
        const bool partner_winning = t.players[w] == partnerOf(me);
        const int n_later = kPlayers - 1 - t.n;
        const bool partner_last = n_later == 1 && nextSeat(me) == partnerOf(me);  // never: partner is 2 seats away
        int trick_pts = 0;
        for (int i = 0; i < t.n; ++i) trick_pts += pts(t.cards[i]);
        // Later seats: me+1 (opponent), me+2 (partner), me+3 (opponent); an
        // opponent is still to play unless I am last or only partner follows.
        const bool opp_later = n_later >= 1;
        (void)partner_last;

        if (partner_winning) {
            if (!opp_later || !threatened(win_card, n_later, false)) {
                // Smear: points that are not worth keeping.
                return best(legal, [&](int c) { return pts(c) - keep(c) - (isTrump(c) ? 4.0 : 0.0); });
            }
            Mask sure = 0;
            for (Mask m = legal; m; m &= m - 1) {
                const int c = lowestCard(m);
                if (wouldWin(c) && !threatened(c, n_later, false)) sure |= bit(c);
            }
            if (sure && trick_pts >= 10) return best(sure, [&](int c) { return -keep(c); });
            return best(legal, [&](int c) { return -dumpCost(c); });
        }
        Mask winners = 0, sure = 0;
        for (Mask m = legal; m; m &= m - 1) {
            const int c = lowestCard(m);
            if (!wouldWin(c)) continue;
            winners |= bit(c);
            if (!threatened(c, n_later, false)) sure |= bit(c);
        }
        if (n_later == 0) {
            if (winners) return best(winners, [&](int c) { return -keep(c) + 0.01 * pts(c); });
            return best(legal, [&](int c) { return -dumpCost(c); });
        }
        if (sure) return best(sure, [&](int c) { return -keep(c) + 0.01 * pts(c); });
        if (winners && !(legal & ~winners)) return best(winners, [&](int c) { return -keep(c); });
        return best(legal, [&](int c) { return -dumpCost(c); });
    }
};

int margin(const RoundResult& res, int team) { return team == 0 ? res.sn - res.ew : res.ew - res.sn; }

}  // namespace

int rulePolicy(const Round& r) {
    if (r.decision() != Round::Decision::kPlay) throw std::logic_error("rulePolicy: not playing");
    const Mask legal = r.legalMoves();
    if (popcount(legal) == 1) return lowestCard(legal);
    const Ctx x(r);
    return x.t.n == 0 ? x.lead() : x.follow();
}

std::vector<Announce> chooseDeclarations(const std::vector<Announce>& offered) { return offered; }

// ================================================================== exact endgame

namespace {

struct Endgame {
    std::uint8_t contract;
    int declarer, hanging, trump;
    long nodes = 0;

    struct State {
        std::array<Mask, kPlayers> hands{};
        Trick t;
        int current = 0;
        std::array<Mask, 2> won{};
        std::array<int, 2> raw{};  // card points won, declarations, belotes, last 10
    };

    int terminal(const State& s) const {
        const RoundResult r = finalizeScore(contract, declarer, s.raw[0], s.raw[1], s.won[0] == 0, s.won[1] == 0, hanging);
        return r.sn - r.ew;
    }

    // Team-0 value with alpha-beta; team 0 maximises.
    int search(const State& s, int alpha, int beta) {
        ++nodes;
        const Mask legal = legalCards(s.hands[std::size_t(s.current)], contract, s.t);
        const bool maxing = teamOf(s.current) == 0;
        int best = maxing ? -1000000 : 1000000;
        for (Mask m = legal; m; m &= m - 1) {
            const int c = lowestCard(m);
            State n = s;
            const int v = apply(n, c, popcount(legal) > 1) ? terminal(n) : search(n, alpha, beta);
            if (maxing) {
                best = std::max(best, v);
                alpha = std::max(alpha, v);
            } else {
                best = std::min(best, v);
                beta = std::min(beta, v);
            }
            if (alpha >= beta) break;
        }
        return best;
    }

    // Plays c; true when the deal is over.
    bool apply(State& s, int c, bool chosen) const {
        const int p = s.current;
        if (chosen && beloteAllowed(s.hands[std::size_t(p)], contract, s.t, c)) s.raw[std::size_t(teamOf(p))] += 20;
        s.hands[std::size_t(p)] &= ~bit(c);
        s.t.cards[s.t.n] = c;
        s.t.players[s.t.n] = p;
        if (++s.t.n < kPlayers) {
            s.current = nextSeat(p);
            return false;
        }
        const int w = s.t.players[trickWinnerIndex(contract, s.t)];
        const int team = teamOf(w);
        for (int i = 0; i < kPlayers; ++i) {
            s.won[std::size_t(team)] |= bit(s.t.cards[i]);
            s.raw[std::size_t(team)] += cardValue(s.t.cards[i], contract);
        }
        s.t = Trick{};
        s.current = w;
        if (s.hands[std::size_t(w)] == 0) {
            s.raw[std::size_t(team)] += 10;
            return true;
        }
        return false;
    }
};

}  // namespace

EndgameResult solveEndgame(const Round& r, int team) {
    if (r.decision() != Round::Decision::kPlay || r.trickNumber() < 2)
        throw std::logic_error("solveEndgame: needs a card decision from trick 2 on");
    Endgame e{r.contract(), r.contractPlayer(), r.hangingBefore(), trumpSuit(r.contract())};
    Endgame::State s;
    for (int p = 0; p < kPlayers; ++p) s.hands[std::size_t(p)] = r.hand(p);
    s.t = r.trick();
    s.current = r.toMove();
    for (int tm = 0; tm < 2; ++tm) {
        s.won[std::size_t(tm)] = r.won(tm);
        for (Mask m = r.won(tm); m; m &= m - 1) s.raw[std::size_t(tm)] += cardValue(lowestCard(m), r.contract());
    }
    for (const Announce& a : r.announces())
        if (a.scored == 1) s.raw[std::size_t(teamOf(a.player))] += a.value();
    EndgameResult res;
    const Mask legal = r.legalMoves();
    const int sign = team == 0 ? 1 : -1;
    int best = -1000000;
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        Endgame::State n = s;
        const int v0 = e.apply(n, c, popcount(legal) > 1) ? e.terminal(n) : e.search(n, -1000000, 1000000);
        res.move_values.push_back({c, sign * v0});
        best = std::max(best, sign * v0);
    }
    res.value = best;
    res.nodes = e.nodes;
    return res;
}

// ================================================================== play-outs and search

double playOut(Round r, int team, int endgame_tricks) {
    while (!r.finished()) {
        switch (r.decision()) {
            case Round::Decision::kAnnounce:
                r.declare(chooseDeclarations(r.offeredAnnounces()));
                break;
            case Round::Decision::kPlay:
                if (endgame_tricks > 0 && r.trickNumber() >= 2 && r.trickNumber() > kTricks - endgame_tricks)
                    return solveEndgame(r, team).value;
                r.play(rulePolicy(r), true);
                break;
            default:
                throw std::logic_error("playOut: unexpected decision");
        }
    }
    return margin(r.result(), team);
}

namespace {

std::mt19937_64& threadRng() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

// The others' past plays the rule policy would not have made, in this world.
int policyMismatches(const PublicState& ps, const Hands& hands) {
    Hands start{};
    for (int p = 0; p < kPlayers; ++p) start[std::size_t(p)] = hands[std::size_t(p)] | ps.playedBy(p);
    int mismatches = 0;
    const PlayHook hook = [&](const Round& r, const PublicPlay& pl) {
        if (pl.seat != ps.me && popcount(r.legalMoves()) > 1 && rulePolicy(r) != pl.card) ++mismatches;
    };
    replayDeal(ps, start, &hook);
    return mismatches;
}

}  // namespace

SearchResult searchCard(const PublicState& ps, const SearchOptions& o) {
    SearchResult res;
    const int team = teamOf(ps.me);
    DealSampler sampler(ps);
    std::mt19937_64 local(o.seed);
    std::mt19937_64& rng = o.seed ? local : threadRng();

    const bool weighted = o.lik_eps < 1.0 && !ps.plays.empty();
    const int n_draw = weighted ? o.worlds * std::max(1, o.pool) : o.worlds;
    std::vector<Hands> drawn;
    std::vector<Round> bases;
    std::vector<double> wt;
    for (int i = 0; i < n_draw; ++i) {
        auto h = sampler.sample(rng);
        if (!h) break;
        auto r = sampler.replay(*h);
        if (!r) continue;
        drawn.push_back(*h);
        bases.push_back(std::move(*r));
        wt.push_back(weighted ? std::pow(o.lik_eps, policyMismatches(ps, *h)) : 1.0);
    }
    if (bases.empty()) throw std::runtime_error("searchCard: no consistent world");
    if (weighted && int(bases.size()) > o.worlds) {
        double sum = 0;
        for (double x : wt) sum += x;
        const double step = sum / o.worlds;
        double u = std::uniform_real_distribution<double>(0.0, step)(rng), cum = wt[0];
        std::size_t j = 0;
        std::vector<Round> picked;
        for (int k = 0; k < o.worlds; ++k, u += step) {
            while (u > cum && j + 1 < wt.size()) cum += wt[++j];
            picked.push_back(bases[j]);
        }
        bases.swap(picked);
    }
    const std::size_t nw = bases.size();
    res.worlds = int(nw);

    const Round& any = bases[0];
    const Mask legal = any.legalMoves();
    res.rule_card = rulePolicy(any);
    res.card = res.rule_card;
    if (popcount(legal) == 1) {
        res.values.push_back({res.card, 0.0});
        return res;
    }
    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    const std::size_t nm = moves.size();
    std::vector<double> val(nm * nw, 0.0);
    const bool exact = any.trickNumber() >= o.root_exact_trick && any.trickNumber() >= 2;
    for (std::size_t w = 0; w < nw; ++w) {
        if (exact) {
            const EndgameResult er = solveEndgame(bases[w], team);
            for (std::size_t i = 0; i < nm; ++i)
                for (const auto& [c, v] : er.move_values)
                    if (c == moves[i]) val[i * nw + w] = v;
            continue;
        }
        for (std::size_t i = 0; i < nm; ++i) {
            Round sim = bases[w];
            sim.play(moves[i], true);
            val[i * nw + w] = playOut(std::move(sim), team, o.endgame_tricks);
        }
    }
    std::vector<double> mean(nm, 0.0);
    std::size_t best = 0, rule_i = 0;
    for (std::size_t i = 0; i < nm; ++i) {
        for (std::size_t w = 0; w < nw; ++w) mean[i] += val[i * nw + w] / double(nw);
        res.values.push_back({moves[i], mean[i]});
        if (mean[i] > mean[best]) best = i;
        if (moves[i] == res.rule_card) rule_i = i;
    }
    if (best == rule_i) return res;
    if (o.z < 0) {
        res.card = moves[best];
        res.overruled = true;
        return res;
    }
    const double m = mean[best] - mean[rule_i];
    double var = 0;
    for (std::size_t w = 0; w < nw; ++w) {
        const double d = val[best * nw + w] - val[rule_i * nw + w] - m;
        var += d * d;
    }
    const double se = std::sqrt(std::max(var / double(nw), 1e-12) / double(nw));
    if (m > o.margin && m > o.z * se) {
        res.card = moves[best];
        res.overruled = true;
    }
    return res;
}

}  // namespace bg
