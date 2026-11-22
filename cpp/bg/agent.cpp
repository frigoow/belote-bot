#include "bg/agent.h"
#include "cardbot/search.h"
#include "bg/match_equity.h"
#include "bg/rule_policy.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace bg {

namespace {

int margin(const RoundResult& res, int team) { return team == 0 ? res.sn - res.ew : res.ew - res.sn; }

}  // namespace

std::vector<Announce> chooseDeclarations(const std::vector<Announce>& offered) { return offered; }

// ================================================================== exact endgame (TT + pruning)

namespace {

constexpr int kInf = 1000000;

enum class Bound : std::uint8_t { kNone = 0, kExact = 1, kLower = 2, kUpper = 3 };

struct TTEntry {
    std::uint64_t key = 0;
    std::uint32_t gen = 0;
    std::int16_t score = 0;
    std::int8_t best_move = -1;
    Bound bound = Bound::kNone;
};

struct TTStorage {
    std::vector<TTEntry> table;
    std::uint32_t gen = 0;

    void prepare(std::size_t size) {
        if (table.size() < size) {
            table.assign(size, TTEntry{});
            gen = 1;
        } else if (++gen == 0) {
            std::fill(table.begin(), table.end(), TTEntry{});
            gen = 1;
        }
    }
};

TTStorage& threadBgTT() {
    thread_local TTStorage st;
    return st;
}

inline std::uint64_t mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct BgZobrist {
    std::uint64_t card[kPlayers][kCards]{};
    std::uint64_t leader[kPlayers]{};
    std::uint64_t capot[4]{};

    BgZobrist() {
        std::uint64_t x = 0x9E3779B97F4A7C15ull;
        auto next = [&]() {
            x += 0x9E3779B97F4A7C15ull;
            return mix64(x);
        };
        for (int p = 0; p < kPlayers; ++p) {
            leader[p] = next();
            for (int c = 0; c < kCards; ++c) card[p][c] = next();
        }
        for (int i = 0; i < 4; ++i) capot[i] = next();
    }
};

const BgZobrist& bgZobrist() {
    static const BgZobrist z;
    return z;
}

// Strips strictly equivalent 0-point cards in the same suit (adjacent in
// strength order with no unplayed card between them; Belot K/Q are non-zero so
// 0-point cards never affect Belot or trick scoring).
Mask pruneEquivalent(Mask legal, Mask in_play, std::uint8_t contract) {
    const bool all = contract & kAllTrumps;
    const bool none = contract & kNoTrumps;
    const int trump = trumpSuit(contract);
    // Trump order: 7 8 Q K 10 A 9 J (only 7, 8 are 0-pt).
    // Plain order: 7 8 9 J Q K 10 A (7, 8, 9 are 0-pt).
    Mask keep = legal;
    for (int s = 0; s < 4; ++s) {
        const Mask ls = legal & suitMask(s);
        if (popcount(ls) <= 1) continue;
        const bool tlike = all || (!none && s == trump);
        const int zero_ranks[3] = {belote::k7, belote::k8, belote::k9};
        const int nz = tlike ? 2 : 3;
        int prev_kept = -1;
        for (int i = 0; i < nz; ++i) {
            const int c = cardOf(s, zero_ranks[i]);
            if (ls & bit(c)) {
                if (prev_kept >= 0) keep &= ~bit(c);
                else prev_kept = i;
            } else if (in_play & bit(c)) {
                // Live 0-point card in another hand breaks adjacency!
                prev_kept = -1;
            }
        }
    }
    return keep;
}

struct Endgame {
    std::uint8_t contract;
    int declarer, hanging, trump;
    bool all, none;
    MatchContext match_ctx;
    const BgZobrist& z;
    TTEntry* tt;
    std::size_t tt_mask;
    std::uint32_t gen;
    long nodes = 0;

    Endgame(std::uint8_t c, int decl, int hang, int tt_log2, const MatchContext& mc)
        : contract(c), declarer(decl), hanging(hang), trump(trumpSuit(c)), all((c & kAllTrumps) != 0),
          none((c & kNoTrumps) != 0), match_ctx(mc), z(bgZobrist()) {
        const std::size_t sz = std::size_t(1) << std::max(8, std::min(20, tt_log2));
        TTStorage& st = threadBgTT();
        st.prepare(sz);
        tt = st.table.data();
        tt_mask = sz - 1;
        gen = st.gen;
    }

    struct State {
        std::array<Mask, kPlayers> hands{};
        Trick t;
        int current = 0;
        std::array<Mask, 2> won{};
        std::array<int, 2> raw{};  // card points won, declarations, belotes, last 10
        std::uint64_t hands_hash = 0;
    };

    int terminal(const State& s) const {
        const RoundResult r = finalizeScore(contract, declarer, s.raw[0], s.raw[1], s.won[0] == 0, s.won[1] == 0, hanging);
        if (match_ctx.enabled) {
            const double eq0 = MatchEquityTable::instance().postDealEquity(
                match_ctx.sn_before, match_ctx.ew_before, match_ctx.first_before, contract, r, 0);
            // Zero-sum tie-breaker by deal margin when match equity saturates at 0.0 or 1.0!
            // v0 in [-300, 9900], exact team-1 complement is 9600 - v0.
            const int tie = std::clamp(r.sn - r.ew, -150, 150) * 2;
            return int(std::lround(eq0 * 9600.0)) + tie;
        }
        return r.sn - r.ew;
    }

    int movePriority(int c, const State& s, int tt_move) const {
        if (c == tt_move) return 10000;
        int p = cardValue(c, contract) * 4;
        if (!all && !none && suitOf(c) == trump) p += 60 + trumpOrder(c) * 2;
        else if (all) p += trumpOrder(c) * 2;
        else p += plainOrder(c) * 2;
        if (s.t.n > 0) {
            Trick u = s.t;
            u.cards[u.n] = c;
            u.players[u.n] = s.current;
            ++u.n;
            if (trickWinnerIndex(contract, u) == s.t.n) p += 200;
        }
        return p;
    }

    // Team-0 value with alpha-beta + Zobrist TT + 0-pt equivalence pruning.
    int search(const State& s, int alpha, int beta) {
        ++nodes;
        const bool maxing = teamOf(s.current) == 0;
        int tt_move = -1;
        std::uint64_t tt_key = 0;
        const bool use_tt = (s.t.n == 0);
        if (use_tt) {
            const int cm = (s.won[0] == 0 ? 1 : 0) | (s.won[1] == 0 ? 2 : 0);
            tt_key = s.hands_hash ^ z.leader[s.current] ^ z.capot[cm] ^
                     mix64(std::uint64_t(std::uint32_t(s.raw[0])) | (std::uint64_t(std::uint32_t(s.raw[1])) << 32));
            const TTEntry& e = tt[tt_key & tt_mask];
            if (e.gen == gen && e.bound != Bound::kNone && e.key == tt_key) {
                tt_move = e.best_move;
                if (e.bound == Bound::kExact) return e.score;
                if (e.bound == Bound::kLower) alpha = std::max(alpha, int(e.score));
                else if (e.bound == Bound::kUpper) beta = std::min(beta, int(e.score));
                if (alpha >= beta) return e.score;
            }
        }

        const Mask legal = legalCards(s.hands[std::size_t(s.current)], contract, s.t);
        const bool chosen = popcount(legal) > 1;
        Mask in_play = s.hands[0] | s.hands[1] | s.hands[2] | s.hands[3];
        for (int i = 0; i < s.t.n; ++i) in_play |= bit(s.t.cards[i]);
        const Mask pruned = chosen ? pruneEquivalent(legal, in_play, contract) : legal;

        int moves[8]{};
        int prio[8]{};
        int nm = 0;
        for (Mask m = pruned; m; m &= m - 1) {
            const int c = lowestCard(m);
            moves[nm] = c;
            prio[nm] = movePriority(c, s, tt_move);
            ++nm;
        }
        // Simple insertion sort (nm <= 8).
        for (int i = 1; i < nm; ++i) {
            const int mc = moves[i], mp = prio[i];
            int j = i - 1;
            while (j >= 0 && prio[j] < mp) {
                moves[j + 1] = moves[j];
                prio[j + 1] = prio[j];
                --j;
            }
            moves[j + 1] = mc;
            prio[j + 1] = mp;
        }

        const int orig_alpha = alpha, orig_beta = beta;
        int best = maxing ? -kInf : kInf;
        int best_card = moves[0];
        for (int i = 0; i < nm; ++i) {
            const int c = moves[i];
            State n = s;
            const int v = apply(n, c, chosen) ? terminal(n) : search(n, alpha, beta);
            if (maxing) {
                if (v > best) {
                    best = v;
                    best_card = c;
                }
                alpha = std::max(alpha, v);
            } else {
                if (v < best) {
                    best = v;
                    best_card = c;
                }
                beta = std::min(beta, v);
            }
            if (alpha >= beta) break;
        }

        if (use_tt && best >= -32000 && best <= 32000) {
            TTEntry& e = tt[tt_key & tt_mask];
            e.key = tt_key;
            e.gen = gen;
            e.score = std::int16_t(best);
            e.best_move = std::int8_t(best_card);
            if (best <= orig_alpha) e.bound = Bound::kUpper;
            else if (best >= orig_beta) e.bound = Bound::kLower;
            else e.bound = Bound::kExact;
        }
        return best;
    }

    // Plays c; true when the deal is over.
    bool apply(State& s, int c, bool chosen) const {
        const int p = s.current;
        if (chosen && beloteAllowed(s.hands[std::size_t(p)], contract, s.t, c)) s.raw[std::size_t(teamOf(p))] += 20;
        s.hands[std::size_t(p)] &= ~bit(c);
        s.hands_hash ^= z.card[p][c];
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

    State initState(const Round& r) const {
        State s;
        for (int p = 0; p < kPlayers; ++p) {
            s.hands[std::size_t(p)] = r.hand(p);
            for (Mask m = s.hands[std::size_t(p)]; m; m &= m - 1) s.hands_hash ^= z.card[p][lowestCard(m)];
        }
        s.t = r.trick();
        s.current = r.toMove();
        for (int tm = 0; tm < 2; ++tm) {
            s.won[std::size_t(tm)] = r.won(tm);
            for (Mask m = r.won(tm); m; m &= m - 1) s.raw[std::size_t(tm)] += cardValue(lowestCard(m), r.contract());
        }
        for (const Announce& a : r.announces())
            if (a.scored == 1) s.raw[std::size_t(teamOf(a.player))] += a.value();
        return s;
    }
};

}  // namespace

EndgameResult solveEndgame(const Round& r, int team, const MatchContext& match_ctx) {
    if (r.decision() != Round::Decision::kPlay || r.trickNumber() < 2)
        throw std::logic_error("solveEndgame: needs a card decision from trick 2 on");
    const int tt_log2 = r.trickNumber() <= 4 ? 18 : 14;
    Endgame e(r.contract(), r.contractPlayer(), r.hangingBefore(), tt_log2, match_ctx);
    const Endgame::State s = e.initState(r);

    const Mask legal = r.legalMoves();
    const bool chosen = popcount(legal) > 1;
    Mask in_play = s.hands[0] | s.hands[1] | s.hands[2] | s.hands[3];
    for (int i = 0; i < s.t.n; ++i) in_play |= bit(s.t.cards[i]);
    const Mask unique_legal = chosen ? pruneEquivalent(legal, in_play, r.contract()) : legal;

    const bool maxing_team = (teamOf(r.toMove()) == team);
    int score_by_card[kCards];
    std::fill(std::begin(score_by_card), std::end(score_by_card), -kInf);
    int best = maxing_team ? -kInf : kInf;
    for (Mask m = unique_legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        Endgame::State n = s;
        const int v0 = e.apply(n, c, chosen) ? e.terminal(n) : e.search(n, -kInf, kInf);
        const int v = (team == 0) ? v0 : (match_ctx.enabled ? (9600 - v0) : -v0);
        score_by_card[c] = v;
        best = maxing_team ? std::max(best, v) : std::min(best, v);
    }

    EndgameResult res;
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        int v = score_by_card[c];
        if (v == -kInf) {
            const int su = suitOf(c);
            if (score_by_card[cardOf(su, belote::k7)] != -kInf) v = score_by_card[cardOf(su, belote::k7)];
            else if (score_by_card[cardOf(su, belote::k8)] != -kInf) v = score_by_card[cardOf(su, belote::k8)];
            else v = score_by_card[cardOf(su, belote::k9)];
        }
        res.move_values.push_back({c, v});
    }
    res.value = best;
    res.nodes = e.nodes;
    return res;
}

int solveEndgameValue(const Round& r, int team, const MatchContext& match_ctx) {
    if (r.decision() != Round::Decision::kPlay || r.trickNumber() < 2)
        throw std::logic_error("solveEndgameValue: needs a card decision from trick 2 on");
    const int tt_log2 = r.trickNumber() <= 4 ? 17 : 12;
    Endgame e(r.contract(), r.contractPlayer(), r.hangingBefore(), tt_log2, match_ctx);
    const Endgame::State s = e.initState(r);
    const int v0 = e.search(s, -kInf, kInf);
    return (team == 0) ? v0 : (match_ctx.enabled ? (9600 - v0) : -v0);
}

// ================================================================== play-outs and search

double playOut(Round r, int team, int endgame_tricks, const BgFastPolicyModel* fast_policy,
               const MatchContext& match_ctx) {
    const bool use_fast = fast_policy != nullptr && fast_policy->loaded();
    while (!r.finished()) {
        switch (r.decision()) {
            case Round::Decision::kAnnounce:
                r.declare(chooseDeclarations(r.offeredAnnounces()));
                break;
            case Round::Decision::kPlay:
                if (endgame_tricks > 0 && r.trickNumber() >= 2 && r.trickNumber() > kTricks - endgame_tricks) {
                    const int v = solveEndgameValue(r, team, match_ctx);
                    return match_ctx.enabled ? double(v) / 96.0 : double(v);
                }
                r.play(use_fast ? fast_policy->chooseCard(r) : rulePolicy(r), true);
                break;
            default:
                throw std::logic_error("playOut: unexpected decision");
        }
    }
    if (match_ctx.enabled) {
        const double eq = MatchEquityTable::instance().postDealEquity(
            match_ctx.sn_before, match_ctx.ew_before, match_ctx.first_before, r.contract(), r.result(), team);
        const int m = std::clamp(margin(r.result(), team), -150, 150);
        return 100.0 * eq + double(m) * (2.0 / 96.0);
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

// Soft play-history log-likelihood under BgFastPolicyModel for the other seats' past plays.
double fastPlayLogLik(const PublicState& ps, const Hands& hands, const BgFastPolicyModel& fp, double temp) {
    Hands start{};
    for (int p = 0; p < kPlayers; ++p) start[std::size_t(p)] = hands[std::size_t(p)] | ps.playedBy(p);
    double ll = 0.0;
    const PlayHook hook = [&](const Round& r, const PublicPlay& pl) {
        if (pl.seat != ps.me && popcount(r.legalMoves()) > 1) {
            ll += fp.legalLogProb(publicState(r, pl.seat), pl.card, temp);
        }
    };
    replayDeal(ps, start, &hook);
    return ll;
}

}  // namespace

namespace {

// What cardbot::drawWorlds asks of the game: Belot's sampler, its replay of
// the public record, the likelihood weights and the belief net's prior.
struct BelotWorlds {
    const PublicState& ps;
    const DealSampler& sampler;
    bool use_belief;
    const std::array<std::array<double, 3>, kCards>& belief_weights;
    bool is_weighted;
    std::function<double(const Hands&)> weight;

    double count() const { return sampler.count(); }
    std::vector<Hands> enumerate(std::uint64_t cap) const { return sampler.enumerate(cap); }
    std::optional<Round> replay(const Hands& h) const { return sampler.replay(h); }
    std::optional<Hands> draw(std::mt19937_64& rng) const {
        auto h = use_belief ? sampler.sampleWeighted(rng, belief_weights) : sampler.sample(rng);
        if (!h && use_belief) h = sampler.sample(rng);
        return h;
    }
    std::optional<Hands> drawHard(std::mt19937_64& rng) const { return sampler.sample(rng, 1000); }
    bool weighted() const { return is_weighted; }
    double logWeight(const Hands& h) const { return weight(h); }
    bool hasPrior() const { return use_belief; }
    // the belief net's marginals as a product prior on enumerated worlds
    void addLogPrior(const Hands& h, double& lw) const {
        const auto& others = sampler.others();
        for (int k = 0; k < 3; ++k) {
            for (Mask m = h[std::size_t(others[std::size_t(k)])]; m; m &= m - 1) {
                const int c = lowestCard(m);
                lw += std::log(std::max(belief_weights[std::size_t(c)][std::size_t(k)], 1e-6));
            }
        }
    }
};

}  // namespace

SearchResult searchCard(const PublicState& ps, const SearchOptions& o) {
    SearchResult res;
    const int team = teamOf(ps.me);
    DealSampler sampler(ps);
    std::mt19937_64 local(o.seed);
    std::mt19937_64& rng = o.seed ? local : threadRng();

    // Resolve contract-specific specialist models (Suit / NT / AT) with fallback to unified defaults.
    const NeuralSuite& spec = (ps.contract & kAllTrumps)   ? o.at_models
                              : (ps.contract & kNoTrumps) ? o.nt_models
                                                          : o.suit_models;
    const BgFastPolicyModel* fast_ptr = (spec.fast && spec.fast->loaded()) ? spec.fast : o.fast_policy;
    const BgBeliefModel* belief_ptr = (spec.belief && spec.belief->loaded()) ? spec.belief : o.belief_model;
    const BgQGRUModel* q_ptr = (spec.q && spec.q->loaded()) ? spec.q : o.q_model;

    const bool use_fast = fast_ptr != nullptr && fast_ptr->loaded();
    const bool use_belief = belief_ptr != nullptr && belief_ptr->loaded();
    const bool use_q = q_ptr != nullptr && q_ptr->loaded();

    MatchContext match_ctx;
    if (o.match_equity && (ps.sn > 0 || ps.ew > 0 || ps.hanging > 0)) {
        match_ctx.enabled = true;
        match_ctx.sn_before = ps.sn;
        match_ctx.ew_before = ps.ew;
        match_ctx.first_before = ps.first;
    }

    const bool rule_play_weight = o.lik_eps < 1.0 && !ps.plays.empty();
    const bool fast_play_weight = use_fast && o.fast_lik_weight > 0.0 && !ps.plays.empty();
    const bool bid_weight = o.auction_weight > 0.0 && !ps.bids.empty();
    const bool weighted = rule_play_weight || fast_play_weight || bid_weight;

    // Precompute Belief-DP per-card weights once at the root (~0.14 ms) if BgBeliefModel is loaded.
    std::array<std::array<double, 3>, kCards> belief_weights{};
    if (use_belief) {
        belief_weights = belief_ptr->samplerWeights(ps, sampler, o.belief_temp);
    }

    // The worlds (cardbot-framework): enumerated when few, else drawn, weighted, resampled, deduplicated.
    const BelotWorlds source{ps, sampler, use_belief, belief_weights, weighted, [&](const Hands& h) -> double {
        double lw = 0.0;
        if (rule_play_weight) lw += double(policyMismatches(ps, h)) * std::log(std::max(o.lik_eps, 1e-12));
        if (fast_play_weight) lw += o.fast_lik_weight * fastPlayLogLik(ps, h, *fast_ptr, o.fast_lik_temp);
        if (bid_weight) lw += o.auction_weight * auctionLogLik(ps, h);
        return lw;
    }};
    cardbot::WorldOptions wo;
    wo.worlds = o.worlds;
    wo.pool = o.pool;
    wo.shuffle = o.race;
    cardbot::WorldSet<Round> ws = cardbot::drawWorlds<Round>(source, wo, rng);
    const std::vector<Round>& bases = ws.states;
    const std::size_t nw = bases.size();
    const Round& any = bases[0];
    const Mask legal = any.legalMoves();

    // Prior card & optional root Q-GRU evaluation (once at root).
    std::vector<std::pair<int, double>> q_root;
    if (use_q) q_root = q_ptr->evalLegal(ps, &sampler.facts());

    if (use_q && !q_root.empty()) {
        int best_q_card = q_root[0].first;
        double best_qv = q_root[0].second;
        for (const auto& [c, v] : q_root) {
            if (v > best_qv) {
                best_qv = v;
                best_q_card = c;
            }
        }
        res.rule_card = best_q_card;
    } else if (use_fast) {
        res.rule_card = fast_ptr->chooseCard(ps, &sampler.facts());
    } else {
        res.rule_card = rulePolicy(any);
    }
    res.card = res.rule_card;
    if (popcount(legal) == 1) {
        res.values.push_back({res.card, 0.0});
        res.worlds = int(nw);
        return res;
    }

    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    const std::size_t nm = moves.size();
    const bool exact = any.trickNumber() >= o.root_exact_trick && any.trickNumber() >= 2;

    // Each move on each world: exact from the root late in the deal, else a play-out.
    auto evaluate = [&](std::size_t w, const std::vector<char>& alive, std::vector<double>& val) {
        if (exact) {
            const EndgameResult er = solveEndgame(bases[w], team, match_ctx);
            const double scale = match_ctx.enabled ? (1.0 / 96.0) : 1.0;
            for (std::size_t i = 0; i < nm; ++i)
                for (const auto& [c, v] : er.move_values)
                    if (c == moves[i]) val[i * nw + w] = double(v) * scale;
        } else {
            for (std::size_t i = 0; i < nm; ++i) {
                if (!alive[i]) continue;
                Round sim = bases[w];
                sim.play(moves[i], true);
                val[i * nw + w] = playOut(std::move(sim), team, o.endgame_tricks, fast_ptr, match_ctx);
            }
        }
    };

    // Optional Root Q-GRU Blend with linear late-deal taper (0 when exact root AB solves the trick!).
    cardbot::QBlend qb;
    qb.weight = (use_q && o.q_blend > 0.0 && !exact && any.trickNumber() < o.q_taper_trick)
                    ? o.q_blend * (1.0 - double(any.trickNumber() - 1) / double(std::max(1, o.q_taper_trick - 1)))
                    : 0.0;
    if (qb.weight > 0.0 && !q_root.empty()) {
        qb.q.assign(nm, 0.0);
        for (std::size_t i = 0; i < nm; ++i)
            for (const auto& [c, v] : q_root)
                if (c == moves[i]) qb.q[i] = v;
        qb.center = match_ctx.enabled;
        qb.slope = match_ctx.enabled
                       ? 100.0 * MatchEquityTable::instance().marginSlope(
                                     team == 0 ? ps.sn : ps.ew, team == 0 ? ps.ew : ps.sn, ps.hanging,
                                     teamOf(ps.first) == team)
                       : 1.0;
    }

    cardbot::RaceOptions ro;
    ro.race = o.race;
    ro.race_min = o.race_min;
    ro.race_step = o.race_step;
    ro.race_z = o.race_z;
    ro.z = o.z;
    ro.margin = o.margin;
    const auto rr = cardbot::raceMoves(moves, ws.weights, ws.enumerated, exact, res.rule_card, evaluate, qb, ro);
    res.card = rr.move;
    res.overruled = rr.overruled;
    res.worlds = rr.worlds;
    res.values = rr.values;
    return res;
}

}  // namespace bg
