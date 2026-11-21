#include "bg/agent.h"
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

SearchResult searchCard(const PublicState& ps, const SearchOptions& o) {
    SearchResult res;
    const int team = teamOf(ps.me);
    DealSampler sampler(ps);
    std::mt19937_64 local(o.seed);
    std::mt19937_64& rng = o.seed ? local : threadRng();

    const bool use_fast = o.fast_policy != nullptr && o.fast_policy->loaded();
    const bool use_belief = o.belief_model != nullptr && o.belief_model->loaded();
    const bool use_q = o.q_model != nullptr && o.q_model->loaded();

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

    // Precompute Belief-DP per-card weights once at the root (~1.2 ms) if BgBeliefModel is loaded.
    std::array<std::array<double, 3>, kCards> belief_weights{};
    if (use_belief) {
        belief_weights = o.belief_model->samplerWeights(ps, sampler, o.belief_temp);
    }

    auto logWeight = [&](const Hands& h) -> double {
        double lw = 0.0;
        if (rule_play_weight) lw += double(policyMismatches(ps, h)) * std::log(std::max(o.lik_eps, 1e-12));
        if (fast_play_weight) lw += o.fast_lik_weight * fastPlayLogLik(ps, h, *o.fast_policy, o.fast_lik_temp);
        if (bid_weight) lw += o.auction_weight * auctionLogLik(ps, h);
        return lw;
    };

    std::vector<Round> bases;
    std::vector<double> wts;
    bool enumerated = false;

    const double total_worlds = sampler.count();
    if (total_worlds > 0.0 && total_worlds <= double(std::max(1, o.worlds))) {
        const auto all = sampler.enumerate(std::uint64_t(std::max(1, o.worlds)));
        if (!all.empty()) {
            std::vector<double> lws;
            lws.reserve(all.size());
            bases.reserve(all.size());
            const auto& others = sampler.others();
            for (const Hands& h : all) {
                auto r = sampler.replay(h);
                if (!r) continue;
                double lw = weighted ? logWeight(h) : 0.0;
                if (use_belief) {
                    // Multiply by BeliefNet marginal product prior on exact enumerated worlds.
                    for (int k = 0; k < 3; ++k) {
                        for (Mask m = h[std::size_t(others[std::size_t(k)])]; m; m &= m - 1) {
                            const int c = lowestCard(m);
                            lw += std::log(std::max(belief_weights[std::size_t(c)][std::size_t(k)], 1e-6));
                        }
                    }
                }
                bases.push_back(std::move(*r));
                lws.push_back(lw);
            }
            if (!bases.empty()) {
                enumerated = true;
                double top = *std::max_element(lws.begin(), lws.end());
                wts.resize(bases.size());
                for (std::size_t i = 0; i < bases.size(); ++i)
                    wts[i] = (weighted || use_belief) ? std::exp(std::max(-40.0, lws[i] - top)) : 1.0;
            }
        }
    }

    if (!enumerated) {
        const int n_draw = weighted ? o.worlds * std::max(1, o.pool) : o.worlds;
        std::vector<Hands> drawn;
        std::vector<Round> pool_bases;
        std::vector<double> lws;
        drawn.reserve(std::size_t(n_draw));
        pool_bases.reserve(std::size_t(n_draw));
        lws.reserve(std::size_t(n_draw));
        for (int i = 0; i < n_draw; ++i) {
            auto h = use_belief ? sampler.sampleWeighted(rng, belief_weights) : sampler.sample(rng);
            if (!h) break;
            auto r = sampler.replay(*h);
            if (!r) continue;
            drawn.push_back(*h);
            pool_bases.push_back(std::move(*r));
            lws.push_back(weighted ? logWeight(*h) : 0.0);
        }
        if (pool_bases.empty()) throw std::runtime_error("searchCard: no consistent world");

        std::vector<std::size_t> picked;
        if (weighted) {
            const int count = std::min(o.worlds, int(pool_bases.size()));
            const double top = *std::max_element(lws.begin(), lws.end());
            std::vector<double> pw(pool_bases.size());
            double sum = 0.0;
            for (std::size_t i = 0; i < pool_bases.size(); ++i) {
                pw[i] = std::exp(std::max(-40.0, lws[i] - top));
                sum += pw[i];
            }
            const double step = sum / double(count);
            double u = std::uniform_real_distribution<double>(0.0, step)(rng);
            double cum = pw[0];
            std::size_t j = 0;
            picked.reserve(std::size_t(count));
            for (int k = 0; k < count; ++k, u += step) {
                while (u > cum && j + 1 < pw.size()) cum += pw[++j];
                picked.push_back(j);
            }
        } else {
            picked.resize(pool_bases.size());
            for (std::size_t i = 0; i < pool_bases.size(); ++i) picked[i] = i;
        }

        // Weight-first deduplication: evaluate each unique sampled world once.
        std::sort(picked.begin(), picked.end(),
                  [&](std::size_t a, std::size_t b) { return drawn[a] < drawn[b]; });
        bases.reserve(picked.size());
        wts.reserve(picked.size());
        for (std::size_t k = 0; k < picked.size(); ++k) {
            const std::size_t idx = picked[k];
            if (k > 0 && drawn[idx] == drawn[picked[k - 1]]) {
                wts.back() += 1.0;
            } else {
                bases.push_back(std::move(pool_bases[idx]));
                wts.push_back(1.0);
            }
        }
        // Shuffle deduplicated worlds before CRN candidate racing to remove lexicographic order bias!
        if (o.race && bases.size() > 1) {
            std::vector<std::size_t> perm(bases.size());
            for (std::size_t i = 0; i < perm.size(); ++i) perm[i] = i;
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<Round> sb;
            std::vector<double> sw;
            sb.reserve(bases.size());
            sw.reserve(wts.size());
            for (std::size_t idx : perm) {
                sb.push_back(std::move(bases[idx]));
                sw.push_back(wts[idx]);
            }
            bases.swap(sb);
            wts.swap(sw);
        }
    }

    const std::size_t nw = bases.size();
    const Round& any = bases[0];
    const Mask legal = any.legalMoves();

    // Prior card & optional root Q-GRU evaluation (once at root).
    std::vector<std::pair<int, double>> q_root;
    if (use_q) q_root = o.q_model->evalLegal(ps, &sampler.facts());

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
        res.rule_card = o.fast_policy->chooseCard(ps, &sampler.facts());
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
    std::vector<double> val(nm * nw, 0.0);
    const bool exact = any.trickNumber() >= o.root_exact_trick && any.trickNumber() >= 2;

    std::vector<char> alive(nm, 1);
    std::size_t w_eval = 0;

    if (exact || !o.race || nw <= std::size_t(std::max(1, o.race_min)) || nm <= 2) {
        for (std::size_t w = 0; w < nw; ++w) {
            if (exact) {
                const EndgameResult er = solveEndgame(bases[w], team, match_ctx);
                const double scale = match_ctx.enabled ? (1.0 / 96.0) : 1.0;
                for (std::size_t i = 0; i < nm; ++i)
                    for (const auto& [c, v] : er.move_values)
                        if (c == moves[i]) val[i * nw + w] = double(v) * scale;
            } else {
                for (std::size_t i = 0; i < nm; ++i) {
                    Round sim = bases[w];
                    sim.play(moves[i], true);
                    val[i * nw + w] = playOut(std::move(sim), team, o.endgame_tricks, o.fast_policy, match_ctx);
                }
            }
        }
        w_eval = nw;
    } else {
        // CRN Candidate Racing over shared worlds in batches.
        std::size_t next_check = std::size_t(std::max(8, o.race_min));
        const std::size_t step = std::size_t(std::max(8, o.race_step));
        std::size_t n_alive = nm;
        for (std::size_t w = 0; w < nw; ++w) {
            for (std::size_t i = 0; i < nm; ++i) {
                if (!alive[i]) continue;
                Round sim = bases[w];
                sim.play(moves[i], true);
                val[i * nw + w] = playOut(std::move(sim), team, o.endgame_tricks, o.fast_policy, match_ctx);
            }
            w_eval = w + 1;
            if (w_eval >= next_check && w_eval < nw && n_alive > 1) {
                next_check = w_eval + step;
                double sw = 0.0, sw2 = 0.0;
                for (std::size_t k = 0; k < w_eval; ++k) {
                    sw += wts[k];
                    sw2 += wts[k] * wts[k];
                }
                const double ess = (sw * sw) / std::max(sw2, 1e-12);
                std::vector<double> cur_m(nm, -1e30);
                std::size_t leader = 0;
                for (std::size_t i = 0; i < nm; ++i) {
                    if (!alive[i]) continue;
                    double acc = 0.0;
                    for (std::size_t k = 0; k < w_eval; ++k) acc += wts[k] * val[i * nw + k];
                    cur_m[i] = acc / sw;
                    if (!alive[leader] || cur_m[i] > cur_m[leader]) leader = i;
                }
                for (std::size_t i = 0; i < nm; ++i) {
                    if (!alive[i] || i == leader) continue;
                    const double diff = cur_m[leader] - cur_m[i];
                    if (diff <= 0.0) continue;
                    double var = 0.0;
                    for (std::size_t k = 0; k < w_eval; ++k) {
                        const double d = (val[leader * nw + k] - val[i * nw + k]) - diff;
                        var += wts[k] * d * d;
                    }
                    const double se = std::sqrt(std::max(var / sw, 1e-12) / std::max(ess, 1.0));
                    if (diff > o.race_z * se) {
                        alive[i] = 0;
                        --n_alive;
                    }
                }
                if (n_alive <= 1) break;
            }
        }
    }

    double sum_w = 0.0, sum_w2 = 0.0;
    for (std::size_t w = 0; w < w_eval; ++w) {
        sum_w += wts[w];
        sum_w2 += wts[w] * wts[w];
    }
    res.worlds = int(std::lround(sum_w));

    std::vector<double> mean(nm, -1e30);
    double alive_search_avg = 0.0;
    int alive_cnt = 0;
    for (std::size_t i = 0; i < nm; ++i) {
        if (!alive[i]) continue;
        double acc = 0.0;
        for (std::size_t w = 0; w < w_eval; ++w) acc += wts[w] * val[i * nw + w];
        mean[i] = acc / sum_w;
        alive_search_avg += mean[i];
        ++alive_cnt;
    }
    if (alive_cnt > 0) alive_search_avg /= double(alive_cnt);

    // Optional Root Q-GRU Blend with linear late-deal taper (0 when exact root AB solves the trick!).
    std::vector<double> score = mean;
    const double w_q = (use_q && o.q_blend > 0.0 && !exact && any.trickNumber() < o.q_taper_trick)
                           ? o.q_blend * (1.0 - double(any.trickNumber() - 1) / double(std::max(1, o.q_taper_trick - 1)))
                           : 0.0;
    if (w_q > 0.0 && !q_root.empty()) {
        std::vector<double> qm(nm, 0.0);
        double q_avg = 0.0;
        for (std::size_t i = 0; i < nm; ++i) {
            for (const auto& [c, v] : q_root)
                if (c == moves[i]) qm[i] = v;
            if (alive[i]) q_avg += qm[i];
        }
        if (alive_cnt > 0) q_avg /= double(alive_cnt);
        const double slope =
            match_ctx.enabled
                ? 100.0 * MatchEquityTable::instance().marginSlope(
                              team == 0 ? ps.sn : ps.ew, team == 0 ? ps.ew : ps.sn, ps.hanging,
                              teamOf(ps.first) == team)
                : 1.0;
        for (std::size_t i = 0; i < nm; ++i) {
            if (!alive[i]) continue;
            const double q_aligned = match_ctx.enabled ? (alive_search_avg + slope * (qm[i] - q_avg)) : qm[i];
            score[i] = (1.0 - w_q) * mean[i] + w_q * q_aligned;
        }
    }

    std::size_t best = 0, rule_i = 0;
    bool best_init = false;
    for (std::size_t i = 0; i < nm; ++i) {
        if (moves[i] == res.rule_card) rule_i = i;
        if (alive[i]) {
            res.values.push_back({moves[i], score[i]});
            if (!best_init || score[i] > score[best]) {
                best = i;
                best_init = true;
            }
        }
    }
    if (best == rule_i) return res;
    if (o.z < 0 || enumerated || !alive[rule_i]) {
        res.card = moves[best];
        res.overruled = true;
        return res;
    }
    const double m = score[best] - score[rule_i];
    const double dm = mean[best] - mean[rule_i];
    double var = 0.0;
    for (std::size_t w = 0; w < w_eval; ++w) {
        const double d = (val[best * nw + w] - val[rule_i * nw + w]) - dm;
        var += wts[w] * d * d;
    }
    const double ess = (sum_w * sum_w) / std::max(sum_w2, 1e-12);
    const double se = (1.0 - w_q) * std::sqrt(std::max(var / sum_w, 1e-12) / std::max(ess, 1.0));
    if (m > o.margin && m > o.z * se) {
        res.card = moves[best];
        res.overruled = true;
    }
    return res;
}

}  // namespace bg
