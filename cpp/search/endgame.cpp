#include "search/endgame.h"

#include "belote/match.h"
#include "search/bid_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace belote {

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

TTStorage& threadTT() {
    thread_local TTStorage storage;
    return storage;
}

// Deterministic splitmix64 Zobrist tables, initialized once.
struct Zobrist {
    std::uint64_t card[kPlayers][kCards]{};
    std::uint64_t leader[kPlayers]{};
    std::uint64_t pts[kDealPoints + 1]{};
    std::uint64_t ronde[4]{};
    std::uint64_t belot[2]{};

    Zobrist() {
        std::uint64_t x = 0x517CC1B727220A95ull;
        auto next = [&]() {
            x += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        };
        for (int p = 0; p < kPlayers; ++p) {
            leader[p] = next();
            for (int c = 0; c < kCards; ++c) card[p][c] = next();
        }
        for (int i = 0; i <= kDealPoints; ++i) pts[i] = next();
        for (int i = 0; i < 4; ++i) ronde[i] = next();
        for (int i = 0; i < 2; ++i) belot[i] = next();
    }
};

const Zobrist& zobrist() {
    static const Zobrist z;
    return z;
}

// Fast leaf score for team 0 minus team 1, exact match to scoreDeal() + recordDeal().
struct RootScoring {
    Config cfg;
    int takers = 0;
    int trump = 0;
    int dealer = 0;
    int normal_toon_team = -1;
    std::array<int, 2> team_toon_pts{0, 0};
    int belot_holder = -1;
    EndgameObjective objective = EndgameObjective::kUnits;
    const BidModel* bid_model = nullptr;
    std::array<int, 2> match_scores{0, 0};
    int match_pot = 0;

    int evalLeaf(const std::array<int, 2>& card_pts, const std::array<int, 2>& tricks_won,
                 bool belot_announced) const {
        const int defenders = 1 - takers;
        int ronde_team = -1;
        if (tricks_won[0] == kTricks) ronde_team = 0;
        else if (tricks_won[1] == kTricks) ronde_team = 1;

        std::array<int, 2> cp = card_pts;
        std::array<int, 2> tp{0, 0};
        if (ronde_team >= 0) {
            cp[ronde_team] = cfg.ronde_points;
            cp[1 - ronde_team] = 0;
            if (team_toon_pts[ronde_team] > 0) tp[ronde_team] = team_toon_pts[ronde_team];
        } else if (normal_toon_team >= 0) {
            tp[normal_toon_team] = team_toon_pts[normal_toon_team];
        }

        std::array<int, 2> bp{0, 0};
        if (belot_announced && belot_holder >= 0) bp[teamOf(belot_holder)] = kBelotPoints;

        std::array<int, 2> totals{cp[0] + tp[0] + bp[0], cp[1] + tp[1] + bp[1]};
        const int T = totals[takers];
        const int D = totals[defenders];
        Outcome outcome;
        if (ronde_team >= 0) {
            outcome = ronde_team == takers ? Outcome::kMade : Outcome::kNat;
        } else if (T > D) {
            outcome = Outcome::kMade;
        } else if (T < D) {
            outcome = Outcome::kNat;
        } else {
            switch (cfg.tie_rule) {
                case TieRule::kHanging: outcome = Outcome::kHanging; break;
                case TieRule::kTakersMake: outcome = Outcome::kMade; break;
                case TieRule::kTakersFail:
                default: outcome = Outcome::kNat; break;
            }
        }

        std::array<int, 2> credited{0, 0};
        int held = 0;
        if (outcome == Outcome::kMade) {
            credited = totals;
        } else if (outcome == Outcome::kNat) {
            credited[defenders] = cp[0] + cp[1] + tp[0] + tp[1] + bp[defenders];
            credited[takers] = bp[takers];
        } else {  // kHanging
            credited[defenders] = D;
            credited[takers] = 0;
            held = T;
        }

        if (objective == EndgameObjective::kMatchEquity && bid_model) {
            DealResult dr;
            dr.outcome = outcome;
            dr.takers = takers;
            dr.credited = credited;
            dr.held = held;
            const double v0 =
                bid_model->evalDealEquity(dr, cfg, dealer, 0, match_scores[0], match_scores[1], match_pot);
            return int(std::llround(10000.0 * (2.0 * v0 - 1.0)));
        }

        if (objective == EndgameObjective::kRaw) return credited[0] - credited[1];

        if (cfg.score_mode != ScoreMode::kComplement) {
            return recordedPoints(credited[0], cfg) - recordedPoints(credited[1], cfg);
        }
        const int deal_units = (credited[0] + credited[1] + held) / 10;
        const int def_units = recordedPoints(credited[defenders], cfg);
        const int tak_units = std::max(0, deal_units - def_units);
        std::array<int, 2> rec{0, 0};
        rec[defenders] = def_units;
        if (outcome != Outcome::kHanging) rec[takers] = tak_units;
        return rec[0] - rec[1];
    }
};

struct State {
    std::array<Mask, kPlayers> hands{};
    Trick trick{};
    int current = 0;
    int tricks_played = 0;
    std::array<int, 2> card_points{0, 0};
    std::array<int, 2> tricks_won{0, 0};
    bool belot_announced = false;
    bool belot_forfeited = false;
    std::uint64_t hands_hash = 0;
};

// Strips cards that are strictly equivalent to another legal card in the same
// suit: adjacent 0-point cards with no unplayed card between them in strength
// (7/8 in trump; 7/8/9 in plain suits). Exact for all Belote rules.
Mask pruneEquivalent(Mask legal, Mask all_unplayed, int trump) {
    Mask kept = legal;
    for (int s = 0; s < kSuits; ++s) {
        const Mask sm = legal & suitMask(s);
        if (popcount(sm) < 2) continue;
        const int c7 = cardOf(s, k7);
        const int c8 = cardOf(s, k8);
        const int c9 = cardOf(s, k9);
        const bool has7 = (sm & bit(c7)) != 0;
        const bool has8 = (sm & bit(c8)) != 0;
        // 7 and 8 are adjacent in strength (0 and 1) and both worth 0 points in every suit.
        if (has7 && has8) kept &= ~bit(c8);
        if (s != trump) {
            // In plain suits, 9 is also 0 points and has strength 2 (right above 8).
            const bool has9 = (sm & bit(c9)) != 0;
            if (has9 && (has8 || (has7 && !(all_unplayed & bit(c8))))) kept &= ~bit(c9);
        }
    }
    return kept;
}

class Solver {
public:
    Solver(const RootScoring& scoring, int tt_log2)
        : sc_(scoring), z_(zobrist()), storage_(threadTT()) {
        const std::size_t size = std::size_t(1) << std::max(8, std::min(22, tt_log2));
        storage_.prepare(size);
        tt_ = storage_.table.data();
        tt_mask_ = size - 1;
        gen_ = storage_.gen;
    }

    std::uint64_t nodes = 0;

    // Exact minimax from Team 0's perspective (Team 0 maximizes, Team 1 minimizes).
    int alphabeta(const State& st, int alpha, int beta) {
        ++nodes;
        if (st.tricks_played == kTricks) return sc_.evalLeaf(st.card_points, st.tricks_won, st.belot_announced);

        const bool maximizing = teamOf(st.current) == 0;
        int tt_move = -1;
        std::uint64_t tt_key = 0;
        const bool use_tt = (st.trick.n == 0);
        if (use_tt) {
            const int ronde_mask = (st.tricks_won[0] == 0 ? 1 : 0) | (st.tricks_won[1] == 0 ? 2 : 0);
            tt_key = st.hands_hash ^ z_.leader[st.current] ^ z_.pts[st.card_points[0]] ^ z_.ronde[ronde_mask] ^
                     z_.belot[st.belot_announced ? 1 : 0];
            const TTEntry& e = tt_[tt_key & tt_mask_];
            if (e.gen == gen_ && e.bound != Bound::kNone && e.key == tt_key) {
                tt_move = e.best_move;
                if (e.bound == Bound::kExact) return e.score;
                if (e.bound == Bound::kLower) alpha = std::max(alpha, int(e.score));
                else if (e.bound == Bound::kUpper) beta = std::min(beta, int(e.score));
                if (alpha >= beta) return e.score;
            }
        }

        const Mask legal = legalMask(st.hands[st.current], st.trick, sc_.trump, st.current, sc_.cfg);
        Mask in_play = st.hands[0] | st.hands[1] | st.hands[2] | st.hands[3];
        for (int i = 0; i < st.trick.n; ++i) in_play |= bit(st.trick.cards[i]);
        const Mask pruned = pruneEquivalent(legal, in_play, sc_.trump);

        int moves[8]{};
        int n_moves = orderMoves(st, pruned, tt_move, moves);

        const int orig_alpha = alpha;
        const int orig_beta = beta;
        int best_move = moves[0];
        int best = maximizing ? -kInf : kInf;

        for (int i = 0; i < n_moves; ++i) {
            const int card = moves[i];
            State next = apply(st, card);
            const int val = alphabeta(next, alpha, beta);
            if (maximizing) {
                if (val > best) {
                    best = val;
                    best_move = card;
                }
                alpha = std::max(alpha, best);
            } else {
                if (val < best) {
                    best = val;
                    best_move = card;
                }
                beta = std::min(beta, best);
            }
            if (alpha >= beta) break;
        }

        if (use_tt) {
            TTEntry& e = tt_[tt_key & tt_mask_];
            e.key = tt_key;
            e.gen = gen_;
            e.score = std::int16_t(best);
            e.best_move = std::int8_t(best_move);
            if (best <= orig_alpha) e.bound = Bound::kUpper;
            else if (best >= orig_beta) e.bound = Bound::kLower;
            else e.bound = Bound::kExact;
        }
        return best;
    }

    State apply(const State& st, int card) const {
        State n = st;
        const int p = st.current;
        if (p == sc_.belot_holder && !n.belot_announced && !n.belot_forfeited && suitOf(card) == sc_.trump) {
            const int r = rankOf(card);
            if (r == kK || (r == kQ && !sc_.cfg.belot_queen_first_forfeits)) n.belot_announced = true;
            else if (r == kQ && sc_.cfg.belot_queen_first_forfeits) n.belot_forfeited = true;
        }
        n.hands[p] &= ~bit(card);
        n.hands_hash ^= z_.card[p][card];
        n.trick.cards[n.trick.n] = card;
        n.trick.players[n.trick.n] = p;
        ++n.trick.n;
        if (n.trick.n < kPlayers) {
            n.current = nextPlayer(p);
            return n;
        }
        const int winner = n.trick.players[trickWinnerIndex(n.trick, sc_.trump)];
        int pts = 0;
        for (int c : n.trick.cards) pts += cardPoints(c, sc_.trump);
        if (n.tricks_played == kTricks - 1) pts += kLastTrickBonus;
        n.card_points[teamOf(winner)] += pts;
        ++n.tricks_won[teamOf(winner)];
        ++n.tricks_played;
        n.trick = Trick{};
        n.current = winner;
        return n;
    }

private:
    int orderMoves(const State& st, Mask m, int tt_move, int (&out)[8]) const {
        int n = 0;
        int keys[8];
        for (; m; m &= m - 1) {
            const int c = lowestCard(m);
            int key = 0;
            if (c == tt_move) {
                key = 10000;
            } else {
                const bool is_trump = suitOf(c) == sc_.trump;
                if (st.trick.n == 0) {
                    // Lead high trumps / high point cards first for faster cutoffs.
                    key = (is_trump ? 50 : 0) + cardPoints(c, sc_.trump) * 2 + cardStrength(c, sc_.trump);
                } else {
                    Trick u = st.trick;
                    u.cards[u.n] = c;
                    u.players[u.n] = st.current;
                    ++u.n;
                    const bool wins = trickWinnerIndex(u, sc_.trump) == st.trick.n;
                    key = (wins ? 100 : 0) + (is_trump ? 20 : 0) + cardPoints(c, sc_.trump);
                }
            }
            int pos = n;
            while (pos > 0 && keys[pos - 1] < key) {
                keys[pos] = keys[pos - 1];
                out[pos] = out[pos - 1];
                --pos;
            }
            keys[pos] = key;
            out[pos] = c;
            ++n;
        }
        return n;
    }

    RootScoring sc_;
    const Zobrist& z_;
    TTStorage& storage_;
    TTEntry* tt_ = nullptr;
    std::size_t tt_mask_ = 0;
    std::uint32_t gen_ = 0;
};

void initEndgameRoot(const Deal& deal, const EndgameOptions& options, RootScoring& sc, State& root) {
    sc.cfg = deal.config();
    sc.takers = deal.takers();
    sc.trump = deal.trump();
    sc.dealer = deal.dealer();
    sc.belot_holder = deal.belotHolder();
    sc.objective = options.objective;
    sc.bid_model = options.bid_model;
    sc.match_scores = options.match_scores;
    sc.match_pot = options.match_pot;

    // Precompute all toon totals once at the root (exact for every trick 0..7,
    // including mandatory declaring when toon_optional == false).
    std::vector<Toon> announced[kPlayers];
    const Trick& t0 = deal.trick();
    for (int p = 0; p < kPlayers; ++p) {
        bool played_first = deal.tricksPlayed() > 0;
        if (!played_first) {
            for (int i = 0; i < t0.n; ++i)
                if (t0.players[i] == p) played_first = true;
        }
        bool will_announce = deal.announced(p);
        if (!will_announce && !played_first && sc.cfg.toon_enabled && deal.declares(p) && !deal.toons(p).empty())
            will_announce = true;
        if (will_announce) {
            announced[p] = deal.toons(p);
            sc.team_toon_pts[teamOf(p)] += totalPoints(announced[p]);
        }
    }
    sc.normal_toon_team = toonWinningTeam(announced, deal.trump(), deal.dealer());

    const Zobrist& z = zobrist();
    for (int p = 0; p < kPlayers; ++p) {
        root.hands[p] = deal.hand(p);
        for (Mask m = root.hands[p]; m; m &= m - 1) root.hands_hash ^= z.card[p][lowestCard(m)];
    }
    root.trick = deal.trick();
    root.current = deal.currentPlayer();
    root.tricks_played = deal.tricksPlayed();
    root.card_points = deal.cardPoints();
    for (int i = 0; i < deal.tricksPlayed(); ++i) ++root.tricks_won[teamOf(deal.trickWinner(i))];
    root.belot_announced = deal.belotAnnounced();
    if (sc.belot_holder >= 0 && !root.belot_announced && sc.cfg.belot_queen_first_forfeits &&
        !(root.hands[sc.belot_holder] & bit(cardOf(sc.trump, kQ)))) {
        root.belot_forfeited = true;
    }
}

}  // namespace

EndgameResult solveEndgame(const Deal& deal, const EndgameOptions& options) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("solveEndgame: deal is not in kPlay");

    RootScoring sc;
    State root;
    initEndgameRoot(deal, options, sc, root);

    Solver solver(sc, options.tt_log2);
    const Mask legal = legalMask(root.hands[root.current], root.trick, sc.trump, root.current, sc.cfg);
    Mask in_play = root.hands[0] | root.hands[1] | root.hands[2] | root.hands[3];
    for (int i = 0; i < root.trick.n; ++i) in_play |= bit(root.trick.cards[i]);
    const Mask unique_legal = pruneEquivalent(legal, in_play, sc.trump);

    const int root_team = teamOf(root.current);
    const int sign = root_team == 0 ? 1 : -1;

    EndgameResult res;
    int best_score = -kInf;
    int score_by_card[kCards];
    std::fill(std::begin(score_by_card), std::end(score_by_card), -kInf);

    // Evaluate every unique legal move with exact window so move_values are all exact.
    for (Mask m = unique_legal; m; m &= m - 1) {
        const int card = lowestCard(m);
        State next = solver.apply(root, card);
        const int val0 = solver.alphabeta(next, -kInf, kInf);
        const int val = sign * val0;
        score_by_card[card] = val;
        if (val > best_score) {
            best_score = val;
            res.best_move = card;
        }
    }

    // Populate scores for every legal card (including pruned 0-point equivalents).
    for (Mask m = legal; m; m &= m - 1) {
        const int card = lowestCard(m);
        int val = score_by_card[card];
        if (val == -kInf) {
            // Copy from the kept 0-point equivalent in the same suit (7 or 8).
            const int s = suitOf(card);
            if (score_by_card[cardOf(s, k7)] != -kInf) val = score_by_card[cardOf(s, k7)];
            else val = score_by_card[cardOf(s, k8)];
        }
        res.move_values.push_back({card, val});
    }
    res.value = best_score;
    res.nodes = solver.nodes;
    return res;
}

int evaluateEndgameLine(const Deal& deal, const std::vector<int>& cards, EndgameObjective objective) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("evaluateEndgameLine: deal is not in kPlay");
    EndgameOptions opts;
    opts.objective = objective;
    RootScoring sc;
    State st;
    initEndgameRoot(deal, opts, sc, st);
    Solver solver(sc, 8);
    for (int card : cards) st = solver.apply(st, card);
    if (st.tricks_played != kTricks) throw std::logic_error("evaluateEndgameLine: line did not finish deal");
    return sc.evalLeaf(st.card_points, st.tricks_won, st.belot_announced);
}

}  // namespace belote
