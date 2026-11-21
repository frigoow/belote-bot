#include "bg/view.h"

#include <algorithm>

namespace bg {

namespace {

Mask strongerInTrumpOrder(int card) {
    Mask m = 0;
    for (int r = 0; r < 8; ++r)
        if (trumpOrder(cardOf(suitOf(card), r)) > trumpOrder(card)) m |= bit(cardOf(suitOf(card), r));
    return m;
}

// The card currently winning the trick and whether it is a trump (suit contracts).
int winningCard(std::uint8_t contract, const Trick& t) { return t.cards[trickWinnerIndex(contract, t)]; }

}  // namespace

Mask PublicState::played() const {
    Mask m = 0;
    for (const PublicPlay& p : plays) m |= bit(p.card);
    return m;
}

Mask PublicState::playedBy(int seat) const {
    Mask m = 0;
    for (const PublicPlay& p : plays)
        if (p.seat == seat) m |= bit(p.card);
    return m;
}

int PublicState::toMove() const {
    if (plays.empty()) return first;
    if (plays.size() % kPlayers != 0) return nextSeat(plays.back().seat);
    Trick t;
    for (std::size_t i = plays.size() - kPlayers; i < plays.size(); ++i) {
        t.cards[t.n] = plays[i].card;
        t.players[t.n] = plays[i].seat;
        ++t.n;
    }
    return t.players[trickWinnerIndex(contract, t)];
}

PublicState publicState(const Round& r, int observer, int sn, int ew) {
    PublicState ps;
    ps.first = r.first();
    ps.me = observer;
    ps.hand = r.hand(observer);
    ps.contract = r.contract();
    ps.declarer = r.contractPlayer();
    ps.bids = r.bids();
    for (const Announce& a : r.announces())
        if (a.type != kBelot) ps.declared.push_back({a.player, a.type, a.player == observer ? a.card : kNoCard});
    for (const auto& pl : r.plays()) ps.plays.push_back({pl.first, pl.second.first, pl.second.second});
    ps.sn = sn;
    ps.ew = ew;
    ps.hanging = r.hangingBefore();
    return ps;
}

Facts inferFacts(const PublicState& ps) {
    Facts f;
    const std::uint8_t c = ps.contract;
    const bool all = c & kAllTrumps, none = c & kNoTrumps;
    const int trump = trumpSuit(c);
    Trick t;
    Mask played = 0;
    std::array<int, kPlayers> left{kTricks, kTricks, kTricks, kTricks};
    for (const PublicPlay& pl : ps.plays) {
        const int p = pl.seat, card = pl.card;
        // The legal-move rule read backwards (RULES_BG.md): with a single card
        // left there was no choice, so nothing follows.
        if (t.n > 0 && left[p] > 1) {
            const int led = suitOf(t.cards[0]);
            if (all || (!none && led == trump)) {
                if (suitOf(card) != led) {
                    f.excluded[p] |= suitMask(led);
                } else {
                    int biggest = t.cards[0];
                    for (int i = 1; i < t.n; ++i)
                        if (suitOf(t.cards[i]) == led && trumpOrder(t.cards[i]) > trumpOrder(biggest)) biggest = t.cards[i];
                    if (trumpOrder(card) < trumpOrder(biggest)) f.excluded[p] |= strongerInTrumpOrder(biggest);
                }
            } else if (none) {
                if (suitOf(card) != led) f.excluded[p] |= suitMask(led);
            } else if (suitOf(card) != led) {
                f.excluded[p] |= suitMask(led);
                int biggest_trump = kNoCard;
                for (int i = 0; i < t.n; ++i)
                    if (suitOf(t.cards[i]) == trump &&
                        (biggest_trump == kNoCard || trumpOrder(t.cards[i]) > trumpOrder(biggest_trump)))
                        biggest_trump = t.cards[i];
                const bool partner_winning = t.n > 1 && t.cards[t.n - 2] == winningCard(c, t);
                if (!partner_winning) {
                    if (biggest_trump != kNoCard) {
                        if (!(suitOf(card) == trump && trumpOrder(card) > trumpOrder(biggest_trump)))
                            f.excluded[p] |= strongerInTrumpOrder(biggest_trump);
                    } else if (suitOf(card) != trump) {
                        f.excluded[p] |= suitMask(trump);
                    }
                }
            }
        }
        if (pl.belote) {
            const int other = cardOf(suitOf(card), rankOf(card) == belote::kK ? belote::kQ : belote::kK);
            if (!(played & bit(other))) f.pinned[p] |= bit(other);
        }
        played |= bit(card);
        --left[p];
        t.cards[t.n] = card;
        t.players[t.n] = p;
        if (++t.n == kPlayers) t = Trick{};
    }
    for (const Declared& d : ps.declared) {
        if (d.seat == ps.me) continue;
        if (d.type == kFourJacks || d.type == kFourNines) {
            const int rank = d.type == kFourJacks ? belote::kJ : belote::k9;
            for (int s = 0; s < 4; ++s) f.pinned[d.seat] |= bit(cardOf(s, rank));
        }
    }
    for (int p = 0; p < kPlayers; ++p) f.pinned[p] &= ~played;
    return f;
}

// ------------------------------------------------------------------ sampler

namespace {

// ways[i][a][b]: assignments of hidden cards i.. to the three other seats with
// remaining capacities a, b and (n - i - a - b), each card within its domain.
std::vector<double> buildWays(const std::vector<std::uint8_t>& dom) {
    const std::size_t n = dom.size();
    std::vector<double> w((n + 1) * 81, 0.0);
    auto at = [&](std::size_t i, int a, int b) -> double& { return w[i * 81 + std::size_t(a * 9 + b)]; };
    for (std::size_t i = n + 1; i-- > 0;) {
        for (int a = 0; a <= 8; ++a) {
            for (int b = 0; b <= 8; ++b) {
                const int cc = int(n - i) - a - b;
                double x = 0;
                if (cc >= 0 && cc <= 8) {
                    if (i == n) {
                        x = (a == 0 && b == 0 && cc == 0) ? 1.0 : 0.0;
                    } else {
                        const std::uint8_t d = dom[i];
                        if ((d & 1) && a > 0) x += at(i + 1, a - 1, b);
                        if ((d & 2) && b > 0) x += at(i + 1, a, b - 1);
                        if ((d & 4) && cc > 0) x += at(i + 1, a, b);
                    }
                }
                at(i, a, b) = x;
            }
        }
    }
    return w;
}

inline double waysAt(const std::vector<double>& w, std::size_t i, int a, int b) { return w[i * 81 + std::size_t(a * 9 + b)]; }

// The cards a declaration of `type` could be: every run of its length, every carré rank.
std::vector<Mask> declarationCandidates(int type) {
    std::vector<Mask> out;
    if (type >= kSeq3 && type <= kSeq8) {
        const int len = type - kSeq3 + 3;
        for (int s = 0; s < 4; ++s)
            for (int top = len - 1; top < 8; ++top) out.push_back(Mask(((1u << len) - 1u) << (s * 8 + top - len + 1)));
    } else if (type == kFourOfAKind) {
        for (int r : {belote::k10, belote::kQ, belote::kK, belote::kA}) out.push_back(Mask(0x01010101u) << r);
    }
    return out;
}

}  // namespace

DealSampler::DealSampler(const PublicState& ps) : ps_(ps), facts_(inferFacts(ps)) {
    for (int k = 0, p = 0; p < kPlayers; ++p)
        if (p != ps.me) others_[std::size_t(k++)] = p;
    for (int k = 0; k < 3; ++k) need_[std::size_t(k)] = ps.handSize(others_[std::size_t(k)]);
    const Mask known = ps.hand | ps.played();
    std::array<int, kCards> index_of{};
    index_of.fill(-1);
    for (int c = 0; c < kCards; ++c) {
        if (known & bit(c)) continue;
        std::uint8_t dom = 0;
        int pinned_to = -1;
        for (int k = 0; k < 3; ++k)
            if (facts_.pinned[std::size_t(others_[std::size_t(k)])] & bit(c)) pinned_to = k;
        for (int k = 0; k < 3; ++k) {
            const int p = others_[std::size_t(k)];
            if (facts_.excluded[std::size_t(p)] & bit(c)) continue;
            if (pinned_to >= 0 && pinned_to != k) continue;
            dom |= std::uint8_t(1u << k);
        }
        index_of[std::size_t(c)] = int(hidden_.size());
        hidden_.push_back(c);
        domain_.push_back(dom);
    }
    ways_ = buildWays(domain_);

    // Opponents' declarations are known by type only. Rejection alone rarely
    // finds two tierces in one hand, so place the declared combinations
    // first: every disjoint choice of their cards, weighted by the deals that
    // complete it (the replay check then removes the few that declare more).
    struct Item {
        int k;                     // index into others_
        std::vector<Mask> cands;
    };
    std::vector<Item> items;
    for (const Declared& d : ps.declared) {
        if (d.seat == ps.me || d.type == kFourJacks || d.type == kFourNines) continue;
        int k = 0;
        while (others_[std::size_t(k)] != d.seat) ++k;
        Mask possible = ps.playedBy(d.seat);
        for (std::size_t i = 0; i < hidden_.size(); ++i)
            if (domain_[i] & (1u << k)) possible |= bit(hidden_[i]);
        Item it{k, {}};
        for (Mask m : declarationCandidates(d.type))
            if ((m & ~possible) == 0) it.cands.push_back(m);
        items.push_back(std::move(it));
    }
    if (items.empty()) return;
    bool overflow = false;
    std::vector<std::uint8_t> dom = domain_;
    std::function<void(std::size_t, Mask)> place = [&](std::size_t j, Mask used) {
        if (overflow) return;
        if (j == items.size()) {
            const auto w = buildWays(dom);
            const double total = waysAt(w, 0, need_[0], need_[1]);
            if (total > 0) placements_.push_back({dom, total});
            if (placements_.size() > 4000) overflow = true;
            return;
        }
        for (Mask m : items[j].cands) {
            if (m & used) continue;
            std::vector<std::pair<int, std::uint8_t>> saved;
            for (Mask x = m; x; x &= x - 1) {
                const int i = index_of[std::size_t(lowestCard(x))];
                if (i < 0) continue;  // already played by the declarer
                saved.push_back({i, dom[std::size_t(i)]});
                dom[std::size_t(i)] = std::uint8_t(1u << items[j].k);
            }
            place(j + 1, used | m);
            for (const auto& [i, d] : saved) dom[std::size_t(i)] = d;
        }
    };
    place(0, 0);
    if (overflow) placements_.clear();
    for (const Placement& p : placements_) placement_total_ += p.weight;
}

double DealSampler::ways(std::size_t i, int a, int b) const { return waysAt(ways_, i, a, b); }

std::optional<Hands> DealSampler::sampleUnchecked(std::mt19937_64& rng) const {
    const std::vector<std::uint8_t>* dom = &domain_;
    const std::vector<double>* table = &ways_;
    std::vector<double> local;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (!placements_.empty()) {
        double x = u(rng) * placement_total_;
        std::size_t j = 0;
        while (j + 1 < placements_.size() && x >= placements_[j].weight) x -= placements_[j++].weight;
        dom = &placements_[j].domain;
        local = buildWays(*dom);
        table = &local;
    }
    int a = need_[0], b = need_[1];
    if (waysAt(*table, 0, a, b) <= 0) return std::nullopt;
    Hands h{};
    h[std::size_t(ps_.me)] = ps_.hand;
    for (std::size_t i = 0; i < hidden_.size(); ++i) {
        const int cc = int(hidden_.size() - i) - a - b;
        const std::uint8_t d = (*dom)[i];
        const double w0 = (d & 1) && a > 0 ? waysAt(*table, i + 1, a - 1, b) : 0.0;
        const double w1 = (d & 2) && b > 0 ? waysAt(*table, i + 1, a, b - 1) : 0.0;
        const double w2 = (d & 4) && cc > 0 ? waysAt(*table, i + 1, a, b) : 0.0;
        const double x = u(rng) * (w0 + w1 + w2);
        int k = x < w0 ? 0 : x < w0 + w1 ? 1 : 2;
        if (k == 2 && w2 <= 0) k = w1 > 0 ? 1 : 0;
        h[std::size_t(others_[std::size_t(k)])] |= bit(hidden_[i]);
        if (k == 0) --a;
        else if (k == 1) --b;
    }
    return h;
}

bool DealSampler::admits(const Hands& h) const {
    for (int k = 0; k < 3; ++k) {
        const int p = others_[std::size_t(k)];
        if (popcount(h[std::size_t(p)]) != need_[std::size_t(k)]) return false;
        if (h[std::size_t(p)] & facts_.excluded[std::size_t(p)]) return false;
        if ((facts_.pinned[std::size_t(p)] & h[std::size_t(p)]) != facts_.pinned[std::size_t(p)]) return false;
    }
    return h[std::size_t(ps_.me)] == ps_.hand;
}

std::optional<Round> DealSampler::replay(const Hands& h) const {
    Hands start{};
    for (int p = 0; p < kPlayers; ++p) start[std::size_t(p)] = h[std::size_t(p)] | ps_.playedBy(p);
    return replayDeal(ps_, start);
}

std::optional<Hands> DealSampler::sample(std::mt19937_64& rng, int max_tries) const {
    bool needs_replay = false;
    for (const Declared& d : ps_.declared) needs_replay = needs_replay || (d.seat != ps_.me);
    std::optional<Hands> last;
    for (int i = 0; i < max_tries; ++i) {
        last = sampleUnchecked(rng);
        if (!last) return std::nullopt;
        if (!needs_replay || replay(*last)) return last;
    }
    return last;  // gave up: an unchecked world
}

// ------------------------------------------------------------------ replay

std::optional<Round> replayDeal(const PublicState& ps, const Hands& start, const PlayHook* before_play) {
    std::array<int, kCards> deal{};
    for (int p = 0; p < kPlayers; ++p) {
        if (popcount(start[std::size_t(p)]) != kTricks) return std::nullopt;
        int k = 0;
        for (Mask m = start[std::size_t(p)]; m; m &= m - 1, ++k) {
            const int slot = k < 5 ? 4 * k + p : 20 + 4 * (k - 5) + p;
            deal[std::size_t(slot)] = lowestCard(m);
        }
    }
    Round r(deal, ps.first, ps.hanging);
    while (r.decision() == Round::Decision::kBid) {
        const std::size_t k = r.bids().size();
        if (k >= ps.bids.size() || ps.bids[k].first != r.toMove()) return std::nullopt;
        const std::uint8_t b = ps.bids[k].second;
        if (b != kPass && !(r.availableBids() & b)) return std::nullopt;
        r.bid(b);
    }
    if (r.bids() != ps.bids) return std::nullopt;
    for (;;) {
        if (r.decision() == Round::Decision::kAnnounce) {
            const int seat = r.toMove();
            const auto& offered = r.offeredAnnounces();
            std::vector<Announce> chosen;
            std::vector<bool> used(offered.size(), false);
            for (const Declared& d : ps.declared) {
                if (d.seat != seat) continue;
                std::size_t hit = offered.size();
                for (std::size_t i = 0; i < offered.size() && hit == offered.size(); ++i)
                    if (!used[i] && offered[i].type == d.type && (d.card == kNoCard || offered[i].card == d.card)) hit = i;
                if (hit == offered.size()) return std::nullopt;
                used[hit] = true;
                chosen.push_back(offered[hit]);
            }
            r.declare(chosen);
            continue;
        }
        const std::size_t j = r.plays().size();
        for (std::size_t i = 0; i < j && i < ps.plays.size(); ++i)
            if (r.plays()[i].second.first != ps.plays[i].card) return std::nullopt;  // a forced card differs
        if (j > ps.plays.size()) return std::nullopt;
        if (j == ps.plays.size()) break;
        if (r.decision() != Round::Decision::kPlay) return std::nullopt;
        const PublicPlay& pl = ps.plays[j];
        if (pl.seat != r.toMove() || !(r.legalMoves() & bit(pl.card))) return std::nullopt;
        if (pl.belote && !beloteAllowed(r.hand(pl.seat), r.contract(), r.trick(), pl.card)) return std::nullopt;
        if (before_play) (*before_play)(r, pl);
        r.play(pl.card, pl.belote);
    }
    // Declarations made are part of the record: every one must have happened.
    std::size_t made = 0;
    for (const Announce& a : r.announces()) made += a.type != kBelot;
    if (made != ps.declared.size()) return std::nullopt;
    return r;
}

}  // namespace bg
