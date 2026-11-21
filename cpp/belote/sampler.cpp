#include "belote/sampler.h"

#include <algorithm>
#include <stdexcept>

namespace belote {

WorldSampler::WorldSampler(const Observation& obs) : obs_(obs) {
    const Constraints c = inferConstraints(obs_);
    for (int k = 0; k < 3; ++k) others_[k] = (obs_.observer + 1 + k) % kPlayers;
    base_.hands[obs_.observer] = obs_.hand;

    // Seats whose longest run is known: sequence announcers, and players who
    // could not have hidden a toon (toon_optional off) and did not announce.
    if (obs_.cfg.toon_enabled) {
        for (int k = 0; k < 3; ++k) {
            const int p = others_[k];
            int target = -1;
            if (obs_.announced[p] && obs_.announcement[p].kind != ToonKind::kCarre)
                target = 1 + int(obs_.announcement[p].kind);  // derde 1, vierde 2, vijfde+ 3
            else if (!obs_.announced[p] && !obs_.cfg.toon_optional && obs_.playedBy(p))
                target = 0;
            if (target >= 0) {
                track_slot_[k] = tracked_;
                target_[tracked_++] = target;
            }
        }
    }

    const Mask hidden = ~(obs_.hand | obs_.played());
    owner_.fill(-1);
    for (int k = 0; k < 3; ++k)
        for (Mask m = obs_.playedBy(others_[k]); m; m &= m - 1) owner_[lowestCard(m)] = std::int8_t(k);
    for (int card = 0; card < kCards; ++card) {
        hidden_[card] = (hidden & bit(card)) != 0;
        if (!hidden_[card]) continue;
        int pinned_to = -1;
        for (int k = 0; k < 3; ++k)
            if (c.pinned[others_[k]] & bit(card)) pinned_to = k;
        std::uint8_t dom = 0;
        for (int k = 0; k < 3; ++k) {
            if (c.excluded[others_[k]] & bit(card)) continue;
            if (pinned_to >= 0 && pinned_to != k) continue;
            dom |= std::uint8_t(1u << k);
        }
        domain_[card] = dom;
    }
    hidden_left_[kCards] = 0;
    for (int i = kCards - 1; i >= 0; --i) hidden_left_[i] = hidden_left_[i + 1] + (hidden_[i] ? 1 : 0);

    start_r0_ = obs_.handSize(others_[0]);
    start_r1_ = obs_.handSize(others_[1]);
    if (start_r0_ + start_r1_ + obs_.handSize(others_[2]) != hidden_left_[0])
        throw std::logic_error("WorldSampler: hand sizes do not match the hidden cards");

    memo_.reserve(tracked_ ? 1 << 14 : 1 << 12);
    count_ = countFrom(0, start_r0_, start_r1_, 0);

    needs_check_ = obs_.cfg.toon_enabled && !obs_.cfg.toon_optional;
    for (int p = 0; p < kPlayers; ++p)
        if (p != obs_.observer && (obs_.announced[p] || obs_.shown[p])) needs_check_ = true;
}

bool WorldSampler::step(int i, int k, Runs& runs) const {
    if (!tracked_) return true;
    Runs out = 0;
    for (int t = 0; t < tracked_; ++t) {
        int cur = int((runs >> (5 * t)) & 7);
        int cls = int((runs >> (5 * t + 3)) & 3);
        if (i % kRanks == 0) cur = 0;  // a new suit
        const bool holds = k >= 0 && track_slot_[k] == t;
        if (holds) {
            cur = std::min(cur + 1, 5);
            const int c = cur >= 5 ? 3 : cur == 4 ? 2 : cur == 3 ? 1 : 0;
            cls = std::max(cls, c);
            if (cls > target_[t]) return false;
        } else {
            cur = 0;
        }
        out |= Runs(cur | (cls << 3)) << (5 * t);
    }
    runs = out;
    return true;
}

std::uint64_t WorldSampler::countFrom(int i, int r0, int r1, Runs runs) {
    const std::uint64_t k = key(i, r0, r1, runs);
    if (auto it = memo_.find(k); it != memo_.end()) return it->second;
    const int r2 = hidden_left_[i] - r0 - r1;
    std::uint64_t ways = 0;
    if (r2 < 0 || r2 > 8) {
        ways = 0;
    } else if (i == kCards) {
        ways = 1;
        for (int t = 0; t < tracked_; ++t)
            if (int((runs >> (5 * t + 3)) & 3) != target_[t]) ways = 0;
    } else if (!hidden_[i]) {
        Runs next = runs;
        if (step(i, owner_[i], next)) ways = countFrom(i + 1, r0, r1, next);
    } else {
        const std::uint8_t dom = domain_[i];
        const int r[3] = {r0, r1, r2};
        for (int s = 0; s < 3; ++s) {
            if (!(dom & (1u << s)) || r[s] == 0) continue;
            Runs next = runs;
            if (!step(i, s, next)) continue;
            ways += countFrom(i + 1, r0 - (s == 0), r1 - (s == 1), next);
        }
    }
    memo_.emplace(k, ways);
    return ways;
}

std::uint64_t WorldSampler::lookup(int i, int r0, int r1, Runs runs) const {
    const auto it = memo_.find(key(i, r0, r1, runs));
    return it == memo_.end() ? 0 : it->second;
}

World WorldSampler::sampleUnchecked(std::mt19937_64& rng) const {
    if (count_ == 0) throw std::logic_error("WorldSampler: no world is consistent with the play");
    World w = base_;
    int r0 = start_r0_, r1 = start_r1_;
    Runs runs = 0;
    for (int i = 0; i < kCards; ++i) {
        if (!hidden_[i]) {
            step(i, owner_[i], runs);
            continue;
        }
        const int r2 = hidden_left_[i] - r0 - r1;
        const int r[3] = {r0, r1, r2};
        std::uint64_t weight[3] = {0, 0, 0};
        Runs next[3] = {runs, runs, runs};
        for (int s = 0; s < 3; ++s) {
            if (!(domain_[i] & (1u << s)) || r[s] == 0 || !step(i, s, next[s])) continue;
            weight[s] = lookup(i + 1, r0 - (s == 0), r1 - (s == 1), next[s]);
        }
        std::uint64_t x =
            std::uniform_int_distribution<std::uint64_t>(0, weight[0] + weight[1] + weight[2] - 1)(rng);
        int s = 0;
        while (x >= weight[s]) x -= weight[s++];
        w.hands[others_[s]] |= bit(i);
        r0 -= (s == 0);
        r1 -= (s == 1);
        runs = next[s];
    }
    return w;
}

bool WorldSampler::admits(const World& w) const {
    if (w.hands[obs_.observer] != obs_.hand) return false;
    for (int k = 0; k < 3; ++k)
        if (popcount(w.hands[others_[k]]) != obs_.handSize(others_[k])) return false;
    for (int card = 0; card < kCards; ++card) {
        if (!hidden_[card]) continue;
        int owner = -1;
        for (int k = 0; k < 3; ++k)
            if (w.hands[others_[k]] & bit(card)) owner = k;
        if (owner < 0 || !(domain_[card] & (1u << owner))) return false;
    }
    return true;
}

Deal WorldSampler::replay(const World& w) const {
    std::array<Mask, kPlayers> start{};
    for (int p = 0; p < kPlayers; ++p) start[p] = w.hands[p] | obs_.playedBy(p);
    Deal d = Deal::fromPlay(obs_.cfg, obs_.dealer, obs_.taker, obs_.trump, start);
    for (int p = 0; p < kPlayers; ++p) {
        if (!d.canSetDeclare(p)) continue;
        if (p == obs_.observer) d.setDeclare(p, obs_.declare_self);
        else if (obs_.playedBy(p)) d.setDeclare(p, obs_.announced[p]);
        // Players yet to play their first card keep the default: they declare.
    }
    for (const Play& pl : obs_.plays) {
        if (d.currentPlayer() != pl.player) throw std::logic_error("replay: wrong player to move");
        d.play(pl.card);
    }
    return d;
}

bool WorldSampler::consistent(const World& w) const {
    // The blik went to the taker (the replayed deal starts after that).
    if (obs_.blik != kNoCard && !((w.hands[obs_.taker] | obs_.playedBy(obs_.taker)) & bit(obs_.blik)))
        return false;
    try {
        const Deal d = replay(w);
        for (int p = 0; p < kPlayers; ++p) {
            if (d.announced(p) != obs_.announced[p] || d.shown(p) != obs_.shown[p]) return false;
            if (obs_.announced[p]) {
                const Announcement& a = d.announcement(p);
                const Announcement& b = obs_.announcement[p];
                if (a.kind != b.kind || a.carre_rank != b.carre_rank) return false;
            }
            if (obs_.shown[p]) {
                Mask cards = 0;
                for (const Toon& t : d.toons(p)) cards |= t.cards;
                if (cards != obs_.shown_cards[p]) return false;
            }
        }
        return d.belotAnnouncer() == obs_.belot_announcer && d.belotCard() == obs_.belot_card;
    } catch (const std::logic_error&) {
        return false;
    }
}

std::optional<World> WorldSampler::sample(std::mt19937_64& rng, int max_tries) const {
    if (count_ == 0) return std::nullopt;
    for (int i = 0; i < max_tries; ++i) {
        World w = sampleUnchecked(rng);
        if (!needs_check_ || consistent(w)) return w;
    }
    return std::nullopt;
}

void WorldSampler::enumerateFrom(int i, int r0, int r1, Runs runs, World& w, std::vector<World>& out) const {
    while (i < kCards && !hidden_[i]) step(i, owner_[i], runs), ++i;
    if (i == kCards) {
        if (!needs_check_ || consistent(w)) out.push_back(w);
        return;
    }
    const int r2 = hidden_left_[i] - r0 - r1;
    const int r[3] = {r0, r1, r2};
    for (int s = 0; s < 3; ++s) {
        Runs next = runs;
        if (!(domain_[i] & (1u << s)) || r[s] == 0 || !step(i, s, next)) continue;
        if (!lookup(i + 1, r0 - (s == 0), r1 - (s == 1), next)) continue;
        w.hands[others_[s]] |= bit(i);
        enumerateFrom(i + 1, r0 - (s == 0), r1 - (s == 1), next, w, out);
        w.hands[others_[s]] &= ~bit(i);
    }
}

std::vector<World> WorldSampler::enumerate(std::uint64_t max_worlds) const {
    std::vector<World> out;
    if (count_ == 0 || count_ > max_worlds) return out;
    World w = base_;
    enumerateFrom(0, start_r0_, start_r1_, 0, w, out);
    return out;
}

}  // namespace belote
