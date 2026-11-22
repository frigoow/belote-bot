#pragma once
// Bulgarian Belot as a cardbot-framework game (cardbot-framework
// cpp/include/cardbot/game.h), so the framework's agents, search and duels
// work on it. Actions are numbered as in the app (app/cpp/belot_api.h):
//   0..31 a card (belot claimed whenever allowed), 32 + b a bid (0 pass, 1..4
//   clubs..spades, 5 no trumps, 6 all trumps, 7 double, 8 redouble), 64 + mask
//   the offered declarations to make.
// One deal; payoff is the deal's game-point margin for the seat's team.

#include "bg/agent.h"
#include "bg/bg.h"
#include "bg/bid_model.h"
#include "bg/view.h"
#include "cardbot/game.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace bg {

class FrameworkGame {
public:
    using Action = int;
    static constexpr int kPlayers = 4;
    static constexpr int kBid = 32;
    static constexpr int kDeclare = 64;

    explicit FrameworkGame(Round r) : r_(std::move(r)) {}

    /** A fresh deal; the first seat to bid rotates with the seed. */
    static FrameworkGame deal(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::array<int, kCards> deck{};
        for (int i = 0; i < kCards; ++i) deck[std::size_t(i)] = i;
        std::shuffle(deck.begin(), deck.end(), rng);
        return FrameworkGame(Round(deck, int(seed % 4)));
    }

    int toMove() const { return r_.toMove(); }
    bool finished() const { return r_.finished(); }
    int team(int seat) const { return teamOf(seat); }
    const Round& round() const { return r_; }

    std::vector<Action> legalActions() const {
        std::vector<Action> out;
        switch (r_.decision()) {
            case Round::Decision::kBid: {
                out.push_back(kBid + 0);
                for (int i = 1; i <= 8; ++i)
                    if (r_.availableBids() & indexToBid(i)) out.push_back(kBid + i);
                break;
            }
            case Round::Decision::kAnnounce: {
                const int n = int(r_.offeredAnnounces().size());
                for (int m = 0; m < (1 << n); ++m) out.push_back(kDeclare + m);
                break;
            }
            case Round::Decision::kPlay:
                for (Mask m = r_.legalMoves(); m; m &= m - 1) out.push_back(lowestCard(m));
                break;
            case Round::Decision::kDone: break;
        }
        return out;
    }

    void apply(Action a) {
        switch (r_.decision()) {
            case Round::Decision::kBid: r_.bid(indexToBid(a - kBid)); break;
            case Round::Decision::kAnnounce: {
                std::vector<Announce> chosen;
                const auto& offered = r_.offeredAnnounces();
                for (int i = 0; i < int(offered.size()); ++i)
                    if ((a - kDeclare) & (1 << i)) chosen.push_back(offered[std::size_t(i)]);
                r_.declare(chosen);
                break;
            }
            case Round::Decision::kPlay:
                r_.play(a, beloteAllowed(r_.hand(r_.toMove()), r_.contract(), r_.trick(), a));
                break;
            case Round::Decision::kDone: break;
        }
    }

    double payoff(int seat) const {
        const RoundResult& res = r_.result();
        return teamOf(seat) == 0 ? double(res.sn - res.ew) : double(res.ew - res.sn);
    }

    /**
     * The deal with what `observer` cannot see redrawn consistently with what it
     * knows. In the auction: its five cards stay, every other card is dealt
     * anew and the bids are replayed (bids do not depend on the cards). Later:
     * the world sampler (voids, declared combinations, belot) and a replay of
     * the public record.
     */
    FrameworkGame determinize(int observer, std::mt19937_64& rng) const {
        if (r_.finished()) return *this;
        if (r_.decision() == Round::Decision::kBid) {
            const Mask mine = r_.hand(observer);
            std::vector<int> rest;
            for (int c = 0; c < kCards; ++c)
                if (!(mine & bit(c))) rest.push_back(c);
            std::shuffle(rest.begin(), rest.end(), rng);
            std::array<int, kCards> deck{};
            std::vector<bool> taken(kCards, false);
            int k = 0;
            for (Mask m = mine; m; m &= m - 1, ++k) {
                const int slot = 4 * k + observer;  // the observer's first five cards
                deck[std::size_t(slot)] = lowestCard(m);
                taken[std::size_t(slot)] = true;
            }
            std::size_t j = 0;
            for (int slot = 0; slot < kCards; ++slot)
                if (!taken[std::size_t(slot)]) deck[std::size_t(slot)] = rest[j++];
            Round w(deck, r_.first(), r_.hangingBefore());
            const auto& bids = r_.bids();
            while (w.bids().size() < bids.size() && w.decision() == Round::Decision::kBid)
                w.bid(bids[w.bids().size()].second);
            return FrameworkGame(std::move(w));
        }
        const DealSampler sampler(publicState(r_, observer));
        for (int attempt = 0; attempt < 4; ++attempt) {
            auto h = sampler.sample(rng, 300);
            if (!h) h = sampler.sampleUnchecked(rng);
            if (!h) break;
            if (auto w = sampler.replay(*h)) return FrameworkGame(std::move(*w));
        }
        return *this;  // no consistent world found (very rare): the true deal
    }

private:
    Round r_;
};

}  // namespace bg
