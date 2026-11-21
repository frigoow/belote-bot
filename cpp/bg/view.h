#pragma once

// What one seat knows during a Bulgarian Belot deal, and the hidden deals
// consistent with it. Our own design (the Manille/Belote sampler approach):
// per-card facts read from the public play (voids, ceilings the legal-move
// rule implies, belot pins), an exact uniform draw over the assignments that
// respect them, then a replay through bg::Round that rejects the rare draws
// that contradict a declaration (opponents' declarations are known by type
// only, as in Kostov's engine).

#include "bg/bg.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace bg {

struct PublicPlay {
    int seat = -1;
    int card = kNoCard;
    bool belote = false;
};

struct Declared {
    int seat = -1;
    int type = 0;
    int card = kNoCard;  // known only for the observer's own declarations
};

// Everything the observer may know when it is to play.
struct PublicState {
    int first = 0;  // first to bid and lead this deal
    int me = 0;
    Mask hand = 0;  // the observer's current cards
    std::uint8_t contract = kPass;
    int declarer = -1;
    std::vector<std::pair<int, std::uint8_t>> bids;  // every bid, forced passes included
    std::vector<Declared> declared;                  // non-belot declarations, in order
    std::vector<PublicPlay> plays;                   // every card played, in order
    int sn = 0, ew = 0, hanging = 0;                 // game score before this deal

    Mask played() const;
    Mask playedBy(int seat) const;
    int handSize(int seat) const { return kTricks - popcount(playedBy(seat)); }
    int toMove() const;
};

// The observer's view of a Round (tests, self-play): its hand and the public record.
PublicState publicState(const Round& r, int observer, int sn = 0, int ew = 0);

struct Facts {
    std::array<Mask, kPlayers> excluded{};  // cards a seat cannot hold now
    std::array<Mask, kPlayers> pinned{};    // cards a seat must hold now (belot partner card, four J/9)
};
Facts inferFacts(const PublicState& ps);

// Current hands of all four seats (the observer's included).
using Hands = std::array<Mask, kPlayers>;

class DealSampler {
public:
    explicit DealSampler(const PublicState& ps);
    const PublicState& state() const { return ps_; }
    const Facts& facts() const { return facts_; }
    // Uniform over the hands respecting the per-card facts and hand sizes.
    std::optional<Hands> sampleUnchecked(std::mt19937_64& rng) const;
    // Uniform over the consistent hands (declarations checked by replay).
    std::optional<Hands> sample(std::mt19937_64& rng, int max_tries = 300) const;
    // The deal replayed up to the observer's decision, or nothing if the hands
    // contradict the public record.
    std::optional<Round> replay(const Hands& hands) const;
    bool admits(const Hands& hands) const;  // per-card facts and sizes

private:
    PublicState ps_;
    Facts facts_;
    std::array<int, 3> others_{};
    std::vector<int> hidden_;                  // hidden cards
    std::vector<std::uint8_t> domain_;          // per hidden card: bit k = others_[k] may hold it
    std::vector<double> ways_;                  // [i][a][b] completions of cards i.. with capacities a, b (c implied)
    std::array<int, 3> need_{};
    // Placements of the opponents' declared combinations (empty: none, or too many).
    struct Placement {
        std::vector<std::uint8_t> domain;
        double weight;
    };
    std::vector<Placement> placements_;
    double placement_total_ = 0;
    double ways(std::size_t i, int a, int b) const;
};

// Replays a deal from every seat's cards at the start of play (current hands +
// cards played) through the public record, or nothing if they contradict it.
// `before_play` (optional) sees the position before every recorded card that
// was a choice (not auto-played).
using PlayHook = std::function<void(const Round&, const PublicPlay&)>;
std::optional<Round> replayDeal(const PublicState& ps, const Hands& start_hands, const PlayHook* before_play = nullptr);

}  // namespace bg
