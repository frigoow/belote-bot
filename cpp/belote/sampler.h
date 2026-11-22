#pragma once

// Worlds consistent with an Observation: the hidden cards dealt to the three
// other players. Ported from the Manille bot's exact uniform sampler
// (manil-bot cpp/sampler/uniform.cpp): every hidden card has a domain of
// players that may hold it (observation.h), and a table counts the ways to
// complete the hands card by card, so a world is drawn exactly uniformly by
// walking the table.
//
// Toon announcements constrain whole hands. A sequence announcement fixes
// the announcer's longest run (a carré would have been announced instead),
// so the table also follows each such announcer's run through the cards in
// sequence order and only counts hands whose longest run matches. What is
// left (no higher carré, a beaten toon really losing, no extra toon beside a
// shown one) is checked by replaying the deal (consistent()); sample() rejects
// the rare worlds that fail, which keeps the result uniform.

#include "belote/deal.h"
#include "belote/observation.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <random>
#include <vector>

namespace belote {

struct World {
    std::array<Mask, kPlayers> hands{};  // current hands, observer included
};

class WorldSampler {
public:
    explicit WorldSampler(const Observation& obs);

    const Observation& observation() const { return obs_; }

    // Worlds that respect the per-card constraints and the announced
    // sequence lengths. Every consistent world is among them; with no toon
    // announced or shown they are all consistent.
    std::uint64_t count() const { return count_; }
    World sampleUnchecked(std::mt19937_64& rng) const;

    // Whether w respects the per-card constraints and the hand sizes.
    bool admits(const World& w) const;

    // Whether hand-level checks are needed at all.
    bool needsCheck() const { return needs_check_; }
    bool consistent(const World& w) const;

    // A world drawn uniformly from the consistent worlds, or nothing after
    // max_tries rejections.
    std::optional<World> sample(std::mt19937_64& rng, int max_tries = 2000) const;

    // Every consistent world, or nothing if count() exceeds max_worlds.
    std::vector<World> enumerate(std::uint64_t max_worlds) const;

    // The deal at the observed point with this world's hands: started from
    // each player's 8 cards, with the observed declare choices, and replayed.
    // Throws std::logic_error if the world contradicts the play.
    Deal replay(const World& w) const;

private:
    // Run state of the tracked players: 5 bits each (current run 0..5,
    // longest-run class 0/3/4/5+ in 2 bits).
    using Runs = std::uint32_t;
    static constexpr int kMaxTracked = 3;

    std::uint64_t key(int i, int r0, int r1, Runs runs) const {
        return (std::uint64_t(runs) << 14) | (std::uint64_t(i) << 8) | (std::uint64_t(r0) << 4) | std::uint64_t(r1);
    }
    // Applies card i, held by seat k of others_ (-1: none of them), to the runs.
    // Returns false if a tracked run went past its target.
    bool step(int i, int k, Runs& runs) const;
    std::uint64_t countFrom(int i, int r0, int r1, Runs runs);
    std::uint64_t lookup(int i, int r0, int r1, Runs runs) const;
    void enumerateFrom(int i, int r0, int r1, Runs runs, World& w, std::vector<World>& out) const;

    Observation obs_;
    std::array<int, 3> others_{};                 // the three hidden seats
    std::array<std::int8_t, kCards> owner_{};     // known holder among others_ (-1: none/hidden)
    std::array<std::uint8_t, kCards> domain_{};   // hidden card: bit k = others_[k] may hold it
    std::array<bool, kCards> hidden_{};
    std::array<int, kCards + 1> hidden_left_{};   // hidden cards at index >= i
    std::array<int, 3> track_slot_{-1, -1, -1};   // per seat of others_: tracked slot or -1
    std::array<int, kMaxTracked> target_{};       // longest-run class each tracked seat must end with
    int tracked_ = 0;
    std::unordered_map<std::uint64_t, std::uint64_t> memo_;
    std::uint64_t count_ = 0;
    int start_r0_ = 0, start_r1_ = 0;
    World base_;                                  // the observer's hand
    bool needs_check_ = false;
};

}  // namespace belote
