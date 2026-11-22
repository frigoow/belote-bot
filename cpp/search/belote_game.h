#pragma once

// Card play as a game for ismcsolver (third_party/ismcsolver, MIT, Steven
// Franzen), the ISMCTS library the Manille bot uses.
//
// cloneAndRandomise(observer) draws a world from what the observer knows
// (belote/sampler.h) and replays the deal from it, so every determinization
// is consistent with the whole public history: voids, trump limits, the blik,
// toon announcements and showing, belot. The search sees no hidden card.
//
// getResult is the deal's margin in recorded units (recordDeal), scaled by
// 1/25: the contract cliff, the ronde and the toon all count as on the score
// sheet. Held-over points of a hanging deal count for neither team.

#include "belote/deal.h"
#include "belote/sampler.h"

#include <ismcts/game.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace belote {

class BeloteGame : public ISMCTS::POMGame<int> {
public:
    // `sampler` (optional) is the prepared sampler of the root's player, so
    // the root's per-iteration determinizations don't rebuild the table.
    // `rng` (optional) provides a deterministic RNG stream when single-threaded.
    explicit BeloteGame(Deal deal, std::shared_ptr<const WorldSampler> sampler = nullptr,
                        std::shared_ptr<std::mt19937_64> rng = nullptr, std::uint64_t prng_seed = 0);

    Clone cloneAndRandomise(Player observer) const override;
    Player currentPlayer() const override { return Player(deal_.currentPlayer()); }
    std::vector<Player> players() const override { return {0, 1, 2, 3}; }
    std::vector<int> validMoves() const override;
    void doMove(int card) override { deal_.play(card); }
    double getResult(Player player) const override;

    const Deal& deal() const { return deal_; }

    // Determinizations that fell back to an unchecked world because
    // rejection sampling gave up (per thread).
    static long fallbacks();

private:
    Deal deal_;
    std::shared_ptr<const WorldSampler> sampler_;
    std::shared_ptr<std::mt19937_64> rng_;
    std::uint64_t prng_seed_ = 0;
    mutable bool prng_seeded_ = false;
};

// The deal's value for `team` in recorded units (held-over points excluded).
double unitMargin(const Deal& finished, int team);

}  // namespace belote
