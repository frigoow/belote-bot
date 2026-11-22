#include "search/belote_game.h"

#include "belote/match.h"

#include <ismcts/utility.h>

#include <random>

namespace belote {

namespace {

std::mt19937_64& threadRng() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

thread_local long fallback_count = 0;

}  // namespace

BeloteGame::BeloteGame(Deal deal, std::shared_ptr<const WorldSampler> sampler,
                       std::shared_ptr<std::mt19937_64> rng, std::uint64_t prng_seed)
    : deal_(std::move(deal)), sampler_(std::move(sampler)), rng_(std::move(rng)), prng_seed_(prng_seed) {}

long BeloteGame::fallbacks() { return fallback_count; }

ISMCTS::Game<int>::Clone BeloteGame::cloneAndRandomise(Player observer) const {
    // ISMCTS::Sequential executes on a std::async thread, so seed ISMCTS::prng()
    // on the first determinization inside that worker thread.
    if (prng_seed_ != 0 && !prng_seeded_) {
        ISMCTS::prng().seed(static_cast<std::mt19937::result_type>(
            prng_seed_ ^ (prng_seed_ >> 32) ^ 0x9E3779B9u));
        prng_seeded_ = true;
    }
    std::shared_ptr<const WorldSampler> local;
    const WorldSampler* sampler = sampler_.get();
    const int played = deal_.tricksPlayed() * kPlayers + deal_.trick().n;
    if (!sampler || sampler->observation().observer != int(observer) ||
        int(sampler->observation().plays.size()) != played) {
        local = std::make_shared<WorldSampler>(observe(deal_, int(observer)));
        sampler = local.get();
    }
    auto& rng = rng_ ? *rng_ : threadRng();
    std::optional<World> world = sampler->sample(rng, 500);
    if (!world) {
        // Every per-card admissible world replays legally; only the toon
        // details of an unchecked world may differ from what was announced.
        ++fallback_count;
        world = sampler->sampleUnchecked(rng);
    }
    return std::make_unique<BeloteGame>(sampler->replay(*world));
}

std::vector<int> BeloteGame::validMoves() const {
    std::vector<int> moves;
    for (Mask m = deal_.legalMoves(); m; m &= m - 1) moves.push_back(lowestCard(m));
    return moves;
}

double unitMargin(const Deal& finished, int team) {
    const RecordedDeal rec = recordDeal(finished.result(), finished.config());
    return double(rec.scores[team] - rec.scores[1 - team]);
}

double BeloteGame::getResult(Player player) const {
    return unitMargin(deal_, teamOf(int(player))) / 25.0;
}

}  // namespace belote
