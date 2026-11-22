#include "search/rollout.h"

#include "agents/agent.h"
#include "belote/sampler.h"
#include "search/belote_game.h"
#include "search/endgame.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace belote {

namespace {

std::mt19937_64& rolloutRng() {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

}  // namespace

RolloutResult rolloutSearch(const Deal& deal, const RolloutOptions& options) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("rolloutSearch: deal is not in kPlay");
    const Mask legal = deal.legalMoves();
    RolloutResult res;
    if (popcount(legal) == 1) {
        res.best_move = lowestCard(legal);
        res.move_values.push_back({res.best_move, 0.0});
        return res;
    }

    const int me = deal.currentPlayer();
    const int my_team = teamOf(me);
    const WorldSampler sampler(observe(deal, me));

    std::vector<World> worlds;
    if (sampler.count() > 0 && sampler.count() <= std::uint64_t(options.worlds)) {
        worlds = sampler.enumerate(std::uint64_t(options.worlds));
        res.exact_enumeration = !worlds.empty();
    }
    if (worlds.empty()) {
        std::mt19937_64 local_rng(options.seed);
        std::mt19937_64& rng = options.seed ? local_rng : rolloutRng();
        worlds.reserve(std::size_t(options.worlds));
        for (int i = 0; i < options.worlds; ++i) {
            auto w = sampler.sample(rng, 500);
            if (!w) w = sampler.sampleUnchecked(rng);
            worlds.push_back(*w);
        }
    }

    std::vector<int> moves;
    for (Mask m = legal; m; m &= m - 1) moves.push_back(lowestCard(m));
    std::vector<double> sums(moves.size(), 0.0);

    const bool use_ab = options.ab_handoff_trick >= 0 && deal.tricksPlayed() >= options.ab_handoff_trick;

    for (const World& w : worlds) {
        const Deal base = sampler.replay(w);
        if (use_ab) {
            const EndgameResult er = solveEndgame(base);
            for (std::size_t i = 0; i < moves.size(); ++i) {
                for (const auto& [c, val] : er.move_values) {
                    if (c == moves[i]) {
                        sums[i] += double(val);
                        break;
                    }
                }
            }
            continue;
        }
        for (std::size_t i = 0; i < moves.size(); ++i) {
            Deal sim = base;
            sim.play(moves[i]);
            while (sim.phase() == Phase::kPlay) sim.play(ruleChooseCard(sim));
            sums[i] += unitMargin(sim, my_team);
        }
    }

    res.worlds_used = int(worlds.size());
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const double avg = res.worlds_used ? sums[i] / res.worlds_used : 0.0;
        res.move_values.push_back({moves[i], avg});
        if (avg > best) {
            best = avg;
            res.best_move = moves[i];
        }
    }
    res.value = best;
    return res;
}

}  // namespace belote
