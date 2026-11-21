#include "search/ismcts.h"

#include "search/belote_game.h"

#include <ismcts/sosolver.h>
#include <ismcts/tree/ucb1.h>

#include <memory>
#include <stdexcept>

namespace belote {

namespace {

template <class Trees>
SearchResult collect(const Trees& trees) {
    SearchResult r;
    double weighted = 0;
    for (const auto& tree : trees) {
        for (const auto& child : tree->children()) {
            const unsigned v = child->visits();
            r.visits[child->move()] += v;
            r.total_visits += v;
            const auto* ucb = dynamic_cast<const ISMCTS::UCBNode<int>*>(child.get());
            if (ucb && v > 0) weighted += v * ucb->ucbScore(0.0);  // mean reward
        }
    }
    unsigned most = 0;
    for (const auto& [card, v] : r.visits) {
        if (v > most) {
            most = v;
            r.best_move = card;
        }
    }
    if (r.total_visits) r.value = weighted / r.total_visits;
    return r;
}

}  // namespace

SearchResult ismctsSearch(const Deal& deal, const IsmctsOptions& options) {
    if (deal.phase() != Phase::kPlay) throw std::logic_error("ismctsSearch: not in the play phase");
    const Mask legal = deal.legalMoves();
    if (popcount(legal) == 1) {
        SearchResult r;
        r.best_move = lowestCard(legal);
        return r;
    }
    auto sampler = std::make_shared<const WorldSampler>(observe(deal, deal.currentPlayer()));
    if (options.threads <= 1) {
        std::shared_ptr<std::mt19937_64> rng;
        if (options.seed != 0) rng = std::make_shared<std::mt19937_64>(options.seed);
        const BeloteGame root(deal, sampler, rng, options.seed);
        ISMCTS::SOSolver<int> solver(options.iterations);
        solver(root);
        return collect(solver.currentTrees());
    }
    const BeloteGame root(deal, sampler);
    ISMCTS::SOSolver<int, ISMCTS::RootParallel> solver(options.iterations, options.threads);
    solver(root);
    return collect(solver.currentTrees());
}

}  // namespace belote
