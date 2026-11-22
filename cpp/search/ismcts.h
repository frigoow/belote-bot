#pragma once

// Plain SO-ISMCTS card play (the M1 baseline ladder: ISMCTS-100/1000/5000),
// adapted from the Manille bot's solver_wrapper.

#include "belote/deal.h"

#include <cstdint>
#include <map>

namespace belote {

struct IsmctsOptions {
    unsigned iterations = 1000;  // total over all threads
    unsigned threads = 1;        // > 1: root-parallel trees, visits summed
    std::uint64_t seed = 0;      // 0: random_device; >0: deterministic (when threads == 1)
};

struct SearchResult {
    int best_move = kNoCard;
    std::map<int, unsigned> visits;  // card -> root visits
    unsigned total_visits = 0;
    double value = 0;                // visit-weighted mean reward of the root's children
};

// Searches the current player's card. The deal must be in phase kPlay.
SearchResult ismctsSearch(const Deal& deal, const IsmctsOptions& options);

}  // namespace belote
