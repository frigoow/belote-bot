#pragma once

// What one player knows during the play phase, and what follows from it.
//
// observe() copies only public information out of a Deal, plus the
// observer's own hand and declare choice. Everything that samples, searches
// or encodes on behalf of a player starts from an Observation, so nothing
// downstream can read another player's cards by accident.
//
// inferConstraints() turns the public play into per-card facts: cards a
// player cannot hold (voids, no trump, no higher trump, belot) and cards a
// player must hold (the blik, shown toon, belot). These facts are exact for
// the legal-move rule and belot: a deal is consistent with the play so far
// if and only if it respects them. Toon announcements are a condition on a
// whole hand and are checked by replaying the deal (sampler.h).

#include "belote/card.h"
#include "belote/deal.h"
#include "belote/rules.h"
#include "belote/toon.h"

#include <array>
#include <vector>

namespace belote {

struct Play {
    int player = -1;
    int card = kNoCard;
};

struct Observation {
    Config cfg;
    int observer = -1;
    int dealer = -1;
    int taker = -1;
    int trump = kNoSuit;
    int blik = kNoCard;              // kNoCard for deals made with Deal::fromPlay
    int bids_made = -1;              // passes before taker bid (0..3 R1, 4..7 R2, -1 fromPlay)
    Mask hand = 0;                   // the observer's current hand
    bool declare_self = true;        // the observer's own declare choice
    std::vector<Play> plays;         // every card played so far, in order

    std::array<bool, kPlayers> announced{};
    std::array<Announcement, kPlayers> announcement{};
    std::array<bool, kPlayers> shown{};
    std::array<Mask, kPlayers> shown_cards{};  // all toon cards of a shown player
    std::array<int, kPlayers> shown_points{};  // total points of a shown player's toons
    int belot_announcer = -1;
    int belot_card = kNoCard;

    Mask played() const;                 // every card played so far
    Mask playedBy(int player) const;
    int handSize(int player) const { return kTricks - popcount(playedBy(player)); }
    int currentPlayer() const;
};

// The observer's view of a deal in phase kPlay or kDone.
Observation observe(const Deal& deal, int observer);

struct Constraints {
    std::array<Mask, kPlayers> excluded{};  // cards the player cannot hold now
    std::array<Mask, kPlayers> pinned{};    // cards the player must hold now
};

Constraints inferConstraints(const Observation& obs);

}  // namespace belote
