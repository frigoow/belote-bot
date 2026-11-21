#pragma once

// Toon: sequences (derde, vierde, vijfde+) and carrés (RULES.md §5.1).

#include "belote/card.h"
#include "belote/rules.h"

#include <vector>

namespace belote {

struct Toon {
    bool carre = false;
    int suit = kNoSuit;  // sequences only
    int rank = 0;        // carré: its rank; sequence: its top rank
    int length = 0;      // sequence length (4 for a carré)
    Mask cards = 0;
    int points = 0;
};

// What a player announces with their first card: the kind, not the cards.
enum class ToonKind { kDerde, kVierde, kVijfde, kCarre };
ToonKind kindOf(const Toon& t);

int sequencePoints(int length);
int carrePoints(int rank, const Config& cfg);  // 0: not a toon

// The toons a hand declares. A card is used in at most one toon; carrés may
// be left out to keep a longer sequence, and the split worth the most points
// is chosen. Sequences are maximal runs and are never split.
std::vector<Toon> findToons(Mask hand, const Config& cfg);

// Comparison key of one toon: a higher key is a better toon. Seat order
// (who sits first after the dealer) breaks the last tie and is applied by
// the caller, since it is a property of the holder, not the toon.
long toonKey(const Toon& t, int trump);

// The team whose toon scores, given each player's declared toons, or -1 if
// nobody declared anything.
int toonWinningTeam(const std::vector<Toon> (&declared)[kPlayers], int trump, int dealer);

int totalPoints(const std::vector<Toon>& toons);

}  // namespace belote
