#include "belote/card.h"

#include <sstream>
#include <stdexcept>

namespace belote {

namespace {
const char* const kRankNames[kRanks] = {"7", "8", "9", "10", "J", "Q", "K", "A"};
const char kSuitNames[kSuits] = {'C', 'D', 'H', 'S'};
}  // namespace

std::string cardName(int card) {
    if (card < 0 || card >= kCards) return "--";
    return std::string(kRankNames[rankOf(card)]) + kSuitNames[suitOf(card)];
}

int parseCard(const std::string& name) {
    if (name.size() < 2) return kNoCard;
    const std::string rank = name.substr(0, name.size() - 1);
    const char suit_char = name.back();
    int suit = kNoSuit;
    for (int s = 0; s < kSuits; ++s)
        if (kSuitNames[s] == suit_char) suit = s;
    if (suit == kNoSuit) return kNoCard;
    for (int r = 0; r < kRanks; ++r)
        if (rank == kRankNames[r]) return cardOf(suit, r);
    return kNoCard;
}

Mask parseCards(const std::string& names) {
    std::istringstream in(names);
    std::string token;
    Mask m = 0;
    while (in >> token) {
        const int card = parseCard(token);
        if (card == kNoCard) throw std::invalid_argument("bad card name: " + token);
        m |= bit(card);
    }
    return m;
}

std::string maskString(Mask m) {
    std::string out;
    while (m) {
        const int card = lowestCard(m);
        m &= m - 1;
        if (!out.empty()) out += ' ';
        out += cardName(card);
    }
    return out;
}

}  // namespace belote
