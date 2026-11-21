#pragma once

// Bulgarian Bridge-Belot, rule for rule as Nikolay Kostov's BelotGameEngine
// (github.com/NikolayIT/BelotGameEngine, MIT, src/Belot.Engine) plays it, so
// our players can be measured against his inside his engine and searched in
// ours with the same outcomes. Cross-checked card for card against logs of his
// engine (bg/oracle, tests/test_bg.cpp). RULES_BG.md describes the rules.
//
// Seats S=0, E=1, N=2, W=3 in playing order; S/N are team 0, E/W team 1.
// Cards and masks are those of belote/card.h (suit * 8 + rank, 7 8 9 10 J Q K A).
// Bid values are Kostov's BidType flags, so logs and bridges map one to one.

#include "belote/card.h"

#include <array>
#include <cstdint>
#include <vector>

namespace bg {

using belote::Mask;
using belote::bit;
using belote::cardOf;
using belote::kCards;
using belote::kNoCard;
using belote::kPlayers;
using belote::kTricks;
using belote::lowestCard;
using belote::popcount;
using belote::rankOf;
using belote::suitMask;
using belote::suitOf;

inline constexpr int teamOf(int seat) { return seat & 1; }
inline constexpr int nextSeat(int seat) { return (seat + 1) & 3; }
inline constexpr int partnerOf(int seat) { return (seat + 2) & 3; }

// BidType flags.
enum : std::uint8_t {
    kPass = 0,
    kClubs = 1,
    kDiamonds = 2,
    kHearts = 4,
    kSpades = 8,
    kNoTrumps = 16,
    kAllTrumps = 32,
    kDouble = 64,
    kReDouble = 128,
};

// The contract's trump suit, or -1 for no trumps / all trumps / none.
int trumpSuit(std::uint8_t contract);
// Card points under the contract (no-trumps doubling happens in the score).
int cardValue(int card, std::uint8_t contract);
// Order within a suit: trump order (J 9 A 10 K Q 8 7) or plain order (A 10 K Q J 9 8 7), 1..8.
int trumpOrder(int card);
int plainOrder(int card);

enum AnnounceType : int {
    kBelot = 1,
    kSeq3 = 2,
    kSeq4 = 3,
    kSeq5 = 4,
    kSeq6 = 5,
    kSeq7 = 6,
    kSeq8 = 7,
    kFourOfAKind = 8,
    kFourNines = 9,
    kFourJacks = 10,
};

struct Announce {
    int type = 0;
    int card = kNoCard;  // a sequence's top card; a carré's card in spades; a belot's played card
    int player = -1;
    int scored = -1;     // -1 not decided yet, 0 no, 1 yes
    int value() const;
};

// Kostov's Announce.CompareTo: value, then type, then (carré) trump order or (sequence) top rank.
int compareAnnounce(const Announce& a, const Announce* b);
Mask announceCards(const Announce& a);
// Combinations a hand may declare: carrés, sequences of the other cards, then
// the sequences through a carré card (alternatives).
std::vector<Announce> availableAnnounces(Mask hand);
// Before trick 2: which declarations score (belotes always do).
void updateActiveAnnounces(std::vector<Announce>& announces);

struct Trick {
    std::array<int, kPlayers> cards{kNoCard, kNoCard, kNoCard, kNoCard};
    std::array<int, kPlayers> players{-1, -1, -1, -1};
    int n = 0;
};

Mask legalCards(Mask hand, std::uint8_t contract, const Trick& trick);
int trickWinnerIndex(std::uint8_t contract, const Trick& trick);
bool beloteAllowed(Mask hand, std::uint8_t contract, const Trick& trick, int card);

struct RoundResult {
    int sn = 0, ew = 0;              // game points recorded this deal
    int sn_total = 0, ew_total = 0;  // raw points (cards, last 10, declarations, x2 no trumps, +90)
    bool no_tricks = false;          // one team took no trick
    int hanging = 0;                 // points held over to the next deal
};

// ScoreManager.GetScore. won[t]: cards team t took; announces with scored == 1 count.
RoundResult scoreRound(std::uint8_t contract, int contract_player, Mask won_sn, Mask won_ew,
                       const std::vector<Announce>& announces, int hanging, int last_trick_winner);
// The second half of scoreRound: from each team's raw total (cards, last 10,
// scoring declarations), before the no-trumps doubling and the capot bonus.
RoundResult finalizeScore(std::uint8_t contract, int contract_player, int sn_raw, int ew_raw, bool sn_none,
                          bool ew_none, int hanging);
int roundPoints(std::uint8_t contract, int points, bool winner);

// One deal: auction, declarations, eight tricks, score. Seats whose only
// option is forced (pass, a single legal card, nothing to declare) are moved
// for automatically, exactly where Kostov's engine does it.
class Round {
public:
    enum class Decision { kBid, kAnnounce, kPlay, kDone };

    // `deal`: the 32 cards in dealing order (5 one at a time to S E N W, then 3).
    Round(const std::array<int, kCards>& deal, int first, int hanging = 0);

    Decision decision() const { return decision_; }
    int toMove() const { return current_; }
    bool finished() const { return decision_ == Decision::kDone; }

    // Bidding.
    std::uint8_t availableBids() const { return available_bids_; }
    void bid(std::uint8_t b);
    // Declarations (trick 1, before the seat's first card): the offered list,
    // then the chosen ones in order; of overlapping ones only the first counts.
    const std::vector<Announce>& offeredAnnounces() const { return offered_; }
    void declare(const std::vector<Announce>& chosen);
    // Play.
    Mask legalMoves() const { return legal_; }
    void play(int card, bool belote);

    // State.
    int first() const { return first_; }
    int hangingBefore() const { return hanging_; }
    Mask hand(int seat) const { return hands_[seat]; }
    std::uint8_t contract() const { return contract_; }
    int contractPlayer() const { return contract_player_; }
    const std::vector<std::pair<int, std::uint8_t>>& bids() const { return bids_; }
    const std::vector<Announce>& announces() const { return announces_; }
    const Trick& trick() const { return trick_; }
    int trickNumber() const { return trick_number_; }  // 1-based
    const std::vector<Trick>& tricks() const { return tricks_; }
    const std::vector<int>& trickWinners() const { return winners_; }
    const std::vector<std::pair<int, std::pair<int, bool>>>& plays() const { return plays_; }  // seat, (card, belote)
    Mask won(int team) const { return won_[team]; }
    const RoundResult& result() const { return result_; }

private:
    void dealCards(int from, int each);
    bool record(std::uint8_t b);
    void moveToNextBidder();
    void startPlay();
    void moveToNextDecision();
    void playCard(int card, bool belote);
    void finish();

    std::array<int, kCards> deal_{};
    std::array<Mask, kPlayers> hands_{};
    int first_ = 0;
    int hanging_ = 0;
    Decision decision_ = Decision::kBid;
    int current_ = 0;

    std::uint8_t contract_ = kPass;
    int contract_player_ = -1;
    std::uint8_t available_bids_ = 0;
    int consecutive_passes_ = 0;
    std::vector<std::pair<int, std::uint8_t>> bids_;

    std::vector<Announce> offered_;
    std::vector<Announce> announces_;
    bool announced_ = false;
    Mask legal_ = 0;
    Trick trick_;
    int trick_number_ = 0;
    std::vector<Trick> tricks_;
    std::vector<int> winners_;
    std::vector<std::pair<int, std::pair<int, bool>>> plays_;
    std::array<Mask, 2> won_{};
    int last_trick_winner_ = -1;
    RoundResult result_;
};

// The game around the deals: 151 points, hanging points, the rotating first seat.
struct GameScore {
    int sn = 0, ew = 0, hanging = 0;
    int first = 0;   // first to bid and lead in the next deal
    int rounds = 0;  // deals played
    int winner = -1; // team, once the game is over
    // Adds a finished deal (a passed-out deal keeps the hanging points) and
    // decides the winner: 151+, ahead, scored this deal, not a capot or a pass.
    void record(std::uint8_t contract, const RoundResult& r);
};

}  // namespace bg
