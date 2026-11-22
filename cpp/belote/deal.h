#pragma once

// One deal of Belgian Belote: the 5-card deal and the blik, the two bidding
// rounds, the completion deal, eight tricks, toon and belot, and the score
// (docs/RULES_BELGIAN.md §1–§6). The match around it (101, hanging points, dealer
// rotation, redeals) is in match.h.
//
// The legal-move rule is also exposed as the free function legalMask(), so
// samplers and play-outs can use it without a Deal.

#include "belote/card.h"
#include "belote/rules.h"
#include "belote/toon.h"

#include <array>
#include <vector>

namespace belote {

enum class Phase { kBid1, kBid2, kPlay, kDone };

enum class Outcome {
    kNone,       // deal not finished
    kAllPassed,  // nobody made trump: redeal
    kMade,
    kNat,        // takers failed: defenders take everything
    kHanging,    // tie under TieRule::kHanging: takers' points held over
};

struct Bid {
    enum Kind { kPass, kTake, kChoose } kind = kPass;
    int suit = kNoSuit;  // kChoose only

    static Bid pass() { return {kPass, kNoSuit}; }
    static Bid take() { return {kTake, kNoSuit}; }
    static Bid choose(int suit) { return {kChoose, suit}; }
    bool operator==(const Bid& o) const { return kind == o.kind && suit == o.suit; }
};

struct Trick {
    std::array<int, kPlayers> cards{kNoCard, kNoCard, kNoCard, kNoCard};
    std::array<int, kPlayers> players{-1, -1, -1, -1};
    int n = 0;
};

// Index into trick.cards of the card currently winning the first trick.n cards.
int trickWinnerIndex(const Trick& trick, int trump);

// The cards `player` may play from `hand` into `trick` (docs/RULES_BELGIAN.md §4).
Mask legalMask(Mask hand, const Trick& trick, int trump, int player, const Config& cfg);

// Everything the deal scored, in raw points (match.h converts to units).
struct DealResult {
    Outcome outcome = Outcome::kNone;
    int takers = -1;  // team
    int trump = kNoSuit;
    std::array<int, 2> card_points{0, 0};   // tricks + last trick (+ ronde bonus)
    std::array<int, 2> toon_points{0, 0};   // toon that scores, by owning team
    std::array<int, 2> belot_points{0, 0};
    std::array<int, 2> totals{0, 0};        // card + toon + belot, before the contract
    std::array<int, 2> credited{0, 0};      // what each team scores this deal
    int held = 0;                           // hanging: takers' total, held over
    int toon_team = -1;
    int belot_team = -1;
    int ronde_team = -1;
};

// The score of a finished deal. card_points are the raw trick points
// (summing to 162); `declared` are each player's declared toons.
DealResult scoreDeal(const Config& cfg, int takers, int trump, int dealer,
                     std::array<int, 2> card_points, std::array<int, 2> tricks_won,
                     const std::vector<Toon> (&declared)[kPlayers], int belot_team);

class Deal {
public:
    // A fresh deal from a shuffled deck: 3 + 2 cards each from the player
    // after the dealer, deck[20] is the blik, deck[21..31] the stock.
    Deal(const Config& cfg, int dealer, const std::array<int, kCards>& deck);

    // A deal that starts at the first trick with known 8-card hands (tests,
    // cross-checks, samplers). Replays can retain the original public auction.
    static Deal fromPlay(const Config& cfg, int dealer, int taker, int trump,
                         const std::array<Mask, kPlayers>& hands, int blik = kNoCard, int passes = -1);

    const Config& config() const { return cfg_; }
    Phase phase() const { return phase_; }
    int dealer() const { return dealer_; }
    int currentPlayer() const { return current_; }
    Mask hand(int player) const { return hands_[player]; }

    // Bidding (phases kBid1, kBid2).
    int blik() const { return blik_; }
    std::vector<Bid> legalBids() const;
    void bid(Bid b);
    int taker() const { return taker_; }
    int takers() const { return taker_ < 0 ? -1 : teamOf(taker_); }
    int trump() const { return trump_; }
    int bidsMade() const { return bids_made_; }  // passes before taker bid (0..3 R1, 4..7 R2, -1 fromPlay)

    // Toon. toons(p) is the best split of p's 8-card hand (private). Whether
    // p declares can be changed until p plays their first card; by default
    // everybody declares. announced/announcement/shown are public: a player
    // announces their best toon's kind (and a carré's rank) with their first
    // card, and shows all their toons with their toon_show_trick-th card if
    // their team holds the best toon (or always, with toon_losers_show).
    const std::vector<Toon>& toons(int player) const { return toons_[player]; }
    bool canSetDeclare(int player) const;
    void setDeclare(int player, bool declare);
    bool declares(int player) const { return declare_[player]; }  // p's own choice (private)
    bool announced(int player) const { return announced_[player]; }
    const Announcement& announcement(int player) const { return announcement_[player]; }  // if announced
    bool shown(int player) const { return shown_[player]; }
    int belotHolder() const { return belot_holder_; }        // private
    bool belotAnnounced() const { return belot_announced_; } // public
    int belotAnnouncer() const { return belot_announced_ ? belot_holder_ : -1; }  // public
    int belotCard() const { return belot_card_; }            // public: the card "belot" was said with

    // Play (phase kPlay).
    Mask legalMoves() const;
    void play(int card);
    const Trick& trick() const { return trick_; }
    int tricksPlayed() const { return tricks_played_; }
    const Trick& completedTrick(int i) const { return history_[i]; }
    int trickWinner(int i) const { return trick_winners_[i]; }
    std::array<int, 2> cardPoints() const { return card_points_; }

    // Phase kDone.
    const DealResult& result() const { return result_; }

private:
    explicit Deal(const Config& cfg, int dealer);
    void startPlay();
    void finish();
    bool showsToon(int player) const;

    Config cfg_;
    Phase phase_ = Phase::kBid1;
    int dealer_;
    int current_;
    std::array<Mask, kPlayers> hands_{};
    std::array<int, kCards> deck_{};
    int blik_ = kNoCard;
    int bids_made_ = 0;
    int taker_ = -1;
    int trump_ = kNoSuit;

    std::vector<Toon> toons_[kPlayers];
    std::array<bool, kPlayers> declare_{};
    std::array<bool, kPlayers> announced_{};
    std::array<Announcement, kPlayers> announcement_{};
    std::array<bool, kPlayers> shown_{};
    int belot_holder_ = -1;
    int belot_first_rank_ = -1;  // rank of the first of K/Q the holder played
    bool belot_announced_ = false;
    int belot_card_ = kNoCard;

    Trick trick_;
    int tricks_played_ = 0;
    std::array<Trick, kTricks> history_{};
    std::array<int, kTricks> trick_winners_{};
    std::array<int, 2> card_points_{0, 0};
    std::array<int, 2> tricks_won_{0, 0};
    DealResult result_;
};

}  // namespace belote
