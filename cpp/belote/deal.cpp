#include "belote/deal.h"

#include <stdexcept>

namespace belote {

int trickWinnerIndex(const Trick& trick, int trump) {
    const int led = suitOf(trick.cards[0]);
    int best = 0;
    for (int i = 1; i < trick.n; ++i) {
        const int c = trick.cards[i];
        const int w = trick.cards[best];
        const bool c_trump = suitOf(c) == trump;
        const bool w_trump = suitOf(w) == trump;
        if (c_trump != w_trump) {
            if (c_trump) best = i;
        } else if (c_trump || (suitOf(c) == led && suitOf(w) == led)) {
            if (cardStrength(c, trump) > cardStrength(w, trump)) best = i;
        }
    }
    return best;
}

namespace {

// Trumps in `trumps` stronger than `card` (a trump).
Mask higherTrumps(Mask trumps, int card, int trump) {
    const int strength = cardStrength(card, trump);
    Mask higher = 0;
    for (Mask m = trumps; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (cardStrength(c, trump) > strength) higher |= bit(c);
    }
    return higher;
}

}  // namespace

Mask legalMask(Mask hand, const Trick& trick, int trump, int player, const Config& cfg) {
    if (trick.n == 0) return hand;

    const int led = suitOf(trick.cards[0]);
    const int w = trickWinnerIndex(trick, trump);
    const int winning_card = trick.cards[w];
    const bool partner_winning = trick.players[w] == partnerOf(player);
    const Mask follow = hand & suitMask(led);

    if (follow) {
        if (led != trump) return follow;
        // Trump led: go over an opponent ("zoals bij manil").
        if (partner_winning && !cfg.trump_lead_partner_must_overtrump) return follow;
        const Mask higher = higherTrumps(follow, winning_card, trump);
        return higher ? higher : follow;
    }

    // Can't follow suit.
    if (partner_winning) return hand;
    const Mask trumps = hand & suitMask(trump);
    if (!trumps) return hand;
    if (suitOf(winning_card) != trump) return trumps;  // kopen
    const Mask higher = higherTrumps(trumps, winning_card, trump);
    if (higher) return higher;                         // overtroeven
    return cfg.must_undertrump ? trumps : hand;        // onderkopen
}

DealResult scoreDeal(const Config& cfg, int takers, int trump, int dealer,
                     std::array<int, 2> card_points, std::array<int, 2> tricks_won,
                     const std::vector<Toon> (&declared)[kPlayers], int belot_team) {
    DealResult r;
    r.takers = takers;
    r.trump = trump;
    r.belot_team = belot_team;
    const int defenders = 1 - takers;

    std::array<int, 2> team_toon{0, 0};
    for (int p = 0; p < kPlayers; ++p) team_toon[teamOf(p)] += totalPoints(declared[p]);

    for (int t = 0; t < 2; ++t)
        if (tricks_won[t] == kTricks) r.ronde_team = t;

    if (r.ronde_team >= 0) {
        // The ronde team's toons all count; the other team's are void.
        card_points[r.ronde_team] = cfg.ronde_points;
        card_points[1 - r.ronde_team] = 0;
        r.toon_team = team_toon[r.ronde_team] > 0 ? r.ronde_team : -1;
    } else {
        r.toon_team = toonWinningTeam(declared, trump, dealer);
    }
    if (r.toon_team >= 0) r.toon_points[r.toon_team] = team_toon[r.toon_team];
    if (belot_team >= 0) r.belot_points[belot_team] = kBelotPoints;
    r.card_points = card_points;
    for (int t = 0; t < 2; ++t)
        r.totals[t] = r.card_points[t] + r.toon_points[t] + r.belot_points[t];

    const int T = r.totals[takers];
    const int D = r.totals[defenders];
    Outcome outcome;
    if (r.ronde_team >= 0) {
        outcome = r.ronde_team == takers ? Outcome::kMade : Outcome::kNat;
    } else if (T > D) {
        outcome = Outcome::kMade;
    } else if (T < D) {
        outcome = Outcome::kNat;
    } else {
        switch (cfg.tie_rule) {
            case TieRule::kHanging: outcome = Outcome::kHanging; break;
            case TieRule::kTakersMake: outcome = Outcome::kMade; break;
            case TieRule::kTakersFail: default: outcome = Outcome::kNat; break;
        }
    }
    r.outcome = outcome;

    switch (outcome) {
        case Outcome::kMade:
            r.credited = r.totals;
            break;
        case Outcome::kNat:
            // Defenders: every card point, the toon that scores, their belot.
            // Belot always stays with its holder.
            r.credited[defenders] = r.card_points[0] + r.card_points[1] +
                                    r.toon_points[0] + r.toon_points[1] +
                                    r.belot_points[defenders];
            r.credited[takers] = r.belot_points[takers];
            break;
        case Outcome::kHanging:
            r.credited[defenders] = D;
            r.credited[takers] = 0;
            r.held = T;
            break;
        default:
            break;
    }
    return r;
}

Deal::Deal(const Config& cfg, int dealer) : cfg_(cfg), dealer_(dealer), current_(nextPlayer(dealer)) {
    declare_.fill(cfg.toon_enabled);
}

Deal::Deal(const Config& cfg, int dealer, const std::array<int, kCards>& deck) : Deal(cfg, dealer) {
    Mask seen = 0;
    for (int c : deck) {
        if (c < 0 || c >= kCards || (seen & bit(c))) throw std::invalid_argument("deck is not a permutation");
        seen |= bit(c);
    }
    deck_ = deck;
    int next = 0;
    for (int packet : {3, 2}) {
        for (int i = 0; i < kPlayers; ++i) {
            const int p = (dealer + 1 + i) % kPlayers;
            for (int k = 0; k < packet; ++k) hands_[p] |= bit(deck[next++]);
        }
    }
    blik_ = deck[next];
}

Deal Deal::fromPlay(const Config& cfg, int dealer, int taker, int trump,
                    const std::array<Mask, kPlayers>& hands) {
    Mask all = 0;
    for (Mask h : hands) {
        if (popcount(h) != kTricks || (all & h)) throw std::invalid_argument("hands must be 4 disjoint 8-card hands");
        all |= h;
    }
    Deal d(cfg, dealer);
    d.hands_ = hands;
    d.taker_ = taker;
    d.trump_ = trump;
    d.startPlay();
    return d;
}

std::vector<Bid> Deal::legalBids() const {
    if (phase_ == Phase::kBid1) return {Bid::pass(), Bid::take()};
    if (phase_ != Phase::kBid2) return {};
    std::vector<Bid> bids{Bid::pass()};
    for (int s = 0; s < kSuits; ++s)
        if (s != suitOf(blik_) || cfg_.round2_allow_blik_suit) bids.push_back(Bid::choose(s));
    return bids;
}

void Deal::bid(Bid b) {
    bool legal = false;
    for (const Bid& l : legalBids()) legal = legal || l == b;
    if (!legal) throw std::logic_error("illegal bid");

    if (b.kind == Bid::kPass) {
        ++bids_made_;
        current_ = nextPlayer(current_);
        if (bids_made_ == kPlayers) phase_ = Phase::kBid2;
        if (bids_made_ == 2 * kPlayers) {
            phase_ = Phase::kDone;
            result_.outcome = Outcome::kAllPassed;
        }
        return;
    }

    taker_ = current_;
    trump_ = b.kind == Bid::kTake ? suitOf(blik_) : b.suit;
    // The taker always takes the blik, plus 2; everyone else 3, dealt in
    // packets clockwise from the player after the dealer.
    hands_[taker_] |= bit(blik_);
    int next = 21;
    for (int i = 0; i < kPlayers; ++i) {
        const int p = (dealer_ + 1 + i) % kPlayers;
        const int packet = p == taker_ ? 2 : 3;
        for (int k = 0; k < packet; ++k) hands_[p] |= bit(deck_[next++]);
    }
    startPlay();
}

void Deal::startPlay() {
    phase_ = Phase::kPlay;
    current_ = nextPlayer(dealer_);
    for (int p = 0; p < kPlayers; ++p) {
        if (cfg_.toon_enabled) toons_[p] = findToons(hands_[p], cfg_);
        const Mask kq = bit(cardOf(trump_, kK)) | bit(cardOf(trump_, kQ));
        if ((hands_[p] & kq) == kq) belot_holder_ = p;
    }
}

bool Deal::canSetDeclare(int player) const {
    if (!cfg_.toon_enabled || !cfg_.toon_optional || phase_ != Phase::kPlay || tricks_played_ > 0)
        return false;
    for (int i = 0; i < trick_.n; ++i)
        if (trick_.players[i] == player) return false;  // already played their first card
    return true;
}

void Deal::setDeclare(int player, bool declare) {
    if (!canSetDeclare(player)) throw std::logic_error("declaration can no longer change");
    declare_[player] = declare;
}

Mask Deal::legalMoves() const {
    if (phase_ != Phase::kPlay) return 0;
    return legalMask(hands_[current_], trick_, trump_, current_, cfg_);
}

void Deal::play(int card) {
    if (card < 0 || card >= kCards || !(legalMoves() & bit(card))) throw std::logic_error("illegal card");
    const int p = current_;
    const int card_number = tricks_played_ + 1;  // this player's n-th card

    if (card_number == 1 && declare_[p] && !toons_[p].empty()) announced_[p] = true;
    if (card_number == cfg_.toon_show_trick && announced_[p]) shown_[p] = true;
    if (p == belot_holder_ && suitOf(card) == trump_ && (rankOf(card) == kK || rankOf(card) == kQ)) {
        if (belot_first_rank_ < 0) belot_first_rank_ = rankOf(card);
        if (belot_first_rank_ == kK || !cfg_.belot_queen_first_forfeits) belot_announced_ = true;
    }

    hands_[p] &= ~bit(card);
    trick_.cards[trick_.n] = card;
    trick_.players[trick_.n] = p;
    ++trick_.n;
    if (trick_.n < kPlayers) {
        current_ = nextPlayer(p);
        return;
    }

    const int winner = trick_.players[trickWinnerIndex(trick_, trump_)];
    int points = 0;
    for (int c : trick_.cards) points += belote::cardPoints(c, trump_);
    if (tricks_played_ == kTricks - 1) points += kLastTrickBonus;
    card_points_[teamOf(winner)] += points;
    ++tricks_won_[teamOf(winner)];
    history_[tricks_played_] = trick_;
    trick_winners_[tricks_played_] = winner;
    ++tricks_played_;
    trick_ = Trick{};
    current_ = winner;
    if (tricks_played_ == kTricks) finish();
}

void Deal::finish() {
    phase_ = Phase::kDone;
    std::vector<Toon> declared[kPlayers];
    for (int p = 0; p < kPlayers; ++p)
        if (shown_[p]) declared[p] = toons_[p];
    const int belot_team = belot_announced_ ? teamOf(belot_holder_) : -1;
    result_ = scoreDeal(cfg_, teamOf(taker_), trump_, dealer_, card_points_, tricks_won_, declared, belot_team);
}

}  // namespace belote
