#include "bg/bg.h"

#include <stdexcept>

namespace bg {

namespace {

constexpr int kTrumpOrders[8] = {1, 2, 7, 5, 8, 3, 4, 6};    // 7 8 9 10 J Q K A
constexpr int kPlainOrders[8] = {1, 2, 3, 7, 4, 5, 6, 8};
constexpr int kTrumpValues[8] = {0, 0, 14, 10, 20, 3, 4, 11};
constexpr int kPlainValues[8] = {0, 0, 0, 10, 2, 3, 4, 11};

inline int sign(int x) { return (x > 0) - (x < 0); }

// Cards of the same suit with a higher trump order.
Mask biggerTrumps(int card) {
    Mask m = 0;
    const int s = suitOf(card);
    for (int r = 0; r < 8; ++r)
        if (kTrumpOrders[r] > kTrumpOrders[rankOf(card)]) m |= bit(cardOf(s, r));
    return m;
}

// ValidCardsService: the led suit with a higher card if possible (all trumps,
// or trumps led in a suit contract).
Mask allTrumpsRule(Mask hand, const Trick& t, int led) {
    const Mask from_suit = hand & suitMask(led);
    if (!from_suit) return hand;
    int best = t.cards[0];
    for (int i = 1; i < t.n && i < 3; ++i)
        if (suitOf(t.cards[i]) == led && trumpOrder(t.cards[i]) > trumpOrder(best)) best = t.cards[i];
    const Mask bigger = from_suit & biggerTrumps(best);
    return bigger ? bigger : from_suit;
}

// Math.Round(points / 10.0): halves to even.
int roundHalfEven(int points) {
    const int q = points / 10, r = points % 10;
    if (r > 5) return q + 1;
    if (r < 5) return q;
    return (q % 2 == 0) ? q : q + 1;
}

void findSequences(std::vector<Announce>& out, int suit, unsigned suit_bits, unsigned required) {
    if (!suit_bits) return;
    int run = 0;
    for (int type = 0; type <= 8; ++type) {
        if (type < 8 && ((suit_bits >> type) & 1u)) {
            ++run;
            continue;
        }
        const unsigned run_types = ((1u << run) - 1u) << (type - run);
        if (run >= 3 && (required == 0 || (run_types & required))) {
            Announce a;
            a.type = kSeq3 + run - 3;
            a.card = cardOf(suit, type - 1);
            out.push_back(a);
        }
        run = 0;
    }
}

bool sameTeam(int a, int b) { return teamOf(a) == teamOf(b); }

}  // namespace

int trumpSuit(std::uint8_t c) {
    if (c & kClubs) return 0;
    if (c & kDiamonds) return 1;
    if (c & kHearts) return 2;
    if (c & kSpades) return 3;
    return -1;
}

int trumpOrder(int card) { return kTrumpOrders[rankOf(card)]; }
int plainOrder(int card) { return kPlainOrders[rankOf(card)]; }

int cardValue(int card, std::uint8_t contract) {
    if (contract & kAllTrumps) return kTrumpValues[rankOf(card)];
    if (contract & kNoTrumps) return kPlainValues[rankOf(card)];
    if ((contract & ~(kDouble | kReDouble)) == kPass) return 0;
    return suitOf(card) == trumpSuit(contract) ? kTrumpValues[rankOf(card)] : kPlainValues[rankOf(card)];
}

// ------------------------------------------------------------------ announces

int Announce::value() const {
    switch (type) {
        case kBelot: return 20;
        case kSeq3: return 20;
        case kSeq4: return 50;
        case kSeq5: case kSeq6: case kSeq7: case kSeq8: return 100;
        case kFourOfAKind: return 100;
        case kFourNines: return 150;
        case kFourJacks: return 200;
        default: return 0;
    }
}

int compareAnnounce(const Announce& a, const Announce* b) {
    if (!b) return 1;
    if (&a == b) return 0;
    if (a.value() != b->value()) return a.value() > b->value() ? 1 : -1;
    if (a.type != b->type) return a.type > b->type ? 1 : -1;
    if (a.type == kFourOfAKind) return sign(trumpOrder(a.card) - trumpOrder(b->card));
    return sign(rankOf(a.card) - rankOf(b->card));
}

Mask announceCards(const Announce& a) {
    const int type = rankOf(a.card);
    switch (a.type) {
        case kBelot: return 0;
        case kFourOfAKind: case kFourNines: case kFourJacks: return Mask(0x01010101u) << type;
        default: {
            const int length = a.type - kSeq3 + 3;
            const int lowest = type - length + 1;
            if (lowest < 0) throw std::logic_error("invalid sequence announce");
            return Mask(((1u << length) - 1u) << (lowest + suitOf(a.card) * 8));
        }
    }
}

std::vector<Announce> availableAnnounces(Mask hand) {
    std::vector<Announce> out;
    const unsigned clubs = hand & 0xFFu, diamonds = (hand >> 8) & 0xFFu, hearts = (hand >> 16) & 0xFFu,
                   spades = hand >> 24;
    const unsigned fours = clubs & diamonds & hearts & spades & 0b11111100u;
    for (unsigned rem = fours; rem; rem &= rem - 1) {
        const int type = __builtin_ctz(rem);
        Announce a;
        a.type = type == belote::kJ ? kFourJacks : type == belote::k9 ? kFourNines : kFourOfAKind;
        a.card = cardOf(3, type);
        out.push_back(a);
    }
    const unsigned free = ~fours;
    findSequences(out, 0, clubs & free, 0);
    findSequences(out, 1, diamonds & free, 0);
    findSequences(out, 2, hearts & free, 0);
    findSequences(out, 3, spades & free, 0);
    if (fours) {
        findSequences(out, 0, clubs, fours);
        findSequences(out, 1, diamonds, fours);
        findSequences(out, 2, hearts, fours);
        findSequences(out, 3, spades, fours);
    }
    return out;
}

static bool isSequence(int type) { return type >= kSeq3 && type <= kSeq8; }
static bool isFour(int type) { return type == kFourOfAKind || type == kFourNines || type == kFourJacks; }

void updateActiveAnnounces(std::vector<Announce>& announces) {
    const Announce* max_four = nullptr;
    const Announce* max_seq = nullptr;
    for (const Announce& a : announces) {
        if (a.type == kBelot) continue;
        if (isFour(a.type)) {
            if (compareAnnounce(a, max_four) > 0) max_four = &a;
        } else if (compareAnnounce(a, max_seq) > 0) {
            max_seq = &a;
        }
    }
    bool tie_across_teams = false;
    for (const Announce& a : announces)
        if (isSequence(a.type) && max_seq && compareAnnounce(a, max_seq) == 0 && !sameTeam(a.player, max_seq->player))
            tie_across_teams = true;
    for (Announce& a : announces) {
        bool active = false;
        if (a.type == kBelot) {
            active = true;
        } else if (isFour(a.type)) {
            active = compareAnnounce(a, max_four) >= 0 || (max_four && sameTeam(a.player, max_four->player));
        } else if (!tie_across_teams) {
            active = compareAnnounce(a, max_seq) >= 0 || (max_seq && sameTeam(a.player, max_seq->player));
        }
        a.scored = active ? 1 : 0;
    }
}

// ------------------------------------------------------------------ play rules

Mask legalCards(Mask hand, std::uint8_t contract, const Trick& t) {
    if (t.n == 0 || popcount(hand) == 1) return hand;
    const int led = suitOf(t.cards[0]);
    if (contract & kAllTrumps) return allTrumpsRule(hand, t, led);
    if (contract & kNoTrumps) {
        const Mask from_suit = hand & suitMask(led);
        return from_suit ? from_suit : hand;
    }
    const int trump = trumpSuit(contract);
    if (led == trump) return allTrumpsRule(hand, t, led);
    const Mask from_suit = hand & suitMask(led);
    if (from_suit) return from_suit;
    const Mask trumps = hand & suitMask(trump);
    if (!trumps) return hand;
    int biggest_trump = kNoCard;
    for (int i = 0; i < t.n; ++i)
        if (suitOf(t.cards[i]) == trump && (biggest_trump == kNoCard || trumpOrder(t.cards[i]) > trumpOrder(biggest_trump)))
            biggest_trump = t.cards[i];
    if (t.n > 1) {
        int biggest = biggest_trump;
        if (biggest == kNoCard) {
            biggest = t.cards[0];
            for (int i = 1; i < t.n && i < 3; ++i)
                if (suitOf(t.cards[i]) == led && plainOrder(t.cards[i]) > plainOrder(biggest)) biggest = t.cards[i];
        }
        if (t.cards[t.n - 2] == biggest) return hand;  // partner holds the trick
    }
    if (biggest_trump != kNoCard) {
        const Mask bigger = hand & biggerTrumps(biggest_trump);
        return bigger ? bigger : hand;
    }
    return trumps;
}

int trickWinnerIndex(std::uint8_t contract, const Trick& t) {
    const int led = suitOf(t.cards[0]);
    int best = 0;
    if (contract & kAllTrumps) {
        for (int i = 1; i < t.n; ++i)
            if (suitOf(t.cards[i]) == led && trumpOrder(t.cards[i]) > trumpOrder(t.cards[best])) best = i;
    } else if (contract & kNoTrumps) {
        for (int i = 1; i < t.n; ++i)
            if (suitOf(t.cards[i]) == led && plainOrder(t.cards[i]) > plainOrder(t.cards[best])) best = i;
    } else {
        const int trump = trumpSuit(contract);
        bool best_trump = led == trump;
        for (int i = 1; i < t.n; ++i) {
            const int c = t.cards[i];
            if (suitOf(c) == trump) {
                if (!best_trump || trumpOrder(c) > trumpOrder(t.cards[best])) {
                    best = i;
                    best_trump = true;
                }
            } else if (!best_trump && suitOf(c) == led && plainOrder(c) > plainOrder(t.cards[best])) {
                best = i;
            }
        }
    }
    return best;
}

bool beloteAllowed(Mask hand, std::uint8_t contract, const Trick& t, int card) {
    const int r = rankOf(card);
    if (r != belote::kQ && r != belote::kK) return false;
    if (contract & kNoTrumps) return false;
    if (contract & kAllTrumps) {
        if (t.n > 0 && suitOf(t.cards[0]) != suitOf(card)) return false;
    } else if (suitOf(card) != trumpSuit(contract)) {
        return false;
    }
    return hand & bit(cardOf(suitOf(card), r == belote::kQ ? belote::kK : belote::kQ));
}

// ------------------------------------------------------------------ scoring

int roundPoints(std::uint8_t contract, int points, bool winner) {
    const int q = points / 10, r = points % 10;
    if (contract & kAllTrumps) {
        if (r > 4) return q + 1;
        if (r == 4) return winner ? q : q + 1;
        return q;
    }
    if (contract & kNoTrumps) return roundHalfEven(points);
    if (r > 6) return q + 1;
    if (r == 6) return winner ? q : q + 1;
    return q;
}

RoundResult scoreRound(std::uint8_t contract, int contract_player, Mask won_sn, Mask won_ew,
                       const std::vector<Announce>& announces, int hanging, int last_trick_winner) {
    int sn = 0, ew = 0;
    for (const Announce& a : announces) {
        if (a.scored != 1) continue;
        (teamOf(a.player) == 0 ? sn : ew) += a.value();
    }
    for (Mask m = won_sn; m; m &= m - 1) sn += cardValue(lowestCard(m), contract);
    for (Mask m = won_ew; m; m &= m - 1) ew += cardValue(lowestCard(m), contract);
    (teamOf(last_trick_winner) == 0 ? sn : ew) += 10;
    return finalizeScore(contract, contract_player, sn, ew, won_sn == 0, won_ew == 0, hanging);
}

RoundResult finalizeScore(std::uint8_t contract, int contract_player, int sn_raw, int ew_raw, bool sn_none,
                          bool ew_none, int hanging) {
    RoundResult r;
    r.sn_total = sn_raw;
    r.ew_total = ew_raw;
    if (contract & kNoTrumps) {
        r.sn_total *= 2;
        r.ew_total *= 2;
    }
    if (sn_none) {
        r.ew_total += 90;
        r.no_tricks = true;
    }
    if (ew_none) {
        r.sn_total += 90;
        r.no_tricks = true;
    }
    const int sn = r.sn_total, ew = r.ew_total;
    const int ct = teamOf(contract_player);
    if (contract & (kDouble | kReDouble)) {
        const int coef = (contract & kReDouble) ? 4 : 2;
        const int all = roundPoints(contract, sn + ew, true) * coef + hanging;
        if (sn > ew) r.sn += all;
        else if (ew > sn) r.ew += all;
        else r.hanging = all;
    } else if (ct == 0 && sn < ew) {
        r.ew += roundHalfEven(sn + ew) + hanging;
    } else if (ct == 0 && sn == ew) {
        r.ew += roundPoints(contract, ew, true);
        r.hanging = hanging + roundPoints(contract, sn, false);
    } else if (ct == 1 && ew < sn) {
        r.sn += roundHalfEven(sn + ew) + hanging;
    } else if (ct == 1 && sn == ew) {
        r.sn += roundPoints(contract, sn, true);
        r.hanging = hanging + roundPoints(contract, ew, false);
    } else {
        r.sn = roundPoints(contract, sn, sn > ew);
        r.ew = roundPoints(contract, ew, ew > sn);
        if (sn > ew) r.sn += hanging;
        else if (ew > sn) r.ew += hanging;
    }
    return r;
}

// ------------------------------------------------------------------ the deal

Round::Round(const std::array<int, kCards>& deal, int first, int hanging)
    : deal_(deal), first_(first), hanging_(hanging), current_(first), contract_player_(first) {
    dealCards(0, 5);
    moveToNextBidder();
}

void Round::dealCards(int from, int each) {
    for (int i = 0; i < each; ++i)
        for (int s = 0; s < kPlayers; ++s) hands_[s] |= bit(deal_[std::size_t(from + 4 * i + s)]);
}

static std::uint8_t availableBidsFor(std::uint8_t contract, int contract_player, int player) {
    const std::uint8_t clean = contract & ~(kDouble | kReDouble);
    std::uint8_t avail = kPass;
    for (std::uint8_t f : {kClubs, kDiamonds, kHearts, kSpades, kNoTrumps, kAllTrumps})
        if (clean < f) avail |= f;
    if (contract != kPass && !(contract & kReDouble)) {
        const bool declaring = teamOf(player) == teamOf(contract_player);
        if (contract & kDouble) {
            if (declaring) avail |= kReDouble;
        } else if (!declaring) {
            avail |= kDouble;
        }
    }
    return avail;
}

bool Round::record(std::uint8_t b) {
    bids_.push_back({current_, b});
    consecutive_passes_ = b == kPass ? consecutive_passes_ + 1 : 0;
    return (contract_ == kPass && consecutive_passes_ == 4) || (contract_ != kPass && consecutive_passes_ == 3);
}

void Round::moveToNextBidder() {
    for (;;) {
        const std::uint8_t avail = availableBidsFor(contract_, contract_player_, current_);
        if (avail != kPass) {
            available_bids_ = avail;
            decision_ = Decision::kBid;
            return;
        }
        if (record(kPass)) {
            startPlay();
            return;
        }
        current_ = nextSeat(current_);
    }
}

void Round::bid(std::uint8_t b) {
    if (decision_ != Decision::kBid) throw std::logic_error("bg::Round::bid: not bidding");
    if ((b != kPass && (b & (b - 1))) || (b != kPass && !(available_bids_ & b))) throw std::logic_error("bg::Round::bid: illegal bid");
    if (b == kDouble || b == kReDouble) {
        contract_ &= ~(kDouble | kReDouble);
        contract_ |= b;
    } else if (b != kPass) {
        contract_ = b;
        contract_player_ = current_;
    }
    if (record(b)) {
        startPlay();
        return;
    }
    current_ = nextSeat(current_);
    moveToNextBidder();
}

void Round::startPlay() {
    if (contract_ == kPass) {
        result_ = RoundResult{};
        result_.hanging = hanging_;
        decision_ = Decision::kDone;
        current_ = -1;
        return;
    }
    dealCards(20, 3);
    current_ = first_;
    trick_number_ = 1;
    trick_ = Trick{};
    announced_ = false;
    moveToNextDecision();
}

void Round::moveToNextDecision() {
    while (decision_ != Decision::kDone) {
        if (!announced_) {
            announced_ = true;
            if (trick_number_ == 1 && !(contract_ & kNoTrumps)) {
                offered_ = availableAnnounces(hands_[current_]);
                if (!offered_.empty()) {
                    decision_ = Decision::kAnnounce;
                    return;
                }
            }
        }
        legal_ = legalCards(hands_[current_], contract_, trick_);
        if (popcount(legal_) > 1) {
            decision_ = Decision::kPlay;
            return;
        }
        playCard(lowestCard(legal_), false);  // forced: no belote
    }
}

void Round::declare(const std::vector<Announce>& chosen) {
    if (decision_ != Decision::kAnnounce) throw std::logic_error("bg::Round::declare: not declaring");
    Mask declared = 0;
    for (Announce a : chosen) {
        bool offered = false;
        for (const Announce& o : offered_) offered = offered || (o.type == a.type && o.card == a.card);
        if (!offered) continue;
        const Mask cards = announceCards(a);
        if (declared & cards) continue;
        declared |= cards;
        a.player = current_;
        a.scored = -1;
        announces_.push_back(a);
    }
    offered_.clear();
    moveToNextDecision();
}

void Round::play(int card, bool belote) {
    if (decision_ != Decision::kPlay || !(legal_ & bit(card))) throw std::logic_error("bg::Round::play: illegal card");
    if (belote && beloteAllowed(hands_[current_], contract_, trick_, card)) {
        Announce a;
        a.type = kBelot;
        a.card = card;
        a.player = current_;
        a.scored = 1;
        announces_.push_back(a);
    } else {
        belote = false;
    }
    playCard(card, belote);
    moveToNextDecision();
}

void Round::playCard(int card, bool belote) {
    hands_[current_] &= ~bit(card);
    plays_.push_back({current_, {card, belote}});
    trick_.cards[trick_.n] = card;
    trick_.players[trick_.n] = current_;
    ++trick_.n;
    current_ = nextSeat(current_);
    announced_ = false;
    if (trick_.n < kPlayers) return;
    const int winner = trick_.players[trickWinnerIndex(contract_, trick_)];
    winners_.push_back(winner);
    for (int i = 0; i < kPlayers; ++i) won_[teamOf(winner)] |= bit(trick_.cards[i]);
    tricks_.push_back(trick_);
    current_ = winner;
    if (trick_number_ == kTricks) {
        last_trick_winner_ = winner;
        finish();
        return;
    }
    ++trick_number_;
    trick_ = Trick{};
    if (trick_number_ == 2) updateActiveAnnounces(announces_);
}

void Round::finish() {
    result_ = scoreRound(contract_, contract_player_, won_[0], won_[1], announces_, hanging_, last_trick_winner_);
    decision_ = Decision::kDone;
    current_ = -1;
}

// ------------------------------------------------------------------ the game

void GameScore::record(std::uint8_t contract, const RoundResult& r) {
    sn += r.sn;
    ew += r.ew;
    hanging = r.hanging;
    ++rounds;
    const bool decided = contract != kPass && !r.no_tricks;
    if (sn >= 151 && sn > ew && r.sn > 0 && decided) winner = 0;
    else if (ew >= 151 && ew > sn && r.ew > 0 && decided) winner = 1;
    if (winner < 0) first = nextSeat(first);
}

}  // namespace bg
