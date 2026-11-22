#include "belote/observation.h"

#include <stdexcept>

namespace belote {

Mask Observation::played() const {
    Mask m = 0;
    for (const Play& pl : plays) m |= bit(pl.card);
    return m;
}

Mask Observation::playedBy(int player) const {
    Mask m = 0;
    for (const Play& pl : plays)
        if (pl.player == player) m |= bit(pl.card);
    return m;
}

int Observation::currentPlayer() const {
    // Replaying the tricks is cheap and keeps the observation minimal.
    if (plays.empty()) return nextPlayer(dealer);
    Trick t;
    int next = nextPlayer(dealer);
    for (const Play& pl : plays) {
        t.cards[t.n] = pl.card;
        t.players[t.n] = pl.player;
        ++t.n;
        next = nextPlayer(pl.player);
        if (t.n == kPlayers) {
            next = t.players[trickWinnerIndex(t, trump)];
            t = Trick{};
        }
    }
    return next;
}

Observation observe(const Deal& d, int observer) {
    if (d.phase() != Phase::kPlay && d.phase() != Phase::kDone)
        throw std::logic_error("observe: the deal is not in the play phase");
    Observation o;
    o.cfg = d.config();
    o.observer = observer;
    o.dealer = d.dealer();
    o.taker = d.taker();
    o.trump = d.trump();
    o.blik = d.blik();
    o.bids_made = d.bidsMade();
    o.hand = d.hand(observer);
    o.declare_self = d.declares(observer);
    for (int i = 0; i < d.tricksPlayed(); ++i) {
        const Trick& t = d.completedTrick(i);
        for (int k = 0; k < t.n; ++k) o.plays.push_back({t.players[k], t.cards[k]});
    }
    const Trick& t = d.trick();
    for (int k = 0; k < t.n; ++k) o.plays.push_back({t.players[k], t.cards[k]});

    for (int p = 0; p < kPlayers; ++p) {
        o.announced[p] = d.announced(p);
        if (o.announced[p]) o.announcement[p] = d.announcement(p);
        o.shown[p] = d.shown(p);
        if (o.shown[p]) {
            for (const Toon& toon : d.toons(p)) o.shown_cards[p] |= toon.cards;
            o.shown_points[p] = totalPoints(d.toons(p));
        }
    }
    o.belot_announcer = d.belotAnnouncer();
    o.belot_card = d.belotCard();
    return o;
}

namespace {

Mask strongerTrumps(int card, int trump) {
    Mask m = 0;
    for (int r = 0; r < kRanks; ++r)
        if (kTrumpStrength[r] > kTrumpStrength[rankOf(card)]) m |= bit(cardOf(trump, r));
    return m;
}

}  // namespace

Constraints inferConstraints(const Observation& o) {
    Constraints c;
    const int trump = o.trump;
    const Config& cfg = o.cfg;
    Mask played = 0;
    Trick t;

    for (const Play& pl : o.plays) {
        const int p = pl.player;
        const int card = pl.card;

        // The legal-move rule (docs/RULES_BELGIAN.md §4), read backwards: every way the
        // play could have been illegal is a set of cards p did not hold.
        if (t.n > 0) {
            const int led = suitOf(t.cards[0]);
            const int w = trickWinnerIndex(t, trump);
            const int winning = t.cards[w];
            const bool partner_winning = t.players[w] == partnerOf(p);
            if (suitOf(card) != led) {
                c.excluded[p] |= suitMask(led);
                if (!partner_winning) {
                    if (suitOf(card) != trump) {
                        if (suitOf(winning) != trump || cfg.must_undertrump)
                            c.excluded[p] |= suitMask(trump);
                        else
                            c.excluded[p] |= strongerTrumps(winning, trump);
                    } else if (suitOf(winning) == trump &&
                               cardStrength(card, trump) < cardStrength(winning, trump)) {
                        c.excluded[p] |= strongerTrumps(winning, trump);
                    }
                }
            } else if (led == trump && (!partner_winning || cfg.trump_lead_partner_must_overtrump) &&
                       cardStrength(card, trump) < cardStrength(winning, trump)) {
                c.excluded[p] |= strongerTrumps(winning, trump);
            }
        }

        // Belot: the holder announces with the first of K/Q (with only the
        // king under belot_queen_first_forfeits). A K/Q played without the
        // announcement, the other still out, means p does not hold the other.
        if (suitOf(card) == trump && (rankOf(card) == kK || rankOf(card) == kQ)) {
            const int other = cardOf(trump, rankOf(card) == kK ? kQ : kK);
            if (!(played & bit(other))) {
                if (o.belot_announcer == p && o.belot_card == card)
                    c.pinned[p] |= bit(other);
                else if (rankOf(card) == kK || !cfg.belot_queen_first_forfeits)
                    c.excluded[p] |= bit(other);
            }
        }

        played |= bit(card);
        t.cards[t.n] = card;
        t.players[t.n] = p;
        if (++t.n == kPlayers) t = Trick{};
    }

    if (o.blik != kNoCard && o.taker >= 0) c.pinned[o.taker] |= bit(o.blik);
    for (int p = 0; p < kPlayers; ++p) {
        c.pinned[p] |= o.shown_cards[p];
        // A carré is announced with its rank: the announcer holds all four.
        if (o.announced[p] && o.announcement[p].kind == ToonKind::kCarre)
            for (int s = 0; s < kSuits; ++s) c.pinned[p] |= bit(cardOf(s, o.announcement[p].carre_rank));
    }
    for (int p = 0; p < kPlayers; ++p) c.pinned[p] &= ~played;
    return c;
}

}  // namespace belote
