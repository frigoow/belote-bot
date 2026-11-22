#include "belote/encoder.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace belote {

namespace {

struct PublicReplay {
    std::array<Mask, kPlayers> played_by{};
    Mask played_all = 0;
    std::array<int, 2> card_points{0, 0};
    std::array<int, 2> tricks_won{0, 0};
    int tricks_played = 0;
    Trick cur_trick{};
    int cur_player = -1;
    bool queen_played_no_belot = false;
};

PublicReplay replayPublic(const Observation& o) {
    PublicReplay r;
    r.cur_player = nextPlayer(o.dealer);
    const int trump = o.trump;
    const int k_card = cardOf(trump, kK);
    const int q_card = cardOf(trump, kQ);

    for (const Play& pl : o.plays) {
        const int p = pl.player;
        const int card = pl.card;
        if (card == q_card && !(r.played_all & bit(k_card)) && o.belot_announcer < 0)
            r.queen_played_no_belot = true;
        r.played_by[p] |= bit(card);
        r.played_all |= bit(card);
        r.cur_trick.cards[r.cur_trick.n] = card;
        r.cur_trick.players[r.cur_trick.n] = p;
        ++r.cur_trick.n;
        r.cur_player = nextPlayer(p);
        if (r.cur_trick.n == kPlayers) {
            const int winner = r.cur_trick.players[trickWinnerIndex(r.cur_trick, trump)];
            int pts = 0;
            for (int c : r.cur_trick.cards) pts += cardPoints(c, trump);
            if (r.tricks_played == kTricks - 1) pts += kLastTrickBonus;
            r.card_points[teamOf(winner)] += pts;
            ++r.tricks_won[teamOf(winner)];
            ++r.tricks_played;
            r.cur_trick = Trick{};
            r.cur_player = winner;
        }
    }
    return r;
}

void writeMaskCanon(Mask m, int trump, float* dst) {
    for (; m; m &= m - 1) dst[canonCard(lowestCard(m), trump)] = 1.0f;
}

}  // namespace

void encodeState(const Observation& o, float* out) {
    std::memset(out, 0, sizeof(float) * kStateDim);
    const PublicReplay rep = replayPublic(o);
    const Constraints c = inferConstraints(o);
    const int obs = o.observer;
    const int trump = o.trump;
    const int us = teamOf(obs);
    const int them = 1 - us;

    const Mask legal = (rep.cur_player == obs && rep.tricks_played < kTricks)
                           ? legalMask(o.hand, rep.cur_trick, trump, obs, o.cfg)
                           : Mask(0);

    int idx = 0;

    // Block A: Cards & Current Trick (300)
    writeMaskCanon(o.hand, trump, out + idx);
    idx += 32;
    writeMaskCanon(legal, trump, out + idx);
    idx += 32;
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        writeMaskCanon(rep.played_by[p], trump, out + idx + r * 32);
    }
    idx += 128;
    for (int i = 0; i < rep.cur_trick.n; ++i) {
        const int r = relSeat(rep.cur_trick.players[i], obs);
        if (r >= 1) out[idx + (r - 1) * 32 + canonCard(rep.cur_trick.cards[i], trump)] = 1.0f;
    }
    idx += 96;
    if (rep.cur_trick.n > 0) out[idx + canonSuit(suitOf(rep.cur_trick.cards[0]), trump)] = 1.0f;
    idx += 4;
    out[idx + rep.cur_trick.n] = 1.0f;
    idx += 4;
    if (rep.cur_trick.n > 0) {
        const int w = trickWinnerIndex(rep.cur_trick, trump);
        const int r = relSeat(rep.cur_trick.players[w], obs);
        if (r >= 1) out[idx + (r - 1)] = 1.0f;
    }
    idx += 3;
    int cur_pts = 0;
    for (int i = 0; i < rep.cur_trick.n; ++i) cur_pts += cardPoints(rep.cur_trick.cards[i], trump);
    out[idx++] = float(cur_pts) / 40.0f;

    // Block B: Inferred Hard Constraints (144)
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        for (int cs = 0; cs < kSuits; ++cs) {
            const Mask sm = suitMask(rawSuit(cs, trump));
            if ((c.excluded[p] & sm) == sm) out[idx + r * 4 + cs] = 1.0f;
        }
    }
    idx += 16;
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        for (int rk = 0; rk < kRanks; ++rk) {
            if (c.excluded[p] & bit(cardOf(trump, rk))) out[idx + r * 8 + rk] = 1.0f;
        }
    }
    idx += 32;
    for (int r = 1; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        writeMaskCanon(c.pinned[p], trump, out + idx + (r - 1) * 32);
    }
    idx += 96;

    // Block C: Contract & Auction Context (51)
    if (o.taker >= 0) out[idx + relSeat(o.taker, obs)] = 1.0f;
    idx += 4;
    out[idx++] = (o.taker >= 0 && teamOf(o.taker) == us) ? 1.0f : 0.0f;
    if (o.dealer >= 0) out[idx + relSeat(o.dealer, obs)] = 1.0f;
    idx += 4;
    if (o.blik != kNoCard) out[idx + canonCard(o.blik, trump)] = 1.0f;
    idx += 32;
    out[idx++] = (o.blik != kNoCard && !(rep.played_all & bit(o.blik))) ? 1.0f : 0.0f;
    if (o.bids_made >= 0 && o.bids_made < 8) out[idx + o.bids_made] = 1.0f;
    idx += 8;
    out[idx++] = (o.bids_made >= 4) ? 1.0f : 0.0f;

    // Block D: Declarations — Toon & Belot (190)
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        if (!o.announced[p]) out[idx + r * 5] = 1.0f;
        else out[idx + r * 5 + 1 + int(o.announcement[p].kind)] = 1.0f;
    }
    idx += 20;
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        if (o.announced[p] && o.announcement[p].kind == ToonKind::kCarre && o.announcement[p].carre_rank >= 0)
            out[idx + r * 8 + o.announcement[p].carre_rank] = 1.0f;
    }
    idx += 32;
    for (int r = 0; r < kPlayers; ++r) {
        const int p = (obs + r) & 3;
        writeMaskCanon(o.shown_cards[p], trump, out + idx + r * 32);
    }
    idx += 128;

    std::array<int, 2> shown_team_pts{0, 0};
    bool any_shown = false;
    for (int p = 0; p < kPlayers; ++p) {
        if (o.shown[p]) {
            shown_team_pts[teamOf(p)] += o.shown_points[p];
            any_shown = true;
        }
    }
    int winning_toon_team = -1;
    if (any_shown) {
        if (!o.cfg.toon_losers_show) {
            if (shown_team_pts[0] > 0) winning_toon_team = 0;
            else if (shown_team_pts[1] > 0) winning_toon_team = 1;
        } else {
            std::vector<Toon> shown_toons[kPlayers];
            for (int p = 0; p < kPlayers; ++p)
                if (o.shown[p]) shown_toons[p] = findToons(o.shown_cards[p], o.cfg);
            winning_toon_team = toonWinningTeam(shown_toons, trump, o.dealer);
        }
    }
    out[idx++] = (winning_toon_team == us) ? 1.0f : 0.0f;
    out[idx++] = (winning_toon_team == them) ? 1.0f : 0.0f;
    out[idx++] = float(winning_toon_team == us ? shown_team_pts[us] : 0) / 200.0f;
    out[idx++] = float(winning_toon_team == them ? shown_team_pts[them] : 0) / 200.0f;

    const Mask kq = bit(cardOf(trump, kK)) | bit(cardOf(trump, kQ));
    const bool self_intact_kq = (o.hand & kq) == kq;
    out[idx++] = self_intact_kq ? 1.0f : 0.0f;
    out[idx++] = (o.belot_announcer == obs && (o.hand & kq) != 0) ? 1.0f : 0.0f;
    out[idx++] = (o.belot_announcer >= 0 && teamOf(o.belot_announcer) == us) ? 1.0f : 0.0f;
    out[idx++] = (o.belot_announcer >= 0 && teamOf(o.belot_announcer) == them) ? 1.0f : 0.0f;
    out[idx++] = (o.cfg.belot_queen_first_forfeits && rep.queen_played_no_belot) ? 1.0f : 0.0f;
    out[idx++] = o.cfg.belot_queen_first_forfeits ? 1.0f : 0.0f;

    // Block E: Contract-Cliff & Trick Progress Scalars (12)
    std::array<int, 2> known_toon{0, 0};
    if (winning_toon_team >= 0) known_toon[winning_toon_team] = shown_team_pts[winning_toon_team];
    std::array<int, 2> known_belot{0, 0};
    if (o.belot_announcer >= 0) known_belot[teamOf(o.belot_announcer)] = kBelotPoints;
    else if (self_intact_kq) known_belot[us] = kBelotPoints;

    const int takers_team = o.taker >= 0 ? teamOf(o.taker) : 0;
    const int def_team = 1 - takers_team;
    const int takers_known = rep.card_points[takers_team] + known_toon[takers_team] + known_belot[takers_team];
    const int def_known = rep.card_points[def_team] + known_toon[def_team] + known_belot[def_team];
    const int deal_total_known =
        kDealPoints + known_toon[0] + known_toon[1] + known_belot[0] + known_belot[1];
    const int make_target =
        (o.cfg.tie_rule == TieRule::kTakersMake) ? (deal_total_known + 1) / 2 : (deal_total_known / 2 + 1);

    out[idx++] = float(rep.card_points[us]) / 162.0f;
    out[idx++] = float(rep.card_points[them]) / 162.0f;
    out[idx++] = float(rep.tricks_won[us]) / 8.0f;
    out[idx++] = float(rep.tricks_won[them]) / 8.0f;
    out[idx++] = float(kTricks - rep.tricks_played) / 8.0f;
    out[idx++] = float(kDealPoints - rep.card_points[0] - rep.card_points[1]) / 162.0f;
    out[idx++] = (rep.tricks_won[them] == 0) ? 1.0f : 0.0f;
    out[idx++] = (rep.tricks_won[us] == 0) ? 1.0f : 0.0f;
    out[idx++] = float(takers_known) / 162.0f;
    out[idx++] = float(def_known) / 162.0f;
    out[idx++] = float(std::max(0, make_target - takers_known)) / 162.0f;
    out[idx++] = (takers_known >= make_target) ? 1.0f : 0.0f;

    assert(idx == kStateDim);
}

int encodeHistory(const Observation& o, float* out) {
    std::memset(out, 0, sizeof(float) * kMaxPlies * kPlyDim);
    const int obs = o.observer;
    const int trump = o.trump;
    const int n_plies = std::min(int(o.plays.size()), kMaxPlies);
    Trick cur;

    for (int t = 0; t < n_plies; ++t) {
        const int p = o.plays[std::size_t(t)].player;
        const int card = o.plays[std::size_t(t)].card;
        float* row = out + t * kPlyDim;

        const int prev_n = cur.n;
        const int led = prev_n > 0 ? suitOf(cur.cards[0]) : suitOf(card);
        int prev_win_card = kNoCard;
        int prev_win_player = -1;
        bool prev_win_is_trump = false;
        if (prev_n > 0) {
            const int w = trickWinnerIndex(cur, trump);
            prev_win_card = cur.cards[w];
            prev_win_player = cur.players[w];
            prev_win_is_trump = suitOf(prev_win_card) == trump;
        }

        cur.cards[cur.n] = card;
        cur.players[cur.n] = p;
        ++cur.n;
        const bool won_now = trickWinnerIndex(cur, trump) == prev_n;

        int idx = 0;
        row[idx + relSeat(p, obs)] = 1.0f;
        idx += 4;
        row[idx + canonCard(card, trump)] = 1.0f;
        idx += 32;
        row[idx + prev_n] = 1.0f;
        idx += 4;
        row[idx + canonSuit(led, trump)] = 1.0f;
        idx += 4;
        row[idx++] = float(t / kPlayers) / 7.0f;
        row[idx++] = float(cardPoints(card, trump)) / 20.0f;
        row[idx++] = won_now ? 1.0f : 0.0f;
        row[idx++] = (prev_n > 0 && suitOf(card) == led) ? 1.0f : 0.0f;
        row[idx++] = (prev_n > 0 && suitOf(card) != led && suitOf(card) == trump) ? 1.0f : 0.0f;
        row[idx++] = (prev_n > 0 && suitOf(card) == trump && prev_win_is_trump &&
                      cardStrength(card, trump) > cardStrength(prev_win_card, trump))
                         ? 1.0f
                         : 0.0f;
        row[idx++] = (prev_n > 0 && suitOf(card) == trump && prev_win_is_trump &&
                      cardStrength(card, trump) < cardStrength(prev_win_card, trump))
                         ? 1.0f
                         : 0.0f;
        row[idx++] = (prev_n > 0 && prev_win_player == partnerOf(p)) ? 1.0f : 0.0f;
        row[idx++] = (prev_n > 0 && suitOf(card) != led && suitOf(card) != trump) ? 1.0f : 0.0f;
        row[idx++] = (o.belot_announcer == p && o.belot_card == card) ? 1.0f : 0.0f;
        assert(idx == kPlyDim);

        if (cur.n == kPlayers) cur = Trick{};
    }
    return n_plies;
}

void encodeAction(const Observation& o, int card, float* out) {
    std::memset(out, 0, sizeof(float) * kActionDim);
    const PublicReplay rep = replayPublic(o);
    const int obs = o.observer;
    const int trump = o.trump;
    const Mask legal = (rep.cur_player == obs && rep.tricks_played < kTricks)
                           ? legalMask(o.hand, rep.cur_trick, trump, obs, o.cfg)
                           : Mask(0);

    Trick u = rep.cur_trick;
    const int prev_n = u.n;
    u.cards[u.n] = card;
    u.players[u.n] = obs;
    ++u.n;
    const bool wins_now = trickWinnerIndex(u, trump) == prev_n;
    int trick_pts = 0;
    for (int i = 0; i < u.n; ++i) trick_pts += cardPoints(u.cards[i], trump);
    if (rep.tricks_played == kTricks - 1 && prev_n == kPlayers - 1) trick_pts += kLastTrickBonus;

    const Mask kq = bit(cardOf(trump, kK)) | bit(cardOf(trump, kQ));
    const bool self_intact_kq = (o.hand & kq) == kq;
    const bool is_trump = suitOf(card) == trump;
    const bool would_belot =
        self_intact_kq && is_trump && (rankOf(card) == kK || (!o.cfg.belot_queen_first_forfeits && rankOf(card) == kQ));
    const bool would_forfeit =
        o.cfg.belot_queen_first_forfeits && self_intact_kq && is_trump && rankOf(card) == kQ;

    int idx = 0;
    out[idx + canonCard(card, trump)] = 1.0f;
    idx += 32;
    out[idx++] = is_trump ? 1.0f : 0.0f;
    out[idx++] = float(cardPoints(card, trump)) / 20.0f;
    out[idx++] = float(cardStrength(card, trump)) / 7.0f;
    out[idx++] = wins_now ? 1.0f : 0.0f;
    out[idx++] = (prev_n == kPlayers - 1) ? 1.0f : 0.0f;
    out[idx++] = float(trick_pts) / 45.0f;
    out[idx++] = would_belot ? 1.0f : 0.0f;
    out[idx++] = would_forfeit ? 1.0f : 0.0f;
    out[idx++] = float(popcount(legal)) / 8.0f;
    assert(idx == kActionDim);
}

EncodedObs encodeObservation(const Observation& obs) {
    EncodedObs e;
    encodeState(obs, e.state.data());
    e.n_plies = encodeHistory(obs, e.history.data());
    return e;
}

}  // namespace belote
