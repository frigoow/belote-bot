"""Pure-Python reference observation & action encoder (mirrors cpp/belote/encoder.cpp).

Golden-tested bit-for-bit against belote_cpp.encode_state / encode_history /
encode_action so training, evaluation, and C++/ONNX inference never diverge.
"""

from __future__ import annotations

import numpy as np

import belote_cpp as bc

ENCODER_VERSION = 1
STATE_DIM = 697
PLY_DIM = 54
MAX_PLIES = 32
ACTION_DIM = 41

K_RANK = 6
Q_RANK = 5
DEAL_POINTS = 162
LAST_TRICK_BONUS = 10
BELOT_POINTS = 20

_KIND_TO_INT = {
    bc.ToonKind.DERDE: 0,
    bc.ToonKind.VIERDE: 1,
    bc.ToonKind.VIJFDE: 2,
    bc.ToonKind.CARRE: 3,
}


def canon_suit(suit: int, trump: int) -> int:
    return (suit - trump) & 3


def raw_suit(csuit: int, trump: int) -> int:
    return (csuit + trump) & 3


def canon_card(card: int, trump: int) -> int:
    return canon_suit(card >> 3, trump) * 8 + (card & 7)


def raw_card(ccard: int, trump: int) -> int:
    return raw_suit(ccard >> 3, trump) * 8 + (ccard & 7)


def rel_seat(player: int, observer: int) -> int:
    return (player - observer) & 3


def _write_mask_canon(mask: int, trump: int, out: np.ndarray, offset: int) -> None:
    for c in bc.mask_cards(mask):
        out[offset + canon_card(c, trump)] = 1.0


def _replay_public(obs: bc.Observation):
    trump = obs.trump
    k_card = bc.card_of(trump, K_RANK)
    q_card = bc.card_of(trump, Q_RANK)

    played_by = [0, 0, 0, 0]
    played_all = 0
    card_points = [0, 0]
    tricks_won = [0, 0]
    tricks_played = 0
    cur_trick = bc.Trick()
    cur_player = (obs.dealer + 1) & 3
    queen_played_no_belot = False

    for pl in obs.plays:
        p = pl.player
        card = pl.card
        if card == q_card and not (played_all & (1 << k_card)) and obs.belot_announcer < 0:
            queen_played_no_belot = True
        played_by[p] |= 1 << card
        played_all |= 1 << card
        cards = list(cur_trick.cards)
        players = list(cur_trick.players)
        cards[cur_trick.n] = card
        players[cur_trick.n] = p
        cur_trick.cards = cards
        cur_trick.players = players
        cur_trick.n += 1
        cur_player = (p + 1) & 3
        if cur_trick.n == 4:
            w = bc.trick_winner_index(cur_trick, trump)
            winner = cur_trick.players[w]
            pts = sum(bc.card_points(c, trump) for c in cur_trick.cards)
            if tricks_played == 7:
                pts += LAST_TRICK_BONUS
            card_points[winner & 1] += pts
            tricks_won[winner & 1] += 1
            tricks_played += 1
            cur_trick = bc.Trick()
            cur_player = winner

    return (
        played_by,
        played_all,
        card_points,
        tricks_won,
        tricks_played,
        cur_trick,
        cur_player,
        queen_played_no_belot,
    )


def _toon_key(toon: bc.Toon, trump: int) -> int:
    trump_strength = (0, 1, 6, 4, 7, 2, 3, 5)
    if toon.carre:
        return (1 << 20) | (trump_strength[toon.rank] << 8)
    return (toon.length << 8) | (toon.rank << 4) | (1 if toon.suit == trump else 0)


def _toon_winning_team(declared: list[list[bc.Toon]], trump: int, dealer: int) -> int:
    best_key = -1
    best_order = 4
    best_player = -1
    for p in range(4):
        order = (p - dealer - 1) & 3
        for t in declared[p]:
            key = _toon_key(t, trump)
            if key > best_key or (key == best_key and order < best_order):
                best_key = key
                best_order = order
                best_player = p
    return -1 if best_player < 0 else (best_player & 1)


def encode_state_py(obs: bc.Observation) -> np.ndarray:
    out = np.zeros(STATE_DIM, dtype=np.float32)
    (
        played_by,
        played_all,
        card_points,
        tricks_won,
        tricks_played,
        cur_trick,
        cur_player,
        queen_played_no_belot,
    ) = _replay_public(obs)
    c = bc.infer_constraints(obs)

    me = obs.observer
    trump = obs.trump
    us = me & 1
    them = 1 - us
    legal = (
        bc.legal_mask(obs.hand, cur_trick, trump, me, obs.config)
        if (cur_player == me and tricks_played < 8)
        else 0
    )

    idx = 0
    # Block A: Cards & Current Trick (300)
    _write_mask_canon(obs.hand, trump, out, idx)
    idx += 32
    _write_mask_canon(legal, trump, out, idx)
    idx += 32
    for r in range(4):
        p = (me + r) & 3
        _write_mask_canon(played_by[p], trump, out, idx + r * 32)
    idx += 128
    for i in range(cur_trick.n):
        r = rel_seat(cur_trick.players[i], me)
        if r >= 1:
            out[idx + (r - 1) * 32 + canon_card(cur_trick.cards[i], trump)] = 1.0
    idx += 96
    if cur_trick.n > 0:
        out[idx + canon_suit(cur_trick.cards[0] >> 3, trump)] = 1.0
    idx += 4
    out[idx + cur_trick.n] = 1.0
    idx += 4
    if cur_trick.n > 0:
        w = bc.trick_winner_index(cur_trick, trump)
        r = rel_seat(cur_trick.players[w], me)
        if r >= 1:
            out[idx + (r - 1)] = 1.0
    idx += 3
    cur_pts = sum(bc.card_points(cur_trick.cards[i], trump) for i in range(cur_trick.n))
    out[idx] = np.float32(cur_pts) / np.float32(40.0)
    idx += 1

    # Block B: Inferred Hard Constraints (144)
    for r in range(4):
        p = (me + r) & 3
        for cs in range(4):
            sm = 0xFF << (raw_suit(cs, trump) * 8)
            if (c.excluded[p] & sm) == sm:
                out[idx + r * 4 + cs] = 1.0
    idx += 16
    for r in range(4):
        p = (me + r) & 3
        for rk in range(8):
            if c.excluded[p] & (1 << bc.card_of(trump, rk)):
                out[idx + r * 8 + rk] = 1.0
    idx += 32
    for r in range(1, 4):
        p = (me + r) & 3
        _write_mask_canon(c.pinned[p], trump, out, idx + (r - 1) * 32)
    idx += 96

    # Block C: Contract & Auction Context (51)
    if obs.taker >= 0:
        out[idx + rel_seat(obs.taker, me)] = 1.0
    idx += 4
    out[idx] = 1.0 if (obs.taker >= 0 and (obs.taker & 1) == us) else 0.0
    idx += 1
    if obs.dealer >= 0:
        out[idx + rel_seat(obs.dealer, me)] = 1.0
    idx += 4
    if obs.blik != bc.NO_CARD:
        out[idx + canon_card(obs.blik, trump)] = 1.0
    idx += 32
    out[idx] = 1.0 if (obs.blik != bc.NO_CARD and not (played_all & (1 << obs.blik))) else 0.0
    idx += 1
    if 0 <= obs.bids_made < 8:
        out[idx + obs.bids_made] = 1.0
    idx += 8
    out[idx] = 1.0 if obs.bids_made >= 4 else 0.0
    idx += 1

    # Block D: Declarations — Toon & Belot (190)
    for r in range(4):
        p = (me + r) & 3
        if not obs.announced[p]:
            out[idx + r * 5] = 1.0
        else:
            out[idx + r * 5 + 1 + _KIND_TO_INT[obs.announcement[p].kind]] = 1.0
    idx += 20
    for r in range(4):
        p = (me + r) & 3
        if (
            obs.announced[p]
            and obs.announcement[p].kind == bc.ToonKind.CARRE
            and obs.announcement[p].carre_rank >= 0
        ):
            out[idx + r * 8 + obs.announcement[p].carre_rank] = 1.0
    idx += 32
    for r in range(4):
        p = (me + r) & 3
        _write_mask_canon(obs.shown_cards[p], trump, out, idx + r * 32)
    idx += 128

    shown_team_pts = [0, 0]
    any_shown = False
    for p in range(4):
        if obs.shown[p]:
            shown_team_pts[p & 1] += obs.shown_points[p]
            any_shown = True
    winning_toon_team = -1
    if any_shown:
        if not obs.config.toon_losers_show:
            if shown_team_pts[0] > 0:
                winning_toon_team = 0
            elif shown_team_pts[1] > 0:
                winning_toon_team = 1
        else:
            shown_toons = [
                bc.find_toons(obs.shown_cards[p], obs.config) if obs.shown[p] else []
                for p in range(4)
            ]
            winning_toon_team = _toon_winning_team(shown_toons, trump, obs.dealer)

    out[idx] = 1.0 if winning_toon_team == us else 0.0
    out[idx + 1] = 1.0 if winning_toon_team == them else 0.0
    out[idx + 2] = np.float32(shown_team_pts[us] if winning_toon_team == us else 0) / np.float32(200.0)
    out[idx + 3] = np.float32(shown_team_pts[them] if winning_toon_team == them else 0) / np.float32(200.0)
    idx += 4

    kq = (1 << bc.card_of(trump, K_RANK)) | (1 << bc.card_of(trump, Q_RANK))
    self_intact_kq = (obs.hand & kq) == kq
    out[idx] = 1.0 if self_intact_kq else 0.0
    out[idx + 1] = 1.0 if (obs.belot_announcer == me and (obs.hand & kq) != 0) else 0.0
    out[idx + 2] = 1.0 if (obs.belot_announcer >= 0 and (obs.belot_announcer & 1) == us) else 0.0
    out[idx + 3] = 1.0 if (obs.belot_announcer >= 0 and (obs.belot_announcer & 1) == them) else 0.0
    out[idx + 4] = 1.0 if (obs.config.belot_queen_first_forfeits and queen_played_no_belot) else 0.0
    out[idx + 5] = 1.0 if obs.config.belot_queen_first_forfeits else 0.0
    idx += 6

    # Block E: Contract-Cliff & Trick Progress Scalars (12)
    known_toon = [0, 0]
    if winning_toon_team >= 0:
        known_toon[winning_toon_team] = shown_team_pts[winning_toon_team]
    known_belot = [0, 0]
    if obs.belot_announcer >= 0:
        known_belot[obs.belot_announcer & 1] = BELOT_POINTS
    elif self_intact_kq:
        known_belot[us] = BELOT_POINTS

    takers_team = (obs.taker & 1) if obs.taker >= 0 else 0
    def_team = 1 - takers_team
    takers_known = card_points[takers_team] + known_toon[takers_team] + known_belot[takers_team]
    def_known = card_points[def_team] + known_toon[def_team] + known_belot[def_team]
    deal_total_known = DEAL_POINTS + sum(known_toon) + sum(known_belot)
    make_target = (
        (deal_total_known + 1) // 2
        if obs.config.tie_rule == bc.TieRule.TAKERS_MAKE
        else (deal_total_known // 2 + 1)
    )

    out[idx] = np.float32(card_points[us]) / np.float32(162.0)
    out[idx + 1] = np.float32(card_points[them]) / np.float32(162.0)
    out[idx + 2] = np.float32(tricks_won[us]) / np.float32(8.0)
    out[idx + 3] = np.float32(tricks_won[them]) / np.float32(8.0)
    out[idx + 4] = np.float32(8 - tricks_played) / np.float32(8.0)
    out[idx + 5] = np.float32(DEAL_POINTS - card_points[0] - card_points[1]) / np.float32(162.0)
    out[idx + 6] = 1.0 if tricks_won[them] == 0 else 0.0
    out[idx + 7] = 1.0 if tricks_won[us] == 0 else 0.0
    out[idx + 8] = np.float32(takers_known) / np.float32(162.0)
    out[idx + 9] = np.float32(def_known) / np.float32(162.0)
    out[idx + 10] = np.float32(max(0, make_target - takers_known)) / np.float32(162.0)
    out[idx + 11] = 1.0 if takers_known >= make_target else 0.0
    idx += 12

    assert idx == STATE_DIM
    return out


def encode_history_py(obs: bc.Observation) -> tuple[np.ndarray, int]:
    out = np.zeros((MAX_PLIES, PLY_DIM), dtype=np.float32)
    me = obs.observer
    trump = obs.trump
    n_plies = min(len(obs.plays), MAX_PLIES)
    cur = bc.Trick()

    for t in range(n_plies):
        p = obs.plays[t].player
        card = obs.plays[t].card
        prev_n = cur.n
        led = (cur.cards[0] >> 3) if prev_n > 0 else (card >> 3)
        prev_win_card = bc.NO_CARD
        prev_win_player = -1
        prev_win_is_trump = False
        if prev_n > 0:
            w = bc.trick_winner_index(cur, trump)
            prev_win_card = cur.cards[w]
            prev_win_player = cur.players[w]
            prev_win_is_trump = (prev_win_card >> 3) == trump

        cards = list(cur.cards)
        players = list(cur.players)
        cards[cur.n] = card
        players[cur.n] = p
        cur.cards = cards
        cur.players = players
        cur.n += 1
        won_now = bc.trick_winner_index(cur, trump) == prev_n

        row = out[t]
        idx = 0
        row[idx + rel_seat(p, me)] = 1.0
        idx += 4
        row[idx + canon_card(card, trump)] = 1.0
        idx += 32
        row[idx + prev_n] = 1.0
        idx += 4
        row[idx + canon_suit(led, trump)] = 1.0
        idx += 4
        row[idx] = np.float32(t // 4) / np.float32(7.0)
        idx += 1
        row[idx] = np.float32(bc.card_points(card, trump)) / np.float32(20.0)
        idx += 1
        row[idx] = 1.0 if won_now else 0.0
        idx += 1
        row[idx] = 1.0 if (prev_n > 0 and (card >> 3) == led) else 0.0
        idx += 1
        row[idx] = 1.0 if (prev_n > 0 and (card >> 3) != led and (card >> 3) == trump) else 0.0
        idx += 1
        row[idx] = (
            1.0
            if (
                prev_n > 0
                and (card >> 3) == trump
                and prev_win_is_trump
                and bc.card_strength(card, trump) > bc.card_strength(prev_win_card, trump)
            )
            else 0.0
        )
        idx += 1
        row[idx] = (
            1.0
            if (
                prev_n > 0
                and (card >> 3) == trump
                and prev_win_is_trump
                and bc.card_strength(card, trump) < bc.card_strength(prev_win_card, trump)
            )
            else 0.0
        )
        idx += 1
        row[idx] = 1.0 if (prev_n > 0 and prev_win_player == ((p + 2) & 3)) else 0.0
        idx += 1
        row[idx] = 1.0 if (prev_n > 0 and (card >> 3) != led and (card >> 3) != trump) else 0.0
        idx += 1
        row[idx] = 1.0 if (obs.belot_announcer == p and obs.belot_card == card) else 0.0
        idx += 1
        assert idx == PLY_DIM

        if cur.n == 4:
            cur = bc.Trick()

    return out, n_plies


def encode_action_py(obs: bc.Observation, card: int) -> np.ndarray:
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    _, _, _, _, tricks_played, cur_trick, cur_player, _ = _replay_public(obs)
    me = obs.observer
    trump = obs.trump
    legal = (
        bc.legal_mask(obs.hand, cur_trick, trump, me, obs.config)
        if (cur_player == me and tricks_played < 8)
        else 0
    )

    u = bc.Trick()
    cards = list(cur_trick.cards)
    players = list(cur_trick.players)
    prev_n = cur_trick.n
    cards[prev_n] = card
    players[prev_n] = me
    u.cards = cards
    u.players = players
    u.n = prev_n + 1
    wins_now = bc.trick_winner_index(u, trump) == prev_n
    trick_pts = sum(bc.card_points(u.cards[i], trump) for i in range(u.n))
    if tricks_played == 7 and prev_n == 3:
        trick_pts += LAST_TRICK_BONUS

    kq = (1 << bc.card_of(trump, K_RANK)) | (1 << bc.card_of(trump, Q_RANK))
    self_intact_kq = (obs.hand & kq) == kq
    is_trump = (card >> 3) == trump
    would_belot = (
        self_intact_kq
        and is_trump
        and ((card & 7) == K_RANK or (not obs.config.belot_queen_first_forfeits and (card & 7) == Q_RANK))
    )
    would_forfeit = (
        obs.config.belot_queen_first_forfeits and self_intact_kq and is_trump and (card & 7) == Q_RANK
    )

    idx = 0
    out[canon_card(card, trump)] = 1.0
    idx += 32
    out[idx] = 1.0 if is_trump else 0.0
    out[idx + 1] = np.float32(bc.card_points(card, trump)) / np.float32(20.0)
    out[idx + 2] = np.float32(bc.card_strength(card, trump)) / np.float32(7.0)
    out[idx + 3] = 1.0 if wins_now else 0.0
    out[idx + 4] = 1.0 if prev_n == 3 else 0.0
    out[idx + 5] = np.float32(trick_pts) / np.float32(45.0)
    out[idx + 6] = 1.0 if would_belot else 0.0
    out[idx + 7] = 1.0 if would_forfeit else 0.0
    out[idx + 8] = np.float32(len(bc.mask_cards(legal))) / np.float32(8.0)
    idx += 9
    assert idx == ACTION_DIM
    return out
