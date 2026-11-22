#include "bg/encoder.h"

#include <algorithm>
#include <cstring>

namespace bg {

namespace {

int cardOrder(int card, std::uint8_t contract) {
    if (contract & kAllTrumps) return trumpOrder(card);
    if (contract & kNoTrumps) return plainOrder(card);
    return suitOf(card) == trumpSuit(contract) ? trumpOrder(card) : plainOrder(card);
}

bool isMasterRemaining(int card, Mask unseen_plus_hand, std::uint8_t contract) {
    const int s = suitOf(card);
    const int ord = cardOrder(card, contract);
    for (Mask m = unseen_plus_hand & suitMask(s); m; m &= m - 1) {
        const int other = lowestCard(m);
        if (other != card && cardOrder(other, contract) > ord) return false;
    }
    return true;
}

Trick currentTrickFrom(const PublicState& ps) {
    Trick t;
    const std::size_t rem = ps.plays.size() % kPlayers;
    const std::size_t base = ps.plays.size() - rem;
    for (std::size_t i = 0; i < rem; ++i) {
        t.cards[i] = ps.plays[base + i].card;
        t.players[i] = ps.plays[base + i].seat;
    }
    t.n = int(rem);
    return t;
}

}  // namespace

void encodeState(const PublicState& ps, float* out, const Facts* precomputed_facts) {
    std::memset(out, 0, std::size_t(kBgStateDim) * sizeof(float));

    const std::uint8_t contract = ps.contract;
    const int me = ps.me;
    const int my_team = teamOf(me);
    const Facts local_facts = precomputed_facts ? Facts{} : inferFacts(ps);
    const Facts& facts = precomputed_facts ? *precomputed_facts : local_facts;

    // 0..32: own hand
    for (Mask m = ps.hand; m; m &= m - 1) {
        out[canonCard(lowestCard(m), contract)] = 1.0f;
    }

    // 32..160: played cards by relative seat (4 x 32)
    Mask all_played = 0;
    std::array<int, kPlayers> belot_cnt{};
    for (const PublicPlay& pl : ps.plays) {
        all_played |= bit(pl.card);
        const int r = relSeat(pl.seat, me);
        out[32 + r * 32 + canonCard(pl.card, contract)] = 1.0f;
        if (pl.belote) ++belot_cnt[std::size_t(r)];
    }

    // Current trick + completed tricks stats
    const Trick cur = currentTrickFrom(ps);
    const std::size_t n_complete = ps.plays.size() / kPlayers;
    int our_pts = 0, opp_pts = 0;
    int our_tricks = 0, opp_tricks = 0;
    for (std::size_t ti = 0; ti < n_complete; ++ti) {
        Trick tr;
        int pts = 0;
        for (int i = 0; i < kPlayers; ++i) {
            const PublicPlay& pl = ps.plays[ti * kPlayers + std::size_t(i)];
            tr.cards[i] = pl.card;
            tr.players[i] = pl.seat;
            pts += cardValue(pl.card, contract);
        }
        tr.n = kPlayers;
        const int winner = tr.players[trickWinnerIndex(contract, tr)];
        if (teamOf(winner) == my_team) {
            our_pts += pts;
            ++our_tricks;
        } else {
            opp_pts += pts;
            ++opp_tricks;
        }
    }

    // 160..192: current trick cards
    int cur_pts = 0;
    for (int i = 0; i < cur.n; ++i) {
        out[160 + canonCard(cur.cards[i], contract)] = 1.0f;
        cur_pts += cardValue(cur.cards[i], contract);
    }

    // 192..224: current trick winning card
    // 224..228: leader rel-seat
    // 228..232: winner rel-seat
    // 232..236: cur.n one-hot
    // 236..241: led canon suit one-hot (4 = leading)
    // 241: cur_pts / 50
    if (cur.n > 0) {
        const int wi = trickWinnerIndex(contract, cur);
        out[192 + canonCard(cur.cards[wi], contract)] = 1.0f;
        out[224 + relSeat(cur.players[0], me)] = 1.0f;
        out[228 + relSeat(cur.players[wi], me)] = 1.0f;
        out[236 + canonSuit(suitOf(cur.cards[0]), contract)] = 1.0f;
    } else {
        out[240] = 1.0f;
    }
    out[232 + std::min(cur.n, 3)] = 1.0f;
    out[241] = float(cur_pts) / 50.0f;

    // 242..338: inferFacts excluded for relSeat 1, 2, 3 (3 x 32)
    // 338..434: inferFacts pinned for relSeat 1, 2, 3 (3 x 32)
    for (int k = 0; k < 3; ++k) {
        const int seat = (me + 1 + k) & 3;
        for (Mask m = facts.excluded[std::size_t(seat)]; m; m &= m - 1) {
            out[242 + k * 32 + canonCard(lowestCard(m), contract)] = 1.0f;
        }
        for (Mask m = facts.pinned[std::size_t(seat)]; m; m &= m - 1) {
            out[338 + k * 32 + canonCard(lowestCard(m), contract)] = 1.0f;
        }
    }

    // 434..437: contract mode (Suit, NT, AT)
    if (contract & kAllTrumps) out[436] = 1.0f;
    else if (contract & kNoTrumps) out[435] = 1.0f;
    else out[434] = 1.0f;

    // 437..440: multiplier (x1, x2, x4)
    if (contract & kReDouble) out[439] = 1.0f;
    else if (contract & kDouble) out[438] = 1.0f;
    else out[437] = 1.0f;

    // 440..444: declarer rel-seat; 444: we_are_declarer
    if (ps.declarer >= 0) {
        out[440 + relSeat(ps.declarer, me)] = 1.0f;
        out[444] = (teamOf(ps.declarer) == my_team) ? 1.0f : 0.0f;
    }

    // 445..449: first rel-seat
    out[445 + relSeat(ps.first, me)] = 1.0f;

    // 449..489: per-rel-seat auction summary (4 x 10)
    std::array<int, kPlayers> non_pass_cnt{};
    for (const auto& [seat, b] : ps.bids) {
        if (b == kPass) continue;
        const int r = relSeat(seat, me);
        const int base = 449 + r * 10;
        ++non_pass_cnt[std::size_t(r)];
        if (b & kClubs) out[base + canonSuit(0, contract)] = 1.0f;
        if (b & kDiamonds) out[base + canonSuit(1, contract)] = 1.0f;
        if (b & kHearts) out[base + canonSuit(2, contract)] = 1.0f;
        if (b & kSpades) out[base + canonSuit(3, contract)] = 1.0f;
        if (b & kNoTrumps) out[base + 4] = 1.0f;
        if (b & kAllTrumps) out[base + 5] = 1.0f;
        if (b & kDouble) out[base + 6] = 1.0f;
        if (b & kReDouble) out[base + 7] = 1.0f;
    }
    for (int r = 0; r < kPlayers; ++r) {
        const int base = 449 + r * 10;
        out[base + 8] = non_pass_cnt[std::size_t(r)] == 0 ? 1.0f : 0.0f;
        out[base + 9] = non_pass_cnt[std::size_t(r)] >= 2 ? 1.0f : 0.0f;
    }

    // 489..513: per-rel-seat declared types (4 x 6)
    // 517..519: raw declared totals per team (leak-free: from public types only)
    int our_decl_raw = 0, opp_decl_raw = 0;
    for (const Declared& d : ps.declared) {
        const int r = relSeat(d.seat, me);
        const int base = 489 + r * 6;
        int val = 0;
        if (d.type == kSeq3) {
            out[base + 0] += 1.0f;
            val = 20;
        } else if (d.type == kSeq4) {
            out[base + 1] += 1.0f;
            val = 50;
        } else if (d.type >= kSeq5 && d.type <= kSeq8) {
            out[base + 2] += 1.0f;
            val = 100;
        } else if (d.type == kFourOfAKind) {
            out[base + 3] += 1.0f;
            val = 100;
        } else if (d.type == kFourNines) {
            out[base + 4] += 1.0f;
            val = 150;
        } else if (d.type == kFourJacks) {
            out[base + 5] += 1.0f;
            val = 200;
        }
        if (teamOf(d.seat) == my_team) our_decl_raw += val;
        else opp_decl_raw += val;
    }

    // 513..517: per-rel-seat belot count / 2.0
    for (int r = 0; r < kPlayers; ++r) {
        out[513 + r] = float(belot_cnt[std::size_t(r)]) * 0.5f;
        if (r == 0 || r == 2) our_pts += 20 * belot_cnt[std::size_t(r)];
        else opp_pts += 20 * belot_cnt[std::size_t(r)];
    }
    out[517] = float(our_decl_raw) / 200.0f;
    out[518] = float(opp_decl_raw) / 200.0f;

    // 519..528: trick number (0..7) scalar + one-hot
    const int trick_idx = std::min(int(n_complete), kTricks - 1);
    out[519] = float(trick_idx + 1) / float(kTricks);
    out[520 + trick_idx] = 1.0f;

    // 528..535: banked pts, tricks, capot flags, hanging
    out[528] = float(our_pts) / 260.0f;
    out[529] = float(opp_pts) / 260.0f;
    out[530] = float(our_tricks) / float(kTricks);
    out[531] = float(opp_tricks) / float(kTricks);
    out[532] = our_tricks > 0 ? 1.0f : 0.0f;
    out[533] = opp_tricks > 0 ? 1.0f : 0.0f;
    out[534] = float(ps.hanging) / 50.0f;

    // 535..539: remaining hand sizes per rel-seat / 8
    for (int r = 0; r < kPlayers; ++r) {
        const int seat = (me + r) & 3;
        out[535 + r] = float(ps.handSize(seat)) / 8.0f;
    }

    // 539..547: unseen remaining per canon suit / 8, own per canon suit / 8
    const Mask unseen = (~(ps.hand | all_played)) & 0xFFFFFFFFu;
    for (int s = 0; s < 4; ++s) {
        const int cs = canonSuit(s, contract);
        out[539 + cs] = float(popcount(unseen & suitMask(s))) / 8.0f;
        out[543 + cs] = float(popcount(ps.hand & suitMask(s))) / 8.0f;
    }

    // 547..579: master-card mask in own hand
    const Mask unplayed = (~all_played) & 0xFFFFFFFFu;
    for (Mask m = ps.hand; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (isMasterRemaining(c, unplayed, contract)) {
            out[547 + canonCard(c, contract)] = 1.0f;
        }
    }

    // 579..611: legal moves mask
    const Mask legal = legalCards(ps.hand, contract, cur);
    for (Mask m = legal; m; m &= m - 1) {
        out[579 + canonCard(lowestCard(m), contract)] = 1.0f;
    }
}

int encodeHistory(const PublicState& ps, float* out) {
    const int n_plies = std::min(int(ps.plays.size()), kBgMaxPlies);
    std::memset(out, 0, std::size_t(kBgMaxPlies * kBgPlyDim) * sizeof(float));
    if (n_plies == 0) return 0;

    const std::uint8_t contract = ps.contract;
    const int trump = trumpSuit(contract);
    const bool is_all = (contract & kAllTrumps) != 0;
    const bool is_suit = trump >= 0;
    const int me = ps.me;

    Trick cur;
    for (int i = 0; i < n_plies; ++i) {
        const PublicPlay& pl = ps.plays[std::size_t(i)];
        float* row = out + std::size_t(i * kBgPlyDim);
        const int pos = cur.n;
        const int trick_idx = i / kPlayers;

        // 0..32: canon card
        row[canonCard(pl.card, contract)] = 1.0f;
        // 32..36: rel seat
        row[32 + relSeat(pl.seat, me)] = 1.0f;
        // 36..40: trick position 0..3
        row[36 + pos] = 1.0f;

        const int led_suit = pos == 0 ? suitOf(pl.card) : suitOf(cur.cards[0]);
        // 40..44: led canon suit
        row[40 + canonSuit(led_suit, contract)] = 1.0f;

        if (pos > 0) {
            const int prev_wi = trickWinnerIndex(contract, cur);
            const int prev_winner = cur.players[prev_wi];
            const int prev_card = cur.cards[prev_wi];
            const bool partner_win = (teamOf(prev_winner) == teamOf(pl.seat));

            Trick next = cur;
            next.cards[pos] = pl.card;
            next.players[pos] = pl.seat;
            next.n = pos + 1;
            const bool beats = (trickWinnerIndex(contract, next) == pos);

            const int cs = suitOf(pl.card);
            const bool followed = (cs == led_suit);
            const bool trumped = is_suit && !followed && cs == trump;
            const bool overtrumped = trumped && beats && suitOf(prev_card) == trump;
            const bool discarded = !followed && !trumped;
            const bool must_try_beat =
                (is_all && followed) ||
                (is_suit && followed && led_suit == trump) ||
                (is_suit && trumped && suitOf(prev_card) == trump && !partner_win);
            const bool could_not_beat = must_try_beat && !beats;

            row[44] = followed ? 1.0f : 0.0f;
            row[45] = trumped ? 1.0f : 0.0f;
            row[46] = overtrumped ? 1.0f : 0.0f;
            row[47] = beats ? 1.0f : 0.0f;
            row[48] = could_not_beat ? 1.0f : 0.0f;
            row[49] = partner_win ? 1.0f : 0.0f;
            row[50] = discarded ? 1.0f : 0.0f;
        } else {
            row[44] = 1.0f;  // leading counts as matching led suit
            row[47] = 1.0f;  // leader is current winner
            row[55] = 1.0f;  // is_lead
        }

        row[51] = pl.belote ? 1.0f : 0.0f;
        row[52] = float(cardValue(pl.card, contract)) / 20.0f;
        row[54] = float(trick_idx + 1) / float(kTricks);

        cur.cards[pos] = pl.card;
        cur.players[pos] = pl.seat;
        ++cur.n;
        if (cur.n == kPlayers) {
            const int wi = trickWinnerIndex(contract, cur);
            const int winner_ply = i - (kPlayers - 1) + wi;
            out[std::size_t(winner_ply * kBgPlyDim + 53)] = 1.0f;
            cur = Trick{};
        }
    }
    return n_plies;
}

void encodeAction(const PublicState& ps, int card, float* out) {
    std::memset(out, 0, std::size_t(kBgActionDim) * sizeof(float));

    const std::uint8_t contract = ps.contract;
    const int trump = trumpSuit(contract);
    const bool is_all = (contract & kAllTrumps) != 0;
    const Trick cur = currentTrickFrom(ps);
    const Mask legal = legalCards(ps.hand, contract, cur);

    Mask all_played = 0;
    for (const PublicPlay& pl : ps.plays) all_played |= bit(pl.card);
    const Mask unplayed = (~all_played) & 0xFFFFFFFFu;

    // 0..32: canon card one-hot
    out[canonCard(card, contract)] = 1.0f;
    // 32..36: canon suit one-hot
    out[32 + canonSuit(suitOf(card), contract)] = 1.0f;
    // 36: card points / 20
    out[36] = float(cardValue(card, contract)) / 20.0f;
    // 37: is_trump_card
    out[37] = (is_all || (trump >= 0 && suitOf(card) == trump)) ? 1.0f : 0.0f;

    // 38: beats_current_trick; 41: trick_pts_with_c / 60
    int tpts = cardValue(card, contract);
    for (int i = 0; i < cur.n; ++i) tpts += cardValue(cur.cards[i], contract);
    if (cur.n > 0) {
        Trick next = cur;
        next.cards[cur.n] = card;
        next.players[cur.n] = ps.me;
        next.n = cur.n + 1;
        out[38] = (trickWinnerIndex(contract, next) == cur.n) ? 1.0f : 0.0f;
    } else {
        out[38] = 1.0f;
    }
    out[39] = isMasterRemaining(card, unplayed, contract) ? 1.0f : 0.0f;
    out[40] = (popcount(legal) > 1 && beloteAllowed(ps.hand, contract, cur, card)) ? 1.0f : 0.0f;
    out[41] = float(tpts) / 60.0f;
    out[42] = (popcount(ps.hand & suitMask(suitOf(card))) == 1) ? 1.0f : 0.0f;
    out[43] = float(cardOrder(card, contract) - 1) / 7.0f;
}

int encodeLegalActions(const PublicState& ps, float* actions_out, std::int32_t* legal_cards_out) {
    const Trick cur = currentTrickFrom(ps);
    const Mask legal = legalCards(ps.hand, ps.contract, cur);
    std::memset(actions_out, 0, std::size_t(kBgMaxLegal * kBgActionDim) * sizeof(float));
    for (int i = 0; i < kBgMaxLegal; ++i) legal_cards_out[i] = -1;

    int n = 0;
    for (Mask m = legal; m && n < kBgMaxLegal; m &= m - 1) {
        const int c = lowestCard(m);
        legal_cards_out[n] = c;
        encodeAction(ps, c, actions_out + std::size_t(n * kBgActionDim));
        ++n;
    }
    return n;
}

}  // namespace bg
