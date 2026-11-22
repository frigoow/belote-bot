#pragma once

// An independent reference for the cross-check test: the legal-move and
// scoring rules of Tanguy Pemeja's classic-French Belote engine, ported from
// open_spiel/games/belote/belote.cc on the belote-cpp-rework branch of
// https://github.com/tpemeja/open_spiel (LegalCardPlays, TrickWinner,
// ApplyPlayAction's trick points, FinalizeScores).
//
// Copyright 2019 DeepMind Technologies Limited; modifications by the
// tpemeja/open_spiel contributors. Licensed under the Apache License,
// Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
//
// Ported here with only the data structures changed (vectors of card
// indices instead of the OpenSpiel state). Its card index scheme,
// suit * 8 + rank with ranks 7 8 9 10 J Q K A, is the same as belote/card.h.
// It is deliberately NOT written in terms of belote/ code, so that a bug
// shared by both would have to be made twice.

#include <algorithm>
#include <utility>
#include <vector>

namespace tanguy {

constexpr int kNonTrumpStrength[8] = {0, 1, 2, 6, 3, 4, 5, 7};
constexpr int kTrumpStrength[8] = {0, 1, 6, 4, 7, 2, 3, 5};
constexpr int kNonTrumpPoints[8] = {0, 0, 0, 10, 2, 3, 4, 11};
constexpr int kTrumpPoints[8] = {0, 0, 14, 10, 20, 3, 4, 11};

inline int Suit(int card) { return card / 8; }
inline int Rank(int card) { return card % 8; }
inline int Strength(int card, int trump) {
    return Suit(card) == trump ? kTrumpStrength[Rank(card)] : kNonTrumpStrength[Rank(card)];
}
inline int Points(int card, int trump) {
    return Suit(card) == trump ? kTrumpPoints[Rank(card)] : kNonTrumpPoints[Rank(card)];
}

using TrickCards = std::vector<std::pair<int, int>>;  // (player, card)

inline bool IsBetter(int card, int other, int led_suit, int trump) {
    bool card_trump = Suit(card) == trump;
    bool other_trump = Suit(other) == trump;
    if (card_trump != other_trump) return card_trump;
    if (card_trump && other_trump) return Strength(card, trump) > Strength(other, trump);
    bool card_led = Suit(card) == led_suit;
    bool other_led = Suit(other) == led_suit;
    if (card_led != other_led) return card_led;
    if (card_led && other_led) return Strength(card, trump) > Strength(other, trump);
    return false;
}

inline int TrickWinner(const TrickCards& trick, int trump) {
    int led_suit = Suit(trick[0].second);
    int best_player = trick[0].first;
    int best_card = trick[0].second;
    for (size_t i = 1; i < trick.size(); ++i) {
        if (IsBetter(trick[i].second, best_card, led_suit, trump)) {
            best_player = trick[i].first;
            best_card = trick[i].second;
        }
    }
    return best_player;
}

inline std::vector<int> LegalCardPlays(const std::vector<int>& hand, const TrickCards& trick,
                                       int trump, int player) {
    std::vector<int> all = hand;
    std::sort(all.begin(), all.end());
    if (trick.empty()) return all;

    int led_suit = Suit(trick[0].second);
    std::vector<int> same_suit;
    for (int c : hand)
        if (Suit(c) == led_suit) same_suit.push_back(c);
    int current_winner = TrickWinner(trick, trump);
    bool partner_winning = (player + 2) % 4 == current_winner;

    if (!same_suit.empty()) {
        std::sort(same_suit.begin(), same_suit.end());
        if (led_suit != trump) return same_suit;
        // Trump led: must play higher than the best trump so far if possible,
        // even if the partner currently holds the trick.
        int highest = -1;
        for (const auto& pc : trick)
            if (Suit(pc.second) == trump) highest = std::max(highest, Strength(pc.second, trump));
        std::vector<int> higher;
        for (int c : same_suit)
            if (Strength(c, trump) > highest) higher.push_back(c);
        return higher.empty() ? same_suit : higher;
    }

    std::vector<int> trumps;
    for (int c : hand)
        if (Suit(c) == trump) trumps.push_back(c);
    std::sort(trumps.begin(), trumps.end());
    if (!trumps.empty() && led_suit != trump) {
        if (partner_winning) return all;
        std::vector<int> trumps_played;
        for (const auto& pc : trick)
            if (Suit(pc.second) == trump) trumps_played.push_back(pc.second);
        if (trumps_played.empty()) return trumps;
        int highest = -1;
        for (int c : trumps_played) highest = std::max(highest, Strength(c, trump));
        std::vector<int> higher;
        for (int c : trumps)
            if (Strength(c, trump) > highest) higher.push_back(c);
        return higher.empty() ? trumps : higher;
    }
    return all;
}

// Final (declarer, other) points from the 8 tricks, as FinalizeScores
// computes them: trick points with 10 (or 100 on a capot) for the last
// trick, the declarer keeps its points only on strictly more (belote
// included), and belote always stays with its holder's team.
inline std::pair<int, int> FinalScores(const std::vector<TrickCards>& tricks, int trump,
                                       int declarer_team, int belote_team) {
    int team_points[2] = {0, 0};
    std::vector<int> winners;
    for (size_t t = 0; t < tricks.size(); ++t) {
        int winner = TrickWinner(tricks[t], trump);
        winners.push_back(winner);
        int points = 0;
        for (const auto& pc : tricks[t]) points += Points(pc.second, trump);
        if (t + 1 == tricks.size()) {
            bool capot = std::all_of(winners.begin(), winners.end(),
                                     [&](int w) { return w % 2 == winner % 2; });
            points += capot ? 100 : 10;
        }
        team_points[winner % 2] += points;
    }
    int other_team = 1 - declarer_team;
    int declarer_points = team_points[declarer_team];
    int other_points = team_points[other_team];
    int trick_total = declarer_points + other_points;
    int declarer_bonus = belote_team == declarer_team ? 20 : 0;
    int other_bonus = belote_team == other_team ? 20 : 0;
    int final_declarer, final_other;
    if (declarer_points + declarer_bonus > other_points + other_bonus) {
        final_declarer = declarer_points;
        final_other = other_points;
    } else {
        final_declarer = 0;
        final_other = trick_total;
    }
    return {final_declarer + declarer_bonus, final_other + other_bonus};
}

}  // namespace tanguy
