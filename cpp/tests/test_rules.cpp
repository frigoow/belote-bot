// Rule-by-rule tests: one case per bullet of docs/RULES_BELGIAN.md §3–§7.

#include "belote/deal.h"
#include "belote/match.h"
#include "belote/toon.h"
#include "check.h"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

using namespace belote;

namespace {

const Config kBelgian = Config::belgian();     // belote-rebelote.be, the default
const Config kWikibooks = Config::wikibooks();

Trick makeTrick(std::initializer_list<std::pair<int, const char*>> plays) {
    Trick t;
    for (const auto& [player, name] : plays) {
        t.cards[t.n] = parseCard(name);
        t.players[t.n] = player;
        ++t.n;
    }
    return t;
}

Mask legal(const char* hand, Trick trick, int player, const Config& cfg = kBelgian) {
    return legalMask(parseCards(hand), trick, kHearts, player, cfg);
}

void testTables() {
    for (int trump = 0; trump < kSuits; ++trump) {
        int total = 0;
        for (int c = 0; c < kCards; ++c) total += cardPoints(c, trump);
        CHECK_EQ(total + kLastTrickBonus, kDealPoints);
    }
    CHECK_EQ(parseCard("10H"), cardOf(kHearts, k10));
    CHECK(cardName(cardOf(kSpades, kJ)) == "JS");
    CHECK(maskString(parseCards("AD 7C JH")) == "7C AD JH");
}

void testTrickWinner() {
    // Trump order J 9 A 10 K Q 8 7; plain A 10 K Q J 9 8 7.
    CHECK_EQ(trickWinnerIndex(makeTrick({{0, "AH"}, {1, "9H"}, {2, "JH"}, {3, "10H"}}), kHearts), 2);
    CHECK_EQ(trickWinnerIndex(makeTrick({{0, "KC"}, {1, "10C"}, {2, "JC"}, {3, "AD"}}), kHearts), 1);
    CHECK_EQ(trickWinnerIndex(makeTrick({{0, "AC"}, {1, "7H"}, {2, "AS"}, {3, "8H"}}), kHearts), 3);
}

void testLegalMoves() {
    // Trump is hearts throughout. Leader is player 0.
    CHECK_EQ(legal("7C AC JH 9D", Trick{}, 0), parseCards("7C AC JH 9D"));

    // Plain suit led: follow, no obligation to beat ("bij de rest niet").
    CHECK_EQ(legal("7C AC JH 9D", makeTrick({{0, "10C"}}), 1), parseCards("7C AC"));

    // Trump led, opponent winning: go over if you can.
    CHECK_EQ(legal("9H 8H 7C", makeTrick({{0, "AH"}}), 1), parseCards("9H"));
    CHECK_EQ(legal("8H 7H 7C", makeTrick({{0, "AH"}}), 1), parseCards("8H 7H"));

    // Trump led, partner winning: any trump ("moet je volgen maar mag je alles smijten").
    CHECK_EQ(legal("9H 8H 7C", makeTrick({{0, "AH"}, {1, "7H"}}), 2), parseCards("9H 8H"));
    Config compat = Config::tanguyCompat();
    CHECK_EQ(legal("9H 8H 7C", makeTrick({{0, "AH"}, {1, "7H"}}), 2, compat), parseCards("9H"));

    // Can't follow, partner winning: anything.
    CHECK_EQ(legal("9H 8D 7S", makeTrick({{0, "AC"}, {1, "7C"}}), 2), parseCards("9H 8D 7S"));
    // Partner trumped and is winning: anything, no undertrump.
    CHECK_EQ(legal("7H 8D", makeTrick({{0, "AC"}, {1, "QH"}, {2, "7D"}}), 3), parseCards("7H 8D"));

    // Can't follow, opponent winning, no trump in the trick: trump (kopen).
    CHECK_EQ(legal("9H 8H 8D", makeTrick({{0, "AC"}}), 1), parseCards("9H 8H"));
    // ... unless you have no trump.
    CHECK_EQ(legal("8D 7S", makeTrick({{0, "AC"}}), 1), parseCards("8D 7S"));

    // Opponent trumped: overtrump if you can.
    CHECK_EQ(legal("KH 7H 8D", makeTrick({{0, "AC"}, {1, "QH"}}), 2), parseCards("KH"));
    // Opponent overtrumped your partner: overtrump the opponent.
    CHECK_EQ(legal("AH 7H 8D", makeTrick({{0, "AC"}, {1, "QH"}, {2, "KH"}}), 3), parseCards("AH"));
    // Can't overtrump: undertrump (onderkopen).
    CHECK_EQ(legal("KH 7H 8D", makeTrick({{0, "AC"}, {1, "JH"}}), 2), parseCards("KH 7H"));
    Config no_under = kBelgian;
    no_under.must_undertrump = false;
    CHECK_EQ(legal("KH 7H 8D", makeTrick({{0, "AC"}, {1, "JH"}}), 2, no_under), parseCards("KH 7H 8D"));
}

int points(const std::vector<Toon>& toons) { return totalPoints(toons); }

void testFindToons() {
    // derde 7-8-9 H and vierde J-A C.
    auto t = findToons(parseCards("7H 8H 9H JC QC KC AC 7S"), kBelgian);
    CHECK_EQ(t.size(), 2u);
    CHECK_EQ(points(t), 70);

    // [WB] example: carré of 9s plus 9-10-J-Q H -> carré + derde from 10 up.
    t = findToons(parseCards("9C 9D 9H 9S 10H JH QH 7C"), kBelgian);
    CHECK_EQ(points(t), 170);
    CHECK_EQ(t.size(), 2u);

    // Carré of jacks + derde Q-K-A beats vierde J-A.
    t = findToons(parseCards("JH QH KH AH JC JD JS 7C"), kBelgian);
    CHECK_EQ(points(t), 220);

    // Two carrés: jacks 200 + aces 100.
    CHECK_EQ(points(findToons(parseCards("JC JD JH JS AC AD AH AS"), kBelgian)), 300);

    // A run of 8 is one toon of 100, never split.
    t = findToons(parseCards("7H 8H 9H 10H JH QH KH AH"), kBelgian);
    CHECK_EQ(t.size(), 1u);
    CHECK_EQ(points(t), 100);
    CHECK(kindOf(t[0]) == ToonKind::kVijfde);

    // Four 7s: nothing by default, 50 by agreement.
    CHECK(findToons(parseCards("7C 7D 7H 7S AC KD QH 9S"), kBelgian).empty());
    Config low = kBelgian;
    low.carre_low_points = 50;
    CHECK_EQ(points(findToons(parseCards("7C 7D 7H 7S AC KD QH 9S"), low)), 50);

    // Equal-points overlap: four queens (100) + 10-J-K-A of hearts (vijfde = 100).
    // Keep the carré, because any carré beats any sequence in comparison.
    t = findToons(parseCards("QC QD QH QS 10H JH KH AH"), kBelgian);
    CHECK_EQ(t.size(), 1u);
    CHECK_EQ(points(t), 100);
    CHECK(t[0].carre && t[0].rank == kQ);

    // No wrap-around: A 7 8 is not a sequence.
    CHECK(findToons(parseCards("AH 7H 8H JC 9D KS 10S QD"), kBelgian).empty());
}

void testToonComparison() {
    auto toons = [](const char* hand) { return findToons(parseCards(hand), kBelgian); };
    {
        // Carré of queens (100) beats a vijfde (100).
        std::vector<Toon> d[kPlayers] = {toons("QC QD QH QS"), toons("7S 8S 9S 10S JS")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 3), 0);
    }
    {
        // Higher carré wins: 9s over aces.
        std::vector<Toon> d[kPlayers] = {toons("AC AD AH AS"), toons("9C 9D 9H 9S")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 3), 1);
    }
    {
        // Longer sequence wins; partner's lower toon counts with it.
        std::vector<Toon> d[kPlayers] = {toons("7C 8C 9C"), toons("7D 8D 9D KS QS JS"), toons("7S 8S 9S 10S")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 3), 0);
    }
    {
        // Equal length: higher top card.
        std::vector<Toon> d[kPlayers] = {toons("7C 8C 9C"), toons("8D 9D 10D")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 3), 1);
    }
    {
        // Equal: the trump sequence.
        std::vector<Toon> d[kPlayers] = {toons("8H 9H 10H"), toons("8D 9D 10D")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 3), 0);
    }
    {
        // Equal, neither trump: first player after the dealer.
        std::vector<Toon> d[kPlayers] = {{}, toons("8D 9D 10D"), toons("8C 9C 10C")};
        CHECK_EQ(toonWinningTeam(d, kHearts, 0), 1);  // player 1 sits first
        CHECK_EQ(toonWinningTeam(d, kHearts, 1), 0);  // player 2 sits first
    }
    {
        std::vector<Toon> d[kPlayers];
        CHECK_EQ(toonWinningTeam(d, kHearts, 0), -1);
    }
}

DealResult score(std::array<int, 2> cards, const std::vector<Toon> (&declared)[kPlayers],
                 int belot_team = -1, const Config& cfg = kBelgian, std::array<int, 2> tricks = {4, 4}) {
    return scoreDeal(cfg, /*takers=*/0, kHearts, /*dealer=*/3, cards, tricks, declared, belot_team);
}

void testScoring() {
    const std::vector<Toon> none[kPlayers];
    const auto vierde = findToons(parseCards("7S 8S 9S 10S"), kBelgian);

    DealResult r = score({90, 72}, none);
    CHECK(r.outcome == Outcome::kMade);
    CHECK_EQ(r.credited[0], 90);
    CHECK_EQ(r.credited[1], 72);

    r = score({70, 92}, none);
    CHECK(r.outcome == Outcome::kNat);
    CHECK_EQ(r.credited[0], 0);
    CHECK_EQ(r.credited[1], 162);

    {   // Toon decides: takers' vierde turns 80-82 into a made deal.
        const std::vector<Toon> d[kPlayers] = {vierde};
        r = score({80, 82}, d);
        CHECK(r.outcome == Outcome::kMade);
        CHECK_EQ(r.credited[0], 130);
        CHECK_EQ(r.credited[1], 82);
    }
    {   // Defenders' vierde: nat, defenders take 162 + the toon.
        const std::vector<Toon> d[kPlayers] = {{}, vierde};
        r = score({80, 82}, d);
        CHECK(r.outcome == Outcome::kNat);
        CHECK_EQ(r.credited[0], 0);
        CHECK_EQ(r.credited[1], 212);
    }
    {   // Takers' toon on nat goes to the defenders.
        const std::vector<Toon> d[kPlayers] = {findToons(parseCards("7S 8S 9S"), kBelgian)};
        r = score({60, 102}, d);
        CHECK(r.outcome == Outcome::kNat);
        CHECK_EQ(r.credited[1], 182);
    }

    // Belot counts toward the contract and stays with its holder on nat.
    r = score({70, 92}, none, /*belot_team=*/0);
    CHECK(r.outcome == Outcome::kNat);
    CHECK_EQ(r.credited[0], 20);
    CHECK_EQ(r.credited[1], 162);
    r = score({72, 90}, none, /*belot_team=*/0);
    CHECK(r.outcome == Outcome::kMade);
    CHECK_EQ(r.credited[0], 92);

    // Tie: hanging (default), or the alternatives.
    r = score({81, 81}, none);
    CHECK(r.outcome == Outcome::kHanging);
    CHECK_EQ(r.credited[0], 0);
    CHECK_EQ(r.credited[1], 81);
    CHECK_EQ(r.held, 81);
    Config make = kBelgian;
    make.tie_rule = TieRule::kTakersMake;
    CHECK(score({81, 81}, none, -1, make).outcome == Outcome::kMade);
    Config fail = kBelgian;
    fail.tie_rule = TieRule::kTakersFail;
    r = score({81, 81}, none, -1, fail);
    CHECK(r.outcome == Outcome::kNat);
    CHECK_EQ(r.credited[1], 162);

    {   // Ronde by the takers: 252 + their toon; the defenders' toon is void.
        const std::vector<Toon> d[kPlayers] = {findToons(parseCards("7S 8S 9S"), kBelgian), vierde};
        r = score({162, 0}, d, /*belot_team=*/1, kBelgian, {8, 0});
        CHECK(r.outcome == Outcome::kMade);
        CHECK_EQ(r.ronde_team, 0);
        CHECK_EQ(r.credited[0], 272);
        CHECK_EQ(r.credited[1], 20);  // only their belot
    }
    // Ronde by the defenders: nat with 252.
    r = score({0, 162}, none, -1, kBelgian, {0, 8});
    CHECK(r.outcome == Outcome::kNat);
    CHECK_EQ(r.credited[1], 252);
}

// Plays a whole deal from fixed hands with a fixed policy: the lowest or the
// highest legal card index.
DealResult playOut(Deal d, bool lowest) {
    while (d.phase() == Phase::kPlay) {
        Mask m = d.legalMoves();
        int card = lowestCard(m);
        if (!lowest) while (m) { card = lowestCard(m); m &= m - 1; }
        d.play(card);
    }
    return d.result();
}

void testBelotAndDealFlow() {
    // Player 0 holds every heart (trump): a run of 8 and K+Q. Every player
    // holds a run of 8 (100); only team 0's count, since it takes a ronde:
    // 252 + 100 (player 0) + 100 (player 2).
    const std::array<Mask, kPlayers> hands = {
        parseCards("7H 8H 9H 10H JH QH KH AH"), parseCards("7C 8C 9C 10C JC QC KC AC"),
        parseCards("7D 8D 9D 10D JD QD KD AD"), parseCards("7S 8S 9S 10S JS QS KS AS")};

    // Lowest card first: the queen goes before the king. [BR]: belot on
    // "la première des deux", so it still counts.
    DealResult r = playOut(Deal::fromPlay(kBelgian, 3, 0, kHearts, hands), true);
    CHECK_EQ(r.ronde_team, 0);
    CHECK_EQ(r.belot_team, 0);
    CHECK_EQ(r.credited[0], 252 + 200 + 20);
    CHECK_EQ(r.credited[1], 0);
    // [WB]: the king must come first; queen first forfeits the 20.
    r = playOut(Deal::fromPlay(kWikibooks, 3, 0, kHearts, hands), true);
    CHECK_EQ(r.belot_team, -1);
    CHECK_EQ(r.credited[0], 252 + 200);

    // Highest first: king before queen, belot counts.
    r = playOut(Deal::fromPlay(kWikibooks, 3, 0, kHearts, hands), false);
    CHECK_EQ(r.belot_team, 0);
    CHECK_EQ(r.credited[0], 252 + 200 + 20);

    // Tanguy's rules: the order doesn't matter (and no toon).
    r = playOut(Deal::fromPlay(Config::tanguyCompat(), 3, 0, kHearts, hands), true);
    CHECK_EQ(r.belot_team, 0);
    CHECK_EQ(r.credited[0], 252 + 20);

    // Declining to declare. Player 1 then sits first among equal runs of 8, so
    // team 1 holds the best toon and player 2 doesn't show; the ronde still
    // counts all of team 0's announced toons.
    Deal d = Deal::fromPlay(kBelgian, 3, 0, kHearts, hands);
    CHECK(d.canSetDeclare(0));
    d.setDeclare(0, false);
    d.play(lowestCard(d.legalMoves()));
    CHECK(!d.canSetDeclare(0));
    CHECK(!d.announced(0));
    CHECK(d.canSetDeclare(1));
    CHECK_EQ(playOut(d, false).credited[0], 252 + 100 + 20);  // only player 2's run

    // Announced with the first card, shown with the third, and only by the
    // team with the best toon ([WB]: "degene met geldige toon"): team 0's
    // run of 8 in trumps beats the other runs of 8.
    d = Deal::fromPlay(kBelgian, 3, 0, kHearts, hands);
    for (int i = 0; i < 8; ++i) d.play(lowestCard(d.legalMoves()));  // 2 tricks
    CHECK(d.announced(0) && d.announced(1));
    CHECK(d.announcement(1).kind == ToonKind::kVijfde);
    CHECK_EQ(d.announcement(1).carre_rank, -1);
    CHECK(!d.shown(0));
    for (int i = 0; i < 4; ++i) d.play(lowestCard(d.legalMoves()));  // trick 3
    CHECK(d.shown(0) && d.shown(2));
    CHECK(!d.shown(1) && !d.shown(3));
    Config losers = kBelgian;
    losers.toon_losers_show = true;
    d = Deal::fromPlay(losers, 3, 0, kHearts, hands);
    for (int i = 0; i < 12; ++i) d.play(lowestCard(d.legalMoves()));
    for (int p = 0; p < kPlayers; ++p) CHECK(d.shown(p));
    // [CGH]: shown after the first trick, once every announcement is in.
    Config early = kBelgian;
    early.toon_show_trick = 1;
    d = Deal::fromPlay(early, 3, 0, kHearts, hands);
    for (int i = 0; i < 3; ++i) d.play(lowestCard(d.legalMoves()));
    CHECK(!d.shown(0));
    d.play(lowestCard(d.legalMoves()));
    CHECK(d.shown(0) && d.shown(2) && !d.shown(1));
}

void testAnnouncement() {
    // The best toon is announced; a carré with its rank.
    auto best = [](const char* hand) { return announcementOf(findToons(parseCards(hand), kBelgian), kHearts); };
    Announcement a = best("JC JD JH JS 7C 8C 9C 10S");  // carré of jacks + derde
    CHECK(a.kind == ToonKind::kCarre);
    CHECK_EQ(a.carre_rank, kJ);
    a = best("7C 8C 9C JS QS KS AS 9D");  // derde + vierde
    CHECK(a.kind == ToonKind::kVierde);
    CHECK_EQ(a.carre_rank, -1);
}

void testBidding() {
    std::array<int, kCards> deck;
    for (int i = 0; i < kCards; ++i) deck[i] = i;
    // Dealer 3: player 0 gets 0-2 and 12-13, ..., the blik is deck[20] = 20 (JH).
    Deal d(kBelgian, 3, deck);
    CHECK_EQ(d.blik(), 20);
    CHECK_EQ(d.currentPlayer(), 0);
    CHECK_EQ(popcount(d.hand(0)), 5);
    CHECK(d.phase() == Phase::kBid1);
    for (int i = 0; i < 4; ++i) d.bid(Bid::pass());
    CHECK(d.phase() == Phase::kBid2);
    CHECK_EQ(d.currentPlayer(), 0);
    CHECK_EQ(d.legalBids().size(), 4u);  // pass + 3 suits, not hearts
    bool threw = false;
    try { d.bid(Bid::choose(kHearts)); } catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
    d.bid(Bid::pass());
    d.bid(Bid::choose(kSpades));  // player 1 makes spades
    CHECK(d.phase() == Phase::kPlay);
    CHECK_EQ(d.taker(), 1);
    CHECK_EQ(d.trump(), kSpades);
    CHECK(d.hand(1) & bit(20));  // took the blik
    Mask all = 0;
    for (int p = 0; p < kPlayers; ++p) {
        CHECK_EQ(popcount(d.hand(p)), 8);
        all |= d.hand(p);
    }
    CHECK_EQ(all, 0xFFFFFFFFu);
    CHECK_EQ(d.currentPlayer(), 0);  // player after the dealer leads

    Deal passed(kBelgian, 0, deck);
    for (int i = 0; i < 8; ++i) passed.bid(Bid::pass());
    CHECK(passed.phase() == Phase::kDone);
    CHECK(passed.result().outcome == Outcome::kAllPassed);
}

DealResult fakeResult(Outcome o, int takers, std::array<int, 2> credited, int held = 0) {
    DealResult r;
    r.outcome = o;
    r.takers = takers;
    r.credited = credited;
    r.held = held;
    return r;
}

void testMatch() {
    CHECK_EQ(recordedPoints(75, kBelgian), 8);
    CHECK_EQ(recordedPoints(74, kBelgian), 7);
    CHECK_EQ(recordedPoints(162, kBelgian), 16);
    CHECK_EQ(recordedPoints(0, kBelgian), 0);

    Match m(kBelgian, 0);
    m.record(fakeResult(Outcome::kHanging, 0, {0, 81}, 81));
    CHECK_EQ(m.scores()[1], 8);
    CHECK_EQ(m.pot(), 8);
    CHECK_EQ(m.dealer(), 1);
    m.record(fakeResult(Outcome::kHanging, 1, {81, 0}, 81));  // pot accumulates
    CHECK_EQ(m.pot(), 16);
    m.record(fakeResult(Outcome::kNat, 0, {0, 162}));        // defenders (team 1) win it
    CHECK_EQ(m.pot(), 0);
    CHECK_EQ(m.scores()[0], 8);
    CHECK_EQ(m.scores()[1], 8 + 16 + 16);

    // Redeals: the deal passes on at once by default...
    Match r(kBelgian, 2);
    r.record(fakeResult(Outcome::kAllPassed, -1, {0, 0}));
    CHECK_EQ(r.dealer(), 3);
    // ... or after the same dealer tried twice.
    Config same = kBelgian;
    same.redeal_next_dealer = false;
    Match s(same, 2);
    s.record(fakeResult(Outcome::kAllPassed, -1, {0, 0}));
    CHECK_EQ(s.dealer(), 2);
    s.record(fakeResult(Outcome::kAllPassed, -1, {0, 0}));
    CHECK_EQ(s.dealer(), 3);

    // Race to 101: both past it and equal plays on; a lead wins ([WB]: any lead).
    Match e(kWikibooks, 0);
    for (int i = 0; i < 6; ++i) e.record(fakeResult(Outcome::kMade, 0, {90, 72}));  // 9 / 7 a deal
    CHECK(!e.finished());
    e.record(fakeResult(Outcome::kMade, 0, {162, 0}));
    e.record(fakeResult(Outcome::kMade, 0, {0, 0}));
    CHECK_EQ(e.scores()[0], 70);
    Match tie(kWikibooks, 0);
    for (int i = 0; i < 7; ++i) tie.record(fakeResult(Outcome::kMade, 0, {150, 150}));
    CHECK_EQ(tie.scores()[0], 105);
    CHECK(!tie.finished());
    tie.record(fakeResult(Outcome::kMade, 0, {10, 0}));
    CHECK(tie.finished());
    CHECK_EQ(tie.winner(), 0);

    // [BR]: a two-point lead is needed past 101.
    Match g(kBelgian, 0);
    for (int i = 0; i < 7; ++i) g.record(fakeResult(Outcome::kMade, 0, {150, 150}));
    g.record(fakeResult(Outcome::kMade, 0, {10, 0}));  // 106-105
    CHECK(!g.finished());
    g.record(fakeResult(Outcome::kMade, 0, {10, 0}));  // 107-105
    CHECK(g.finished());
    CHECK_EQ(g.winner(), 0);
}

RecordedDeal rec(Outcome o, std::array<int, 2> credited, int held = 0, const Config& cfg = kBelgian) {
    return recordDeal(fakeResult(o, /*takers=*/0, credited, held), cfg);
}

void testRecording() {
    // [BR] example: takers 98, defenders 64 -> 60 -> 6; takers 16 - 6 = 10.
    RecordedDeal r = rec(Outcome::kMade, {98, 64});
    CHECK_EQ(r.scores[0], 10);
    CHECK_EQ(r.scores[1], 6);
    // ... with a tierce for the takers: 2 more.
    r = rec(Outcome::kMade, {118, 64});
    CHECK_EQ(r.scores[0], 12);
    CHECK_EQ(r.scores[1], 6);
    // Defenders at 75 round up: 8, takers 8 (each rounded on its own: 9 + 8).
    r = rec(Outcome::kMade, {87, 75});
    CHECK_EQ(r.scores[0], 8);
    CHECK_EQ(r.scores[1], 8);
    r = rec(Outcome::kMade, {87, 75}, 0, kWikibooks);
    CHECK_EQ(r.scores[0], 9);
    CHECK_EQ(r.scores[1], 8);
    // Defenders' own toon: 82 + 50 -> 13; takers 16 + 2 (belot) + ... - 8.
    r = rec(Outcome::kMade, {150, 132});  // e.g. takers 80 + 50 + 20, defenders 82 + 50
    CHECK_EQ(r.scores[1], 13);
    CHECK_EQ(r.scores[0], 28 - 13);
    // Nat: 16 (+ toon) to the defenders; the takers keep only their belot.
    r = rec(Outcome::kNat, {20, 212});
    CHECK_EQ(r.scores[0], 2);
    CHECK_EQ(r.scores[1], 21);
    // Ronde: 25.
    r = rec(Outcome::kMade, {252, 0});
    CHECK_EQ(r.scores[0], 25);
    CHECK_EQ(r.scores[1], 0);
    // Hanging 81-81: defenders 8, 8 held.
    r = rec(Outcome::kHanging, {0, 81}, 81);
    CHECK_EQ(r.scores[0], 0);
    CHECK_EQ(r.scores[1], 8);
    CHECK_EQ(r.held, 8);
}

}  // namespace

int main() {
    testTables();
    testTrickWinner();
    testLegalMoves();
    testFindToons();
    testToonComparison();
    testScoring();
    testBelotAndDealFlow();
    testAnnouncement();
    testBidding();
    testMatch();
    testRecording();
    return checkResult("test_rules");
}
