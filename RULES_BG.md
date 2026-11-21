# Bulgarian Bridge-Belot — engine rules

The rules of `cpp/bg/` are those of Nikolay Kostov's
[BelotGameEngine](https://github.com/NikolayIT/BelotGameEngine) (MIT,
`src/Belot.Engine`, version 2.0), the engine whose bots we want to beat. Where
his code and his [rules text](https://github.com/NikolayIT/BelotGameEngine/blob/master/etc/Rules.md)
differ, **the code wins**, because the matches are played in his engine. The
C++ engine is checked against logs of his engine card for card
(`bg/oracle`, `cpp/tests/test_bg.cpp`). The check covered 12,000 random games,
62,187 deals and 1.49 million decisions with no differences.

Seats S, E, N, W (0–3) play in that order. S/N and E/W are the teams. Cards use
the same numbering as the Belgian engine: suit × 8 + rank, ranks 7 8 9 10 J Q K
A, suits ♣ ♦ ♥ ♠.

## Deal and auction

- A fresh shuffle every deal. Five cards are dealt one at a time, S E N W. The
  last three come one at a time after the auction.
- The first seat bids first and leads the first trick. It moves one seat on
  every deal, passed-out deals included.
- Bids in increasing order: ♣, ♦, ♥, ♠, no trumps, all trumps. A bid must beat
  the current contract.
- An opponent of the declarer may double. The declaring team may then
  redouble. Doubling keeps the contract with the declarer. A new, higher
  contract clears any double. After a redouble nobody may double again.
- A seat whose only option is to pass passes automatically. Example: everyone
  after an all-trumps redouble.
- The auction ends after three passes in a row once someone has bid, or after
  four passes if nobody has. Four passes mean a passed-out deal: no score, and
  the hanging points stay on the table.

## Card play

- Trick winner: the highest trump, otherwise the highest card of the led suit.
  - Trump order: J 9 A 10 K Q 8 7.
  - Plain order: A 10 K Q J 9 8 7.
  - In all trumps every suit uses trump order. In no trumps every suit uses
    plain order.
- **All trumps, or trump led in a suit contract.** Follow suit, and beat the
  highest card of the led suit if you can, even when partner is winning.
  Without the suit, play any card.
- **No trumps.** Follow suit, otherwise any card.
- **Suit contract, plain suit led.**
  - With the suit: follow, with any card of it. There is no obligation to beat.
  - Without the suit, if partner holds the trick: any card.
  - Without the suit, if an opponent has ruffed: overruff if you can,
    otherwise any card. There is **no undertrumping**.
  - Without the suit, if nobody has ruffed: you must ruff with any trump.
  - Without trumps: any card.
- A seat with a single legal card plays it automatically.

## Declarations

- **When.** Trick 1, before a seat's first card, except in no trumps. The seat
  is offered:
  - carrés: four 9s, 10s, J, Q, K or A, but not 7s or 8s;
  - sequences of 3, 4, 5, 6, 7 or 8 cards in one suit, in the order
    7 8 9 10 J Q K A;
  - when a carré card is also part of a run, that sequence as an alternative.
- **Declaring.** The seat declares any subset, in any order. Of declarations
  that share a card, only the first counts. Declaring is optional.
- **Values.**

  | Combination | Points |
  |---|---|
  | Tierce (3 in sequence) | 20 |
  | Quarte (4) | 50 |
  | Quint (5–8) | 100 |
  | Carré of 10s, Q, K or A | 100 |
  | Carré of 9s | 150 |
  | Carré of J | 200 |
  | Belot | 20 |

- **Comparison, before trick 2.** Carrés and sequences are compared
  separately, so both teams can score: one with a carré, the other with a
  sequence.
  - In each group the highest combination wins, by value, then type, then
    top card. Among carrés of equal value: A > 10 > K > Q.
  - The winning team scores every combination it declared in that group.
  - If the best sequences of the two teams are equal, nobody scores any
    sequence.
  - Opponents learn the type of a declaration, not its cards.
- **Belot.** K and Q of the trump suit, or of any suit in all trumps. Never in
  no trumps.
  - Claimed with whichever of the two is played first, and only while holding
    the other.
  - In all trumps the card must follow the led suit or lead the trick.
  - Optional, and **impossible when the card is forced** (the only legal
    card).
  - Always scores 20.

## Score of a deal

- **Raw totals.** Each team counts its cards, plus 10 for the last trick, plus
  its scoring declarations.
  - Card points:

    | Card | 7 | 8 | 9 | 10 | J | Q | K | A |
    |---|---|---|---|---|---|---|---|---|
    | Trump (and every suit in all trumps) | 0 | 0 | 14 | 10 | 20 | 3 | 4 | 11 |
    | Plain (and every suit in no trumps) | 0 | 0 | 0 | 10 | 2 | 3 | 4 | 11 |

  - No trumps doubles the totals, **then** a team that took every trick adds
    90.
- **Doubled or redoubled contract.** All points of the deal, rounded as a
  winner's, ×2 or ×4, plus the hanging points, go to the team with more. An
  equal score leaves all of it hanging.
- **Contract failed** (the declaring team has less). The other team scores
  everything, plus the hanging points. This total is rounded half to even:
  165 → 16 and 175 → 18. That is C# `Math.Round`.
- **Tie.** The other team scores its half. The declaring team's half is added
  to the hanging points.
- **Contract made.** Each team scores its own total, and the team with more
  also takes the hanging points.
- **Rounding of a team's total to game points.**
  - Suit contract: …7–9 rounds up and …0–5 rounds down. …6 rounds down for the
    team with more and up for the team with less.
  - All trumps: the same rule with …4 as the pivot.
  - No trumps: half to even.

## The game

- Game points accumulate until a team has **151 or more and is ahead**, at the
  end of a deal in which it scored.
- A deal where one team took every trick (capot) never ends the game, and
  neither does a passed-out deal.
- Hanging points are carried to the next deal as the score function returns
  them.
