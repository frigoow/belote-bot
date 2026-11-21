# Belgian Belote -- engine rules

The rules the engine implements. The profile is that of **belote-rebelote.be**
(`Config::belgian()`, the default). Where it is silent, [WB] and the house
rules fill in. Where the sources are silent or disagree, the default chosen
here is marked **[default]** and is a config flag (see the table at the end),
so it can change without touching the engine. `Config::wikibooks()` keeps the
[WB] + house profile the engine started from.

Sources:
- **[BR]** [belote-rebelote.be/regles-belote](https://www.belote-rebelote.be/regles-belote)
  and [/compter-points-belote](https://www.belote-rebelote.be/compter-points-belote),
  a Belgian online implementation. This is the primary profile (chosen
  2026-11-21). It is silent on deal direction, toon timing and comparison,
  runs longer than 5, and rounding at exactly 5.
- **[WB]** [nl.wikibooks.org/wiki/Kaartspel/Belot](https://nl.wikibooks.org/wiki/Kaartspel/Belot).
  The original primary source. Its trump selection matches the house rules the
  players described, and it agrees with [BR] there.
- **[CGH]** [cardgameshub.com/nl/kenniscentrum/spelregels-belote](https://cardgameshub.com/nl/kenniscentrum/spelregels-belote/).
- **[House]** House rules confirmed by the players (Jonas, Niels). Where these
  differ from [WB] or [CGH], the house rule wins. Confirmed: [WB] is right, the
  match is played to 101. Where [BR] differs (recording, the 2-point lead,
  belot order), [BR] wins, and the house choice is kept as a flag.

Dutch game terms are kept in *italics*.

---

## 1. Players, deck, dealing

- 4 players, two fixed partnerships sitting opposite each other. Seats 0–3,
  partners are `s` and `s+2`.
- 32 cards: 7, 8, 9, 10, J (*zot*, *boer*), Q (*dame*), K (*heer*), A (*aas*)
  in four suits.
- Everything goes **clockwise** (*in wijzerzin*): dealing, bidding, play.
  [CGH] mentions counter-clockwise as a variant.
- **Deal**: 3 cards each, then 2 each, starting with the player after the
  dealer. The next card of the remaining 12 is turned face up on top of the
  stack: the **blik**.
- After trump is chosen:
  - the trump chooser (*troefkiezer*) takes the blik and gets 2 more cards,
  - everyone else gets 3,
  - dealt clockwise from the player after the dealer, so everyone ends up with 8.
- **Redeal**: if nobody makes trump in either round, the next player deals
  [House, BR]. [WB]'s "two fruitless attempts" are read as the two bidding rounds.
- The next deal passes clockwise to the next dealer.
- Shuffling and cutting are cosmetic for the engine (random deal).

## 2. Choosing trump

1. **Round 1**: starting with the player after the dealer, each player in turn
   either takes the blik's suit as trump or passes (*pas*). The dealer speaks last.
2. **Round 2**: if all four passed, again starting with the player after the
   dealer, each player in turn may name a trump suit himself or pass (*pas nog*).
   - The suit named must differ from the blik's suit ("een andere troef" [House];
     "autre que celle de la carte retournée" [BR]).
   - The player still takes the blik into their hand [House].
3. If all four pass again, the cards are redealt (§1).

- The first player to take ends the bidding. That player's team is the
  **taking team**; the other team defends.
- Neither source has other contracts (sans atout / tout atout) or doubling.
- There is no special rule for a jack as the blik.

## 3. Card order and points

| Trump suit | J | 9 | A | 10 | K | Q | 8 | 7 |
|---|---|---|---|---|---|---|---|---|
| Points | 20 | 14 | 11 | 10 | 4 | 3 | 0 | 0 |

| Other suits | A | 10 | K | Q | J | 9 | 8 | 7 |
|---|---|---|---|---|---|---|---|---|
| Points | 11 | 10 | 4 | 3 | 2 | 0 | 0 | 0 |

- Trump suit 62, each other suit 30, plus **10 for the last trick** (*laatste
  slag*) = **162** card points per deal.
- A trick is won by the highest trump in it, otherwise by the highest card of
  the suit led.
- The winner of a trick leads the next one. The player after the dealer leads
  the first trick.

## 4. Playing a card

Let **"partner is winning"** mean the card currently winning the trick was
played by your partner.

Legal cards, in order:

1. **You can follow suit** → you must (*bekennen*).
   - If trump was led and an **opponent** is winning, you must play a higher
     trump if you have one (*boven gaan*).
   - If trump was led and your partner is winning, any trump will do: "als
     het aan de maat ligt moet je volgen maar mag je alles smijten" [House, WB].
     [CGH] states no partner exception here.
   - Unlike Manille, there is **no obligation to beat in a non-trump suit**
     ("bij de rest niet" [House]).
   - If a non-trump suit was led, any card of that suit will do.
2. **You can't follow suit, and your partner is winning** → play any card
   (house rule, consistent with both sources).
3. **You can't follow suit, and an opponent is winning:**
   - If you have no trump → play any card.
   - If no trump is in the trick yet → you must trump (*kopen*, *introeven*);
     any trump will do.
   - If the opponent is winning with a trump:
     - you must overtrump if you can (*overtroeven*);
     - if you can't, you must still play a lower trump (*onderpissen*,
       *onderkopen*). Both sources, [BR] ("la sous-coupe obligatoire") and
       [House] agree. This differs from Manille, where you may then discard a
       non-trump.

An illegal play counts as cheating at the table. The engine simply never offers it.

## 5. Declarations

### 5.1 Toon (sequences and four of a kind)

**Sequences** use the order 7 8 9 10 J Q K A, in one suit, with no wrap-around.

| Toon | Points |
|---|---|
| *Derde* (3 in sequence) | 20 |
| *Vierde* (4) | 50 |
| *Vijfde* (5) and longer, up to 8 | 100 [CGH] |
| Carré of Q, K, A or 10 | 100 |
| Carré of 9s | 150 |
| Carré of jacks | 200 |
| Carré of 7s or 8s | 0 (some places count 50 [WB]: flag) |

- A player may declare several toons, but **a card can be in only one toon**
  [WB]. The engine picks the split worth the most points. A carré may be left
  out to keep a longer sequence. A sequence is always its whole run and is
  never split, so a run of 8 is one toon of 100.
- **Announcing**: a player announces their toon when playing their first card
  [WB]. They say the kind of their **best** toon (*derde*, *vierde*, *vijfde*
  or carré), and for a carré also its rank, e.g. "carré van boeren"
  **[default]**. The height of a sequence isn't said. Declaring is optional
  **[default]**: a player declares all their toons or none of them.
- **Showing**: at the 3rd trick, when playing their 3rd card, the declarer
  shows their toons [WB, House]. Only the team holding the **highest toon**
  shows ("degene met geldige toon" [WB]); a beaten toon lapses and stays
  hidden **[default]**, flag `toon_losers_show`. A toon that isn't shown in time
  scores nothing. Cards already played in tricks 1–2 still count. [CGH] settles
  the comparison after the first trick instead (flag; the toons are then shown
  once trick 1 is complete).
  - [WB]'s trick-3 exchange between equal toons ("men zegt wat de hoogste kaart
    van zijn toon is") is not modelled: the engine compares the toons itself,
    and the public outcome is who shows.
  - [WB] lets the lower toon show in trick 4 if the winner forgets to show.
    Players in the engine never forget, so this is not modelled.
- **Comparing**: only the team with the **highest toon** scores, and then it
  scores all of its toons, including the partner's lower ones. The other
  team's toons score nothing. To find the highest toon:
  1. **a carré beats any sequence** [CGH]; a higher carré beats a lower one
     (J > 9 > A > 10 > K > Q **[default]**);
  2. a longer sequence beats a shorter one;
  3. between equal-length sequences, the higher top card wins;
  4. if still equal, a sequence in the trump suit wins;
  5. if still equal, the sequence of the player who comes first clockwise from
     the player after the dealer wins [CGH, WB].

### 5.2 Belot / rebelot

- K (*belot*) and Q (*rebelot*) of trumps in the same hand: 20 points.
- The player says "belot" when playing the first of the two and "rebelot" when
  playing the second.
- [BR] says "belote" is called "en jouant la première des deux": either card
  may come first **[default]**. [WB] says the king must be played first; with
  flag `belot_queen_first_forfeits`, playing the queen first forfeits the 20.
- Belot is always 20 [CGH]. It is never compared, always counts for the
  holder's team (even on *nat* and even if that team takes no trick, [CGH]),
  and may share cards with a toon **[default]**.

## 6. Scoring a deal

Each team's total = card points + scoring toon + belot.

| Result | Condition | Taking team | Defenders |
|---|---|---|---|
| **Made** | takers' total > defenders' total | own total | own total |
| **Nat** (failed) | takers' total < defenders' total | its belot only | 162 + the scoring toon (whichever team owns it) + own belot [CGH] |
| **Tie** | totals equal | see `tie_rule` | own total |
| **Ronde** | one team takes all 8 tricks | 252 + its toon + belot | only its own belot |

- **Ronde**: the Ronde row is the same whichever team takes all 8 tricks.
  [WB] calls it *ronde*; [CGH] calls it a 90-point bonus. All of the ronde
  team's declared toons count, even when the other team held the higher toon,
  because a team that takes no trick scores no toon [CGH].
- A ronde by the takers is made. A ronde by the defenders is nat, worth 252
  to the defenders.

- **`tie_rule`**: the two sources disagree.
  - `hanging` **[WB, House, BR]**: the defenders score their total. The takers'
    points are held over (*op den boek*) and go to the team that wins the next
    deal (made or nat), accumulating over consecutive ties.
  - `takers_make` [CGH: "minimaal evenveel"]: a tie counts as made.
- [WB] phrases the target as "at least 81". That is the same thing when nobody
  has toon or belot.
- **Recording** (`complement` mode, [BR], the default): the defenders' total
  is rounded to the nearest ten and written down in units; the takers get the
  rest of the deal's units: 16, plus every toon and belot point divided by 10.
  [BR]'s example: defenders 64 → 6, takers 16 − 6 = 10; with a *derde* for
  the takers, 12. [BR] doesn't say how 5 rounds; the engine rounds it up
  (75 → 8) as [WB] does **[default]**. A *nat* gives the defenders 16 + toon
  (+ own belot), a ronde 25, a hanging 81–81 writes 8 and holds 8.
  - `units` mode ([WB]): each team's total is rounded on its own, 5 up
    (75 → 8, 74 → 7), so the two need not add up to 16.
  - `raw` mode ([CGH]): points are recorded as-is.

## 7. The match

- **[WB, House, BR]**: the first team to reach **101 recorded units** wins
  (*de boom*), and **[BR]**: with a lead of at least 2 ("avec au moins 2
  points d'écart"). [WB] needs any lead (`match_win_margin = 0`).
- [CGH] plays to **1001 raw points** ("vaak", so a custom, not a rule).
- If both teams pass the target in the same deal, the higher score wins if it
  leads by the margin; otherwise another deal is played **[default]**.

---

## Config flags

| Flag | Default | Alternative |
|---|---|---|
| `redeal_next_dealer` | true [House] | the same dealer deals again first |
| `round2_allow_blik_suit` | false [House] | true ([CGH] doesn't exclude it) |
| `trump_lead_partner_must_overtrump` | false [WB, House] | true ([CGH] lists no exception) |
| `must_undertrump` | true | play anything when you can't overtrump |
| `carre_low_points` | 0 | 50 for four 7s / four 8s |
| `toon_optional` | true | must declare |
| `toon_show_trick` | 3 [WB, House] | 1 ([CGH]) |
| `toon_losers_show` | false [WB] | true: every announced toon is shown |
| `belot_queen_first_forfeits` | false [BR] | true [WB]: queen first forfeits the 20 |
| `tie_rule` | `hanging` [WB, House, BR] | `takers_make` [CGH] |
| `score_mode` | `complement` [BR] | `units` [WB, House], `raw` [CGH] |
| `match_target` | 101 (units) [House, BR] | 1001 (raw) |
| `match_win_margin` | 2 [BR] | 0 [WB] |
| `ronde_points` | 252 | |
