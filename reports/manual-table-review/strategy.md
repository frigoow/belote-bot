# Belgian Belote strategy from four matches

The first rule I would give the agent concerns discards on partner’s winning trick. Before counting how many points a discard banks, ask **what trick the card could win later, and what suit discarding it opens for a ruff**. Both questions changed a contract in these logs.

I played seat 0 through match 505, including the auction. In matches 501, 502 and 503 I played the opening deal from seats 0, 1 and 2 respectively. I saw my hand, legal choices and public play. The project’s existing rule2 player filled the other seats and finished those three matches. Everybody declared their toons. Across the four matches there were 38 auctions, four redeals and 34 scored deals. I selected ten decisions for paired replays. Each pair has the same cards, auction and history up to the decision. Both branches use the same continuation policy. These are examples of how a rule can gain or cost points on particular deals. They do not estimate its win rate.

## 1. Keep a side winner when you can use it to take the lead

In match 501, deal 3, seat 2 was last to play. Partner had already ruffed a heart. Throwing A♠ banked eleven points. Throwing J♠ banked two and left A♠ in hand alongside a strong trump, 9♣. Seat 2 later won the lead with 9♣, cashed A♠ over Q♠ and partner’s K♠, then led a diamond to partner’s ace. The takers went from **99 card points and nat** to **128 and made**. The defenders’ 40 toon made 102 the takers’ winning card-point threshold. The second declaration became visible after the discard. The entry and the ace were already in seat 2’s hand. [Pair A](paired-replays.log)

Apply the same reasoning to a promoted ten. In match 503, deal 5, every trump and A♥ had gone. Seat 2 could discard 10♥ or J♠ on partner’s A♣. Keeping 10♥ let it beat K♥ on the next trick and put seat 2 on lead for the last. The defenders took **87 card points instead of 48** and wrote **9 units instead of 5**, even though the takers’ 150 toon secured the contract in both lines. A revealed honor also matters. In match 501, deal 6, keeping A♣ let seat 1 catch the 10♣ an opponent had shown in a toon. [Pairs B and C](paired-replays.log)

The exception is a winner with little future work. In match 502, deal 4, keeping A♣ gained the lead later but both lines wrote 12 units for the takers. In the live match, deal 10, both opponents had discarded on spades while our team was winning. Their obligation to trump proved both were out of diamonds. When partner led 10♦ at trick seven, the only outstanding trump was Q♦, also with partner. I could put A♣ on the ten because our team could account for both remaining tricks. [Pair H](paired-replays.log) · [Match 505](match-505.log)

## 2. Create a void in a suit partner can actually return

In match 502, deal 1, I was void in diamonds and last to play on partner’s winning K♦. My options included a singleton 8♣ and J♥ from J♥–10♥. Partner had publicly shown J♣–Q♣–K♣ and was about to lead. Discarding 8♣ gave partner’s club length a purpose. On the return, an opponent played A♣, I ruffed and the fourth player followed with 10♣. In the paired replay my team made with **72 card points plus 40 toon**. Discarding J♥ collected two points immediately but forced me to follow with 8♣ next. We finished **nat on 59 plus 40**. [Pair D](paired-replays.log)

For the agent, give a singleton discard extra weight when three things coincide. Partner will lead, public play or a shown toon gives partner a likely return in that suit, and you still have a trump for the return. If an opponent can still overruff, include their seat position in the choice.

## 3. Spend the right trump on a ruff

In that same club trick I held 10♠ and Q♠. The only unseen trumps were 9♠ and A♠. Either could beat both my cards, and no remaining trump could beat just the queen. The two cards therefore had identical protection against an overruff. The trick already contained A♣, and both unseen trumps outranked anything I could save. Ruffing with 10♠ banked seven more points on this deal. With the same continuation policy, the takers wrote **12 units versus 11**. This is a tie-breaker when the current ruff looks more likely to hold than a later one. An outstanding trump ranked between your candidates changes the decision. [Pair E](paired-replays.log)

When partner already owns the trick and you are last, include the cost of taking the lead away. In match 501, deal 4, discarding 10♦ on partner’s winning 10♥ kept A♠ available to ruff the last trick. The takers made with 111 card points. Ruffing partner with A♠ left the last trick to the defenders and sent the takers nat with 84 against 78 plus 20 toon. [Pair J](paired-replays.log)

## 4. Stop drawing trump when the remaining trump belongs to partner

In match 502, deal 10, both opponents discarded on the second trump lead. By trick five, the taker held 7♠, 10♦, K♦ and J♦. The only other trump, 8♠, had to be with partner. Leading 7♠ forced partner’s 8♠ out and spent both partnership trumps on a zero-point trick. Leading J♦ drove out A♦. The taker could then ruff A♣ with 7♠, cash 10♦ and let partner ruff the last diamond with 8♠. The partnership gained **17 card points and one recorded unit**. A transfer of the lead can justify playing trump to partner. Here there was a whole side suit to establish and two separate ruffs to preserve. [Pair F](paired-replays.log)

## 5. Count the units that keep the match alive

In match 502, deal 12, the takers started **91–98**. On a made 16-unit deal, conceding six units meant losing **101–104**. Conceding four meant **103–102** and another deal because the lead must be two. At trick five, the defenders already held 35 card points, enough for four units. The immediate objective was to hold them below 55.

Seat 0 could discard either singleton, 8♣ or 7♠, on partner’s winning 10♥. Both opponents had shown out of trumps. A♣, 10♣ and K♣ had gone, while A♠, 10♠ and K♠ were all still live. Discarding 7♠ created a void in the suit with the expensive honors. Partner ruffed the club return and seat 0 ruffed the final spade. The defenders stayed on 35. Discarding 8♣ let the defenders collect A♠, 10♠ and K♠ together and reach 60. The spade discard bought exactly one more deal. The opponents won that deal and the match **119–107**. [Pair G and its continuation](paired-replays.log)

The same arithmetic can identify plays with equal value. In the live match, deal 9, leading my losing 7♠ at trick seven saved the last trump for the final trick. That gained ten card points over cashing the trump first. Both lines set the takers and wrote exactly **20–0**. [Pair I](paired-replays.log)

## Table records

Seats 0 and 2 are partners. All score pairs in the logs list their team first. Card totals include the last-trick bonus. A ronde adds 90 when the deal is scored. I checked the recorded tricks against reconstructed hands for follow-suit, trump and overtrump obligations, trick winners and points. The paired replay log includes both continuations, including those that produced the same recorded score.

| Match | Seat played manually | Final score, seats 0/2–1/3 |
| --- | --- | --- |
| [501](match-501.log) | Seat 0, opening deal | 102–76 |
| [502](match-502.log) | Seat 1, opening deal | 101–104 |
| [503](match-503.log) | Seat 2, opening deal | 110–34 |
| [505](match-505.log) | Seat 0, full match | 101–69 |

The [project rules](../../RULES.md) define the Belgian profile used at this table. The strategy judgments came from the games and their replays.
