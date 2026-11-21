# Kleurenwiezen bots and rules

## Which rules define the game?

### Takeaway

Whisthub and IWWA both distinguish ordinary Wiezen from Kleurenwiezen. Kleurenwiezen is also called Colour Whist and, in French, whist à la couleur. It is a 52-card game with individual scores and partnerships chosen by the deal's contract.

### Cited Findings

- In Whisthub's ordinary Wiezen, the last dealt card supplies the ordinary trump. In Kleurenwiezen, bidding chooses the contract, trump and alliances, aside from Troel determining these specially. Whisthub's French pages call the two games Wiezen and Whist à la couleur. [Wiezen lobby and rules distinction](https://www.whisthub.com/fr/wiezen) · [Kleurenwiezen lobby in French](https://www.whisthub.com/fr) · [Pagat's rules from players and operators](https://www.pagat.com/boston/kleurenwiezen.html)
- Whisthub's documented Kleurenwiezen profile uses 52 cards, 13 per player, rotating dealer, follow suit with optional trump when void and no Belote-style obligation to overtrump. A suit proposal and an acceptance form a pair for at least 8 tricks. Solo contracts, Troel, Abondance, Misery and Grand Slam alter objective, side size and/or trump. Misery variants may involve private discards, a player's hand becoming public, and concurrent bidders. Whisthub defaults to a 5-trick solo minimum and zero-sum per-deal settlement, and optionally supports a 6-trick minimum, Small Misery on/off, Piccolo and Troel on/off. [Whisthub rules](https://www.whisthub.com/rules) · [Rule options](https://www.whisthub.com/premium) · [Developer confirms only follow-suit rather than overtrump](https://www.whisthub.com/blog/the-kingen-ai-has-been-cheating)
- IWWA publishes a distinct Kleurenwiezen tournament rulebook. It specifies at least 8 tricks for a pair, a 6-trick solo minimum, shuffle each deal, and a nonzero-sum tournament scoring system. The Whisthub profile cuts only between deals. A separate club rulebook describes optional trump when void. [IWWA rules index](https://www.whistiwwa.com/kopie-van-reglementen-3) · [IWWA tournament rules](https://www.whistiwwa.com/_files/ugd/d0d088_65a0d59965024bf6bc80c960f161094d.pdf) · ['t goe gezelschap club rules](https://goegezelschaphome.wordpress.com/wp-content/uploads/2019/06/kleurenwiezen.pdf)
- Naming is not universally precise. Whist-Belgique labels its game both Wiezen and Kleurenwiezen, and publishes a hybrid profile with the last card designating a reference trump and players being allowed to propose a suit. Its claim that the two names are identical should not override the clear distinction in IWWA and Whisthub. [Whist-Belgique names](https://www.whist-belgique.be/wiezen-online) · [Whist-Belgique rules](https://www.whist-belgique.be/regles-du-whist)

### Inferences

- Pin the exact rules and scoring before comparing bots. A Belote engine can reuse the card/seat history, hidden-deal sampler interface and benchmark harness. The option to discard while void without trumping means that a discard reveals a void in the led suit, but generally cannot exclude held trumps or higher trumps the way Belote obligations often can. Auction likelihoods and the four-seat payout need their own model.
- Use a four-seat payoff vector and make temporary alliances depend on the contract. A 2-v-2 or 1-v-3 fixed-side search fits many single-bid contracts when partner identity is public, while competing individual Misery bids and other individual scoring need contract-specific utilities. For play-only comparisons fix the auction and contract, then rotate a test agent through seats on duplicate deals. Compare points per deal, contract-type success and decision latency. Test bids separately under a shared play policy. Use shared seeds / identical shuffle or cut protocol across policies, and label whether you measured that or live session cut-only play.

### Gaps

- No universal ruleset applies to all tables. Whisthub and IWWA profiles differ materially in scoring, solo threshold and shuffling.

## Which implementations and strength results are public?

### Takeaway

Whisthub is the clearest documented bot, with production details and a reported speed figure. J10M also offers a playable alternative with documented hidden-hand rollouts. I found no public controlled bot-vs-bot or human-vs-bot strength benchmark for Kleurenwiezen.

### Cited Findings

- Whisthub's creator Sebastiaan Marynissen describes an ownership-probability matrix updated from visible cards and voids, with expected hand-size constraints. Bid-conditioned rank probabilities were refit from a few million actual logged games. Derived probabilities such as void likelihoods feed hand-written if-then rules. Its suit-spread enumeration examines candidate distributions for the current situation and should not be classified as multi-ply Monte Carlo tree search. Hundreds of regression tests tune behavior. He reports the deterministic AI can simulate **over 200 games per second on his PC**. The article gives no hardware details, timing protocol, controlled win rates, or public repository for the production bot or training logs. [Whisthub bot technical writeup](https://www.whisthub.com/blog/how-to-write-an-ai-for-a-card-game) · [Play Colour Whist bots](https://www.whisthub.com/)
- J10M's live Colour Whist game offers a solo table against three simulated players and private tables that fill seats with the same bots. Its published card hint uses the same calculation as those bots, taking only seat-visible information, sampling possible hidden deals, and playing out the rest of the deal for each candidate card. Its bid heuristic estimates tricks from aces, guarded kings, trump length and short suits, with separate approaches to Misery and Piccolo. Its profile plays a queens deal if everyone passes rather than Whisthub's redeal with doubled next score. The provider publishes neither matched strength results nor rollout budget or full source license. [J10M game and AI documentation](https://www.jeuxgratuitsenligne.com/fr/jeux/whist-a-la-couleur)
- Whist-Belgique fills player seats with bots it describes as “expert.” This is a vendor description and no controlled performance figure, algorithm description or licensed source was found. [Whist-Belgique playable bots](https://www.whist-belgique.be/jouer-au-whist-en-ligne)
- The MIT-licensed Go project go_trumpcards has CLI/web Colour Whist and public computer routines. The bidder chooses Alleen with at least two aces and a six-card suit, Samen with an ace and a five-card suit, otherwise passes. It chooses the longest trump suit and uses highest or lowest legal card based on partial-trick side and declarer Misery. It is a simplified profile. Samen calls a hidden partner card rather than asking/accepting, all passes force the dealer to play Samen, and Troel skips bidding. Its play code also accesses internal partner identity, so verify the seat-view boundary before treating this as a fair hidden-information baseline. [Bot source pinned to 26 October 2026](https://github.com/yuta-yoshinaga/go_trumpcards/blob/6cd93dbb186f36fdf153d1bd78fbaf4f9cfdd8c9/internal/domain/ColourWhist.go#L660-L725) · [Game manual](https://github.com/yuta-yoshinaga/go_trumpcards/blob/6cd93dbb186f36fdf153d1bd78fbaf4f9cfdd8c9/docs/manual/cui/colourwhist.md) · [Repository and license](https://github.com/yuta-yoshinaga/go_trumpcards)

- Dewaeq's public Rust kleuren-wiezen repository is a no-trump, follow-suit-only, 13-trick prototype with random-legal and greedy play. Its benchmark runs separately filled four-random and four-greedy tables, defaulting to 800,000 deals each, and prints throughput and total tricks by seat. It has no auction, contract, partnership, or controlled mixed-policy strength measurement. No license file or published benchmark output was found. [Pinned engine](https://github.com/Dewaeq/kleuren-wiezen/blob/5e8a8465600e52fa460c00f2bb6180a78b54e434/src/game.rs) · [Pinned benchmark](https://github.com/Dewaeq/kleuren-wiezen/blob/5e8a8465600e52fa460c00f2bb6180a78b54e434/src/bench.rs)

### Inferences

- Whisthub is a design reference and manual sparring partner. J10M is an independently playable rollout reference under its own rules. A compatible benchmark can compare a transparent history-aware heuristic against constrained, bid-weighted Monte Carlo and exact late solving once remaining hidden deal size is small. Multi-strategy and mixed-human testing matter because the partner changes every deal.
- The published 200+ games/s is throughput, and the few million game logs are an input to the belief model. Neither number measures playing strength.

### Gaps

- No published head-to-head win rate, independent expert-human comparison, downloadable Whisthub policy, or reusable full-fidelity Kleurenwiezen benchmark suite found. Generic Whist, Bridge and other card-game AI results do not establish a rank for these Belgian games.
