# Transferable bot evidence for Wiezen

## Which benchmark is closest?

### Takeaway

Leiden University's Rikken thesis provides the closest quantitative comparison found for a four-player, 52-card game in the same family. It isolates ordinary card play, and simplifications limit what the figures establish for Wiezen.

### Cited Findings

- Mika Bakker (2024), Table 1. Monte Carlo declarer and mate made their contract in **62.1% of 10,000 deals against greedy opponents**, and **80.2% of 1,000 deals against random opponents**. Four random players yielded 47.9% declarer-side contract success over 1,000,000 deals. A Monte Carlo declarer with a random mate against random opponents made 67.3% over 1,000 deals. The Monte Carlo bot samples opponent cards consistent with suit voids and plays out with random legal moves, using 8 times squared hand size as its simulation budget. [Leiden thesis, sections 4.3 and 5.2](https://theses.liacs.nl/pdf/2023-2024-BakkerMMichal.pdf)
- The benchmark removes bidding, fixes the eight-trick contract, assigns the mate, allows the evaluator to know that mate before play reveals it, and always gives the declarer the opening lead. It uses contract completion as the only result. [Leiden thesis, sections 2 and 4.3.1](https://theses.liacs.nl/pdf/2023-2024-BakkerMMichal.pdf)

### Inferences

- The result motivates testing a void-constrained rollout baseline against a heuristic on fixed contracts before testing a full auction. Annotate search by contract and sample the partnership when the contract leaves it hidden.

### Gaps

- No public Rikken thesis source repository identified. These results do not measure Wiezen or Kleurenwiezen strength.

## What other 52-card results and code transfer?

### Takeaway

The DTCard paper supplies a public no-auction simulator and weak or middling neural results according to opponent. Bridge supplies an established full-information solver whose partnership assumptions matter.

### Cited Findings

- Decision Transformer + state-evolution inference won **47.1% ±2.1 against random**, **30.6% ±1.3 against a rule-based bot**, and **33.7% ±2.5 against DQN** in the 2026 DTCard authors' Whist environment. The paper reports 10,000 independent games per evaluation and 95% CIs across 10 seeds. It scores individual trick counts in a 13-trick deal, sets trump from the last card with no auction, and has no partnership modeling. This is other Whist evidence at a different objective, even compared with fixed-team classic Whist. [Paper Table 1](https://www.mdpi.com/2076-3417/16/7/3117) · [Environment code](https://github.com/bugrakaandem/dtcard/blob/main/games/whist.py)
- Bridge's Apache-2.0 DDS library gives optimal fully exposed card-play outcomes, with bindings in C++, Python, .NET and Java, for two alternating fixed partnerships. It returns trick values for a current deal/trick and nominated trump. It is a reference for search optimizations and compatible fixed-seat situations. Generic Wiezen 2v2 can have adjacent partners, while solo contracts and multi-misère require different objectives. Merely reseating four cards to fit DDS partner parity changes the play order. [DDS project](https://github.com/dds-bridge/dds) · [DDS interface](https://github.com/dds-bridge/dds/blob/develop/doc/dll-description.md) · [Wiezen rules, player observations](https://www.pagat.com/boston/wiezen.html)
- For sampled hidden deals, Cazenave and Ventos's bridge alpha-mu search describes strategy fusion, the risk of assigning incompatible follow-up strategies across the sampled worlds when the player cannot tell which world is real. Their algorithm improves PIMC under their own Bridge tests. [Original paper](https://tristancazenave.com/papers/AlphaMu.pdf)

### Inferences

- Encode a player-to-contract-role mapping alongside the card history and score each sampled outcome by the actual contract payout. Add a late exact solver under the same rule semantics, rather than making a fixed N/S-versus-E/W bridge solver the interface for every contract.

### Gaps

- A direct modern comparison between a Whisthub policy, an open implementation and human tournament-level players wasn't verified. General 52-card evidence establishes methods to test, not a ranking in Wiezen.

## How does the documented Whisthub AI use possible deals?

### Takeaway

Whisthub computes weighted probabilities over current card spreads and then applies hand-written thresholds. Its developer contrasts this with multi-move search.

### Cited Findings

- The developer explicitly evaluates current unseen-suit counts distributed across the other players, often abstracting away individual ranks in the spread. The author likens this to shallow Monte Carlo, yet says it examines only possible current situations, with no play sequence rollout. Bids shape per-card probabilities, with calibration from a few million game logs. The service's deterministic decision tree handles the derived probabilities. [Whisthub AI article, assumptions matrix through decision tree](https://www.whisthub.com/en/blog/how-to-write-an-ai-for-a-card-game)
- The developer reports more than 200 self-play games per second on their PC and hundreds of scenario regression cases. These are engineering results and do not measure relative playing strength. The documented algorithm examples cover Colour Whist, and the service also advertises solo ordinary Wiezen against AI. [AI article](https://www.whisthub.com/en/blog/how-to-write-an-ai-for-a-card-game) · [Ordinary Wiezen lobby](https://www.whisthub.com/nl/wiezen)

### Inferences

- A weighted hidden-hand model and contract-aware heuristics give a plausible fast baseline for either ruleset. Check how much full-hand rollouts improve it under a fixed latency budget.

### Gaps

- No standalone open-source Wiezen or Kleurenwiezen engine with a documented license and benchmark surfaced in the transfer search. The Whisthub blog is public and provides a testing example, while source and benchmark figures were not verified.
