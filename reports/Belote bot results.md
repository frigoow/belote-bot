# Belote bot benchmark results

Research snapshot as of 21 November 2026. These are author-reported measurements. The matches were not rerun. Win rates and score margins use the rules and opponents of each study.

## Bulgarian Bridge-Belot

September 30 `CreateMaster` hybrid from the MIT-licensed BelotGameEngine. Sample sizes count mirrored games. The ± figures are one standard error across mirrored pairs, in percentage points.

| Comparison | Win rate | Games |
| --- | ---: | ---: |
| Master vs 100 ms ISMCTS | **63.60% ±1.30 pp** | 1,000 |
| Master vs a port of Belot 2.06 | **79.50% ±0.83 pp** | 2,000 |
| Master vs SmartPlayer | **93.65% ±0.53 pp** | 2,000 |
| Natural-bidding Master vs older relay-bidding Master, each paired with Belot 2.06 | **64.78% ±0.64 pp** | 4,000 |

Master took **13.1 ms per card on average**, with a 95th percentile of **30.8 ms**, on the author's desktop. All four comparisons used bot opponents. [Validation results](https://github.com/NikolayIT/BelotGameEngine/blob/master/HUMAN_PLAY.md#final-validation-september-30-fresh-seeds-otherwise-idle-machine) · [Mixed-partner test](https://github.com/NikolayIT/BelotGameEngine/blob/master/HUMAN_PLAY.md#what-the-relay-costs-a-human-partner) · [Source and license](https://github.com/NikolayIT/BelotGameEngine)

## Belgian Belote

The Belgian starting point is classic upcard Belote with two bidding rounds and a redeal after eight passes. The provisional scoring profile uses a target of 101 with a two-point lead and carries the taker's points on an 81–81 tie. The Belgian site offers playable bots. I found no public strength benchmark or bot source for that profile. [Belgian rules](https://www.belote-rebelote.be/regles-belote) · [Scoring](https://www.belote-rebelote.be/compter-points-belote) · [Playable bots](https://www.belote-rebelote.be/)

### Closest benchmark on classic upcard Belote

Tanguy Pemeja's French classic simulator measures raw points per deal. The figures in square brackets are 95% confidence intervals for the score margin. A dash indicates that the sample size was unavailable in the research notes.

| Comparison | Margin in points per deal | Deals | Reported timing |
| --- | ---: | ---: | --- |
| 32-world Monte Carlo rollout vs heuristic play, same bidder | **+38.68** [35.18, 42.19] | — | 114.2 ms/deal |
| 4-world Monte Carlo rollout vs heuristic play, same bidder | **+26.03** [22.52, 29.54] | — | 14.5 ms/deal |
| PlayNet vs Monte Carlo card play, same bidder | −5.17 [−6.59, −3.74] | — | — |
| BidNet vs rule bidder, both using PlayNet | +5.59 [4.18, 6.99] | 4,695 | — |
| BidNet vs rule bidder, both using Monte Carlo card play | +5.50 [4.05, 6.96] | 6,000 | — |
| BidNet + PlayNet vs rule bidder + Monte Carlo card play | −0.63 [−2.06, 0.81] | 6,000 | Reported 148× speedup |

The final row changes both bidding and card play. The fixed-bidding comparison measures PlayNet's card-play deficit directly. The public simulator provides a starting point for benchmarking under Belgian scoring. [Experiment results and protocol](https://engineering.kaseovo.com/projects/belote-ai/) · [Simulator source](https://github.com/tpemeja/open_spiel/tree/belote-cpp-rework)

### UCLouvain upcard variant

Alexandre Serrao's simulator includes no-trump, all-trump, and forced trump selection after two rounds of passes. The thesis's headline results give no match counts or uncertainty intervals.

| Comparison | Win rate | Mean thinking time |
| --- | ---: | --- |
| Outcome-sampling MCCFR vs ISMCTS | **84.0%** | 40 s/move vs 3 s/move |
| External-sampling MCCFR vs ISMCTS | **81.5%** | 40 s/move vs 3 s/move |
| Either MCCFR variant vs random | 99.0% | 40 s/move for MCCFR |
| ISMCTS vs random | 86.8% | 3 s/move for ISMCTS |
| Deep CFR vs random across tested settings | About 45–60% | — |

The MCCFR comparisons used a larger per-move time budget than ISMCTS. [Thesis results, sections 5–6](https://perso.uclouvain.be/eric.piette/master/2025%20-%20Performance%20Evaluation%20of%20Counterfactual%20Regret%20Minimization%20and%20Monte%20Carlo%20Tree%20Search%20Variants%20-%20A%20Case%20Study%20on%20Belote.pdf)

## Other variants with published measurements

| Variant | Comparison | Result |
| --- | --- | --- |
| French Contrée | Colver Bid V6 vs V5, both using DouDou50 card play | 55.8% wins |
| French Contrée | Colver Bid V6 vs V5, both using information-set double-dummy card play | 57.3% wins and +181 points per match |
| Moldovan Belot | BelotAI with a stranger partner, excluding hands with a platform replacement bot | +0.00 ±0.73 game points per hand over 1,412 hands |

Colver's results are internal match comparisons. The Moldovan result comes from observational online play. [Colver bidder results](https://github.com/Avo-k/colver/blob/master/docs/hf/bid-v6/README.md) · [BelotAI live results](https://github.com/MrUnBaiat/BelotAI#what-live-play-has-measured)

For the implementation comparison and ruleset details, see the [full research report](State%20of%20Belote%20bots.md).
