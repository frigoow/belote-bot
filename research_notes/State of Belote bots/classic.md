# Classic Belote bots and academic comparison

## What has been demonstrated for classic Belote?

### Takeaway

Tanguy Pemeja's 2026 first-party project provides the clearest public quantitative classic benchmark I found. Constrained Monte Carlo rollouts over a hand-written policy produced the main card-play gain. Learned bidding improved on heuristic bidding, while distillation reduced playing cost but ceded some strength with bidding held fixed.

### Cited Findings

- Scope is classic French Belote à la retourne. Five cards per player and an upcard are visible to the appropriate seats in bidding. The first round offers the upcard's suit, the second other suits. Eight passes produce a redeal. Test unit is a single completed deal and return is the signed final-score difference. The normal trick pool is 162, or 252 if a side wins every trick. On declarer failure the defenders collect all trick points. The engine only has automatic optional belote/rebelote as a meld, default off in the currently accessible fork. The project summary omits the experiment-wide flag setting. [Pemeja benchmark and rule choices](https://engineering.kaseovo.com/projects/belote-ai/) · [Python simulator source](https://raw.githubusercontent.com/tpemeja/open_spiel/belote-cpp-rework/open_spiel/python/games/belote.py)
- Paired tests use an identical seeded shuffle twice with the dealer rotated by one seat, averaging margins and reporting 95% CIs. The reported pairs test yields standard deviation 63.0 versus 87.0 expected from two independent orientations or 123.0 from one. Bidding guard verifies tests meant to isolate card play. [Project measurement section](https://engineering.kaseovo.com/projects/belote-ai/)
- The card player compares every legal card in the same sampled legal deals, completing each with its hand-written `win_or_dump` strategy. This is a one-step rollout algorithm with neither tree nor network. For 32 worlds and identical bidding policies, +38.68 pts/deal [35.18, 42.19] over `win_or_dump`, 114.2 ms/deal. With 4 worlds, +26.03 [22.52, 29.54] at 14.5 ms/deal. Rung-specific sample size for this comparison is absent from the project page. [Project Monte Carlo section](https://engineering.kaseovo.com/projects/belote-ai/)
- PlayNet learned to predict each candidate card's advantage over the average legal card from 846,086 Monte Carlo labels over 25,000 deals. With the card engine tested head-to-head at identical bidding, PlayNet trailed Monte Carlo by 5.17 pts/deal [−6.59, −3.74]. BidNet scored bids using the same 24 compatible sampled deals across candidates with PlayNet card play and beat the strongest tested bid rule by +5.59 [4.18, 6.99] with PlayNet over 4,695 deals, or +5.50 [4.05, 6.96] with Monte Carlo card play over 6,000 deals. [Project PlayNet and BidNet sections](https://engineering.kaseovo.com/projects/belote-ai/)
- Search-free BidNet+PlayNet came within −0.63 pts/deal [−2.06, +0.81] of the Monte Carlo agent in a 6,000-deal matchup and the project reported a 148× speedup. The opposing Monte Carlo agent in that comparison uses rule bidding (the page calls the BidNet+Monte Carlo variant the strongest agent, +5.50 over Monte Carlo with rule bidding). “No detected difference” applies to the published whole-agent matchup, while equal-bidding card play has a measured deficit. [Project comparison](https://engineering.kaseovo.com/projects/belote-ai/)
- Tested local Gemma 4 12B had worse point-loss grading than random legal choices on 150 selected decisions (9.28 vs 6.13 points of teacher-estimated regret per decision), and trailed Monte Carlo by 35.92 pts/deal [−52.54, −19.29] over 24 deals. The grader is a rollout-derived comparison, and the actual match compares to Monte Carlo only. [Project LLM experiment](https://engineering.kaseovo.com/projects/belote-ai/)

### Inferences

- Strong default choice for classic development is a decent heuristic, sampling hands consistent with card-play obligations, and comparing candidates on the same deals. This experiment motivates testing search complexity against that baseline, and preserving auction/play isolation even if the shipped agent integrates both.
- The 2026 ladder is better evidence of within-simulator improvement than comparative evidence against other implementations. None of its numbers establishes a human skill level.

### Gaps

- The methods uniformly sample the deals compatible with play and intentionally omit action-likelihood inference about bids/discards and deliberate partner signalling. A known-full-deal oracle ceiling was also unmeasured. The project describes a human-facing demo and tracking but gives no skilled-human tournament rating. [Project gaps](https://engineering.kaseovo.com/projects/belote-ai/)
- The linked [agent repository](https://github.com/tpemeja/belote-ai) returned 404 in both browser fetch and unauthenticated GitHub REST API on Nov 21, 2026. The engine fork is independently available, including [Python](https://raw.githubusercontent.com/tpemeja/open_spiel/belote-cpp-rework/open_spiel/python/games/belote.py) and [C++](https://raw.githubusercontent.com/tpemeja/open_spiel/belote-cpp-rework/open_spiel/games/belote/belote.cc) sources, so it is reproducible at the engine level. Belote remains an [open upstream request](https://github.com/google-deepmind/open_spiel/issues/1577). The upstream change that actually merged was the [Python ISMCTS seeded-resampling fix](https://github.com/google-deepmind/open_spiel/pull/1584).

## What does the UCLouvain thesis establish about ISMCTS and CFR?

### Takeaway

Alexandre Serrao's 2025 master's thesis compares search and regret minimization on its own upcard rule set. Online MCCFR variants beat the thesis ISMCTS in round-robin tests using much more per-move compute. This was distinct from professional or cross-project benchmarking.

### Cited Findings

- The chapter 2 variant deals five cards and a public upcard before selecting a trump. It allows all-trump and no-trump in the second auction round and forces first player to pick if everyone passes, unlike Pemeja's classic engine. The C++ OpenSpiel game represents one round internally, but the author says match testing repeats rounds until a score threshold (§4.2.1). The summary refers to 1,000 points while the rules description says above 100, so omit the threshold when quoting experimental win rates. [Thesis full PDF hosted by advisor](https://perso.uclouvain.be/eric.piette/master/2025%20-%20Performance%20Evaluation%20of%20Counterfactual%20Regret%20Minimization%20and%20Monte%20Carlo%20Tree%20Search%20Variants%20-%20A%20Case%20Study%20on%20Belote.pdf) · [Institutional metadata](https://thesis.dial.uclouvain.be/entities/masterthesis/bea00f0f-0660-44c6-bfeb-84a9b9bd2b61)
- Ch. 6 Tables 6.1/6.2 report outcome-sampling online MCCFR at 84% and external-sampling online MCCFR at 81.5% against the thesis ISMCTS. Both show 99% against random and 50% head-to-head. ISMCTS shows 86.8% vs random. Average thinking time is 40 s/move for each MCCFR against 3 s/move for ISMCTS. Interpret outcomes by column agent, as verified by the accompanying prose. Ch. 5 adaptations include determinizations and online search with caching. [Thesis](https://perso.uclouvain.be/eric.piette/master/2025%20-%20Performance%20Evaluation%20of%20Counterfactual%20Regret%20Minimization%20and%20Monte%20Carlo%20Tree%20Search%20Variants%20-%20A%20Case%20Study%20on%20Belote.pdf)
- DeepCFR yielded about 45–60% against random across settings, using roughly 8 hours to 5 days of training for most agents. The thesis reports too few traversals for the game size and estimates one bidding state in its replay buffer per 2,000 other states. It attributes some of the bottleneck to chance branching and rare training coverage early. [Thesis, §5.4](https://perso.uclouvain.be/eric.piette/master/2025%20-%20Performance%20Evaluation%20of%20Counterfactual%20Regret%20Minimization%20and%20Monte%20Carlo%20Tree%20Search%20Variants%20-%20A%20Case%20Study%20on%20Belote.pdf)

### Inferences

- These results support online MCCFR as a research contender at a higher compute budget. They cannot establish equal-compute superiority over ISMCTS, or rank MCCFR against Pemeja's heuristic-guided rollout.

### Gaps

- Headline round-robin presents no test count, confidence interval, or paired-deal description. No public code link is present in the thesis. Its public main PDF is on the advisor's site despite closed access at the university repository.
- No direct MCCFR versus Pemeja search match under the same rules and time budget found.

## Which other classic implementations are worth knowing?

### Takeaway

Several projects provide runnable algorithm examples, while open strength evidence is limited.

### Cited Findings

- Mahdi Ben Amar's Python implementation describes classic two-round bidding, void tracking, random determinization, 30 worlds with 150 MCTS simulations each by default, and vote aggregation. Its documentation supplies rules and parameters plus terminal and Tkinter interfaces. [Public repository](https://github.com/MahdiBenAmar21/projet-belote) · [Report](https://github.com/MahdiBenAmar21/projet-belote/blob/main/rapport.md)
- adUst0's C++20/SFML work in progress advertises MCTS for card play and hill climbing for bid heuristics. [Repository](https://github.com/adUst0/Belote)
- Commercial examples prominently offer solo AI difficulty without measurable public rankings. Summit Games claims three difficulties and no access to other players' hands. Cleveroad's app says its strongest level counts cards, tracks voids and reads its partner. Their own stores support existence and features, not relative strength. [Summit Games listing](https://play.google.com/store/apps/details?id=com.summitgames.belote) · [Cleveroad listing](https://play.google.com/store/apps/details?id=com.cleveroad.belote)

### Inferences

- Public playable bots and publicly ranked bots are different sets. An independent benchmark would need fixed rules plus head-to-head against external projects or human teams.

### Gaps

- No match-level accuracy, rating or timing comparisons for these project examples against stronger external opponents were found.
