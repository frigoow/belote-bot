# belote-bot

Strong bots for two Belote games, built the same way as the
[Manille bot](https://github.com/frigoow/manil-bot), and the Bulgarian Belot
app. The game-independent parts (app building blocks, network kernels, and
more to come) live in [cardbot-framework](https://github.com/frigoow/cardbot-framework),
the git submodule `cardbot-framework/`.

| Game | Rules | Engine and bots | Training | More |
|---|---|---|---|---|
| **Belgian Belote** | [docs/RULES_BELGIAN.md](docs/RULES_BELGIAN.md) | `cpp/belote`, `cpp/search`, `cpp/agents` | `python/belote` | [docs/PLAN_BELGIAN.md](docs/PLAN_BELGIAN.md); the rest of this README |
| **Bulgarian Belot** (bridge-belot, Kostov's rules) | [docs/RULES_BULGARIAN.md](docs/RULES_BULGARIAN.md) | `cpp/bg` | `python/bg` | the app in [`app/`](app/README.md); the arena against Kostov's bots in `tools/kostov/` |

| Path | What |
|---|---|
| `cpp/` | The C++ engines, search and bots of both games, their tests and tools (`belote_duel`, `bg_duel`, `bg_agent`), and the Python modules `belote_cpp` and `bg_cpp` |
| `python/` | Training, evaluation and export (`python/belote`, `python/bg`) and the Python tests |
| `app/` | The Bulgarian Belot app for Android, Windows, macOS and Linux ([app/README.md](app/README.md)); releases are this repository's `vX.Y.Z` tags |
| `tools/kostov/` | C#: the oracle that checks `cpp/bg` card for card against Kostov's BelotGameEngine (`oracle/`), and the arena that plays our bots against his inside his engine (`bridge/`). Both expect his engine checked out next to this repository |
| `docs/` | The rules of both games and the Belgian plan (private reports and research notes are kept outside git) |
| `checkpoints/`, `runs/` | Trained networks and training logs (not in git) |
| `cardbot-framework/` | The shared framework (submodule) |

The Bulgarian bots: the unmasked neural bidder with doubles, card play by
world sampling with belief, fast-policy and Q-GRU nets per contract kind,
candidate racing and an exact endgame. Inside Kostov's engine they won
54.35% ± 1.74 pp of 2000 games against his Master bot
(`runs/bg_bridge_e2e_bid_dbl_5mspec_1000p.log`).

# Belgian Belote

The rules engine, the world sampler, the search baselines and the learned
bots ([docs/PLAN_BELGIAN.md](docs/PLAN_BELGIAN.md)):

- [docs/RULES_BELGIAN.md](docs/RULES_BELGIAN.md): the rules the engine implements, with their sources
  and every house-rule choice as a config flag.
- [cpp/](cpp/): the C++ engine, sampler, ISMCTS, baseline agents and the
  duplicate-deal duel.
- [python/](python/): training, and tests of the `belote_cpp` Python module.

## The rules engine

`cpp/belote/` is the only implementation of the rules:

| File | What it covers |
|---|---|
| `card.h` | Card encoding (suit × 8 + rank, ranks 7 8 9 10 J Q K A), point and strength tables |
| `rules.h` | `Config`: the flags of RULES.md; `Config::belgian()` is the belote-rebelote.be ruleset (the default), `Config::wikibooks()` the earlier [WB] + house one |
| `toon.h` | Toon (sequences and carrés): finding the best split, comparing teams |
| `deal.h` | One deal: deal and blik, two bidding rounds, eight tricks, belot, scoring. `legalMask()` is the legal-move rule on its own, for samplers and play-outs |
| `match.h` | The race to 101: recorded units (`recordDeal`), points held over (*op den boek*), dealer rotation and redeals |
| `observation.h` | What one player knows (`observe()`), and the per-card facts that follow from the play: voids, no (higher) trump, the blik, shown toon, announced carrés, belot (`inferConstraints()`) |
| `sampler.h` | `WorldSampler`: the hidden cards dealt exactly uniformly over every deal consistent with an observation (ported from the Manille bot's counting sampler, plus announced sequence lengths), and `replay()` of the deal from a world |

## Search and baselines

| Path | What it is |
|---|---|
| `cpp/third_party/ismcsolver` | The ISMCTS library the Manille bot uses (Steven Franzen, MIT) |
| `cpp/search/belote_game.h` | Card play as an ismcsolver game: determinizations from `WorldSampler`, result = the deal's margin in recorded units |
| `cpp/search/ismcts.h` | `ismctsSearch()`: SO-ISMCTS, sequential or root-parallel |
| `cpp/search/endgame.h` | `solveEndgame()`: contract-aware exact double-dummy alpha-beta solver with Zobrist TT (`0.18 ms/solve` at 16 cards left) |
| `cpp/search/rollout.h` | `rolloutSearch()`: shared-world heuristic rollouts with optional exact AB endgame handoff |
| `cpp/agents/agent.h` | `random`, `rule`, `ismcts:<iterations>`, and `rollout:<worlds>[:ab<trick>]` agents |
| `cpp/agents/rule_agent.h` | Rule agent v2 (Manille structure): honest human-style policy `rule2`, its play-out search `ruleplay:<worlds>` (rule card overruled only on a significant paired gain; worlds weighted by the auction and by the others' plays), and the simulation bidder `simbid<worlds>` |
| `cpp/tools/rule_review.cpp` | `belote_rule_review`: plays rule2 deals, scores every legal card, and reports where the rule card loses value by reason, role, seat and trick |
| `cpp/agents/duel.h`, `cpp/tools/duel.cpp` | Evaluation harness: duplicate card-play (`play`), bidding-only (`bid`), full-deal (`deal`), mixed-partner (`mixed`), and full 101-point match (`match`) modes with mirrored swaps, paired 95% CIs, and config-version logging |

```bash
build/belote_duel --a rollout:32:ab4 --b rule --deals 1000 --threads 16
build/belote_duel --mode match --a rollout:32:ab4 --b rule --bidder rule --matches 200
```

Baseline ladder (`Config::belgian()`, rule bidder for every seat, A's margin per deal):

| A vs B | Decks × 2 | Units | Raw points | A ms/card |
|---|---:|---:|---:|---:|
| rule vs random | 911 | +3.85 ± 0.24 | +38.9 ± 2.5 | 0.00 |
| ISMCTS-1000 vs rule | 911 | +2.64 ± 0.30 | +26.3 ± 3.1 | 3.9 |
| ISMCTS-1000 vs ISMCTS-100 | 911 | +1.39 ± 0.29 | +14.0 ± 3.0 | 4.0 |
| ISMCTS-5000 vs ISMCTS-1000 | 911 | +0.31 ± 0.17 | +3.1 ± 1.8 | 17.4 |
| rollout:32 vs rule | 9,032 | +2.55 ± 0.09 | +25.2 ± 0.9 | 0.16 |
| rollout:32:ab4 vs rollout:32 | 2,246 | +0.40 ± 0.17 | +4.0 ± 1.8 | 0.51 |
| rollout:32 vs ISMCTS-1000 | 1,797 | −0.24 ± 0.19 | −2.4 ± 2.0 | 0.19 |
| rollout:32:ab4 vs ISMCTS-1000 | 1,797 | +0.10 ± 0.20 | +0.9 ± 2.0 | 0.56 |
| fast (1.8M Q-GRU) vs rule | 911 | +1.56 ± 0.28 | +15.8 ± 2.9 | <0.01 |
| hybrid:64:ab4 vs ISMCTS-1000 | 911 | **+0.48 ± 0.29** | **+4.6 ± 3.0** | 12.6 |
| hybrid:64:ab4 vs ISMCTS-5000 | 450 | −0.05 ± 0.41 | −0.4 ± 4.1 | 12.4 |
| fast (ExIt1) vs fast (1.8M) | 1,797 | +0.30 ± 0.19 | +2.9 ± 1.9 | <0.01 |
| hybrid (ExIt1):192:ab4:rab5:race:z2.0 vs pure Q-GRU (ExIt1) | 400 | **+1.92 ± 0.48** | — | 17.5 |
| hybrid (ExIt1):192:ab4:rab5:race:z2.0 vs ISMCTS-5000 | 911 | **+0.72 ± 0.26** | **+7.1 ± 2.7** | 16.2 |
| hybrid (ExIt1):256:ab4:rab5:race:z2.0 vs ruleplay:128:ab4 | 898 | **+0.29 ± 0.25** | **+2.8 ± 2.6** | 20.6 |
| hybrid (ExIt1):256:ab4:rab5:race:z2.0:hauct vs ISMCTS-5000 | 911 | **+1.04 ± 0.28** | **+10.3 ± 2.8** | 24.8 |

*(Neural checkpoints in `checkpoints/` are gitignored and local; reproduce via `python -m belote.train_dmc`, `python -m belote.train_exit`, and `python -m belote.train_bidding`.)*

Full 101-point matches (`--mode match`, mirrored pairs, `Config::belgian()`):
- `rollout:32:ab4` vs `rule` (200 pairs, fixed rule bidder): **75.25% ± 3.47 pp** (+24.5 ± 2.1 units/match).
- `BidNet v4` vs `RuleBidder` (2,000 pairs = 4,000 matches, `fast:q_gru_exit1` play): **52.05% ± 1.40 pp** pure, **51.75% ± 1.21 pp** mixed-partner (`<0.15 ms/bid`); with Stage 3 search (`:s3=120:m=0.04`, 400 pairs): **54.00% ± 3.10 pp** (+2.96 ± 2.41 units/match).
- Full M4 stack (`bid:bidding_v4+hybrid:q_gru_exit1:192:ab4:rab6:race:z2.0:me`): beats `rule` **82.0% ± 5.7 pp** (100 pairs), `rule2` **73.0% ± 6.0 pp** (100 pairs), `ismcts:1000` **64.3% ± 5.0 pp** (150 pairs), `ismcts:5000` **59.3% ± 5.3 pp** (150 pairs), and mixed-partner `M4 + rule` vs `ismcts:1000 + rule` **54.5% ± 5.8 pp** (100 pairs).
- Full M4 stack (`bid:bidding_v4+hybrid:q_gru_exit1:256:ab4:rab5:race:z2.0:me:hauct`) vs `ruleplay:128:ab4:simbid512`: **52.75% ± 3.69 pp** (200 pairs, seed 2; **53.00% ± 4.47 pp** over 150 pairs with `192:ab4:rab5:race:z2.0:me`, seed 1).

### Rule agent

`rule2` uses the public auction, shown cards and its own hand. It favors a discard that sets up a valuable ruff, supports the taker on an opening trump lead, and weighs known declarations and trump length when bidding. From trick five, if at most 48 layouts remain before the final declaration checks, it compares the possible endings. It changes its initial play only when the alternative scores at least as well under best play in every compatible layout and better in at least one. In matches, the comparison accounts for the target, winning margin and hanging pot. The value of a continuing match is a heuristic based on the score lead. The [manual table review](docs/reports/manual-table-review/strategy.md) records the original decisions and paired replays.

Use `rule2:nocount` for the same bidding and table heuristics with the exhaustive check disabled. The full search agent is `ruleplay:128:ab4:simbid512`. Its early rollouts evaluate the final two tricks exactly, and its late searches and simulated bids use the current scoreboard when available. The match harness supplies it automatically. In a Python game loop, call `agent.set_match_state(match)` before each deal.

Head-to-head results against a frozen copy of the previous rule agent use fresh shuffles, swapped teams and paired 95% confidence intervals. The deal counts below exclude decks on which both tables passed.

| Revised agent | Comparison | Mirrored pairs | Seed | Result |
|---|---|---:|---:|---:|
| `rule2` | Full matches to 101 | 3,500 | 88102 | **57.53% ± 0.98 pp** wins |
| `rule2` | Own bidding and card play | 32,335 | 88101 | **+0.714 ± 0.044** units/deal |
| `rule2` | Same auction at both tables | 31,528 | 88103 | +0.567 ± 0.030 units/deal |
| `rule2` | Same auction and a basic `rule` partner | 19,816 | 88108 | +0.305 ± 0.027 units/deal |
| `rule2:nocount` | Own bidding and card play | 46,162 | 88104 | +0.277 ± 0.035 units/deal |
| `rule2:nocount` | Full matches to 101 | 4,000 | 88111 | 52.79% ± 0.87 pp wins |
| `rule2` | Bidding only, identical card player | 55,381 | 88112 | +0.165 ± 0.027 units/deal |
| `ruleplay:32` | Same auction and 32 worlds per player | 6,482 | 88106 | +0.140 ± 0.093 units/deal |
| `ruleplay:128:ab4` | Same auction and 128 worlds per player | 1,205 | 88107 | +0.220 ± 0.215 units/deal |
| `ruleplay:128:ab4:simbid512` | Full matches to 101 | 1,200 | 88110 | 49.96% ± 1.66 pp wins |

The revised `rule2` averaged 0.03 ms per card in the full-match comparison. For the full search agent, the initial 240-pair match run was 52.71% ± 3.76 pp (seed 88109), and the larger run above was consistent with even play. The following cross-agent benchmarks predate this revision.

| A vs B (`Config::belgian()`) | Mode | Decks × 2 | Units | Raw points |
|---|---|---:|---:|---:|
| rule2 vs rule | play | 3,600 | +1.14 ± 0.14 | +11.5 ± 1.4 |
| ruleplay:32:noab vs rollout:32 ¹ | play | 1,357 | +0.45 ± 0.23 | +4.7 ± 2.4 |
| ruleplay:128:ab4 vs ISMCTS-1000 | play | 2,245 | **+0.78 ± 0.16** | +7.7 ± 1.7 |
| ruleplay:128:ab4 vs ISMCTS-5000 ¹ | play | 704 | +0.35 ± 0.29 | +3.5 ± 2.9 |
| ruleplay:128:ab4 vs hybrid:64:ab4 (1.8M) ¹ | play | 899 | +0.63 ± 0.29 | +6.4 ± 2.9 |
| rule2:simbid512 vs rule2 | bid | 9,597 | +0.37 ± 0.09 | +3.9 ± 0.9 |
| ruleplay:128:ab4:simbid512 vs ISMCTS-1000 | deal | 2,402 | **+0.88 ± 0.22** | +8.9 ± 2.2 |

¹ Measured with an earlier version of the agent: before the last two policy switches (`effrank`, `partnervoid`, together +0.03 ± 0.02 in rule2 duels), and for the `noab` row also before the world weights.

Search-layer findings (`ruleplay:64:ab4` baseline, about 2,200–2,700 decks each): the rule prior with overrule test is worth +0.32 over plain play-out argmax; 128 worlds +0.25 ± 0.15; auction plus play-likelihood world weights +0.25 ± 0.14 and +0.22 ± 0.15 (two seeds); overrule at z = 2 −0.45 ± 0.16 (z between 0 and 1 is flat); AB from trick 3 +0.21 ± 0.16, from trick 2 no gain at 25× the time. The ms/card of the search agents is about 2–8 ms on 16 threads, comparable to ISMCTS-1000 run alongside it.

## Python

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/cmake -B build -S cpp -DBELOTE_PYTHON=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python \
    -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')
.venv/bin/cmake --build build -j
.venv/bin/python -m pytest python
```

`belote_cpp` exposes the engine (`Config`, `Deal`, `Match`, `record_deal`),
`observe` / `WorldSampler`, `ismcts_search`, the agents and `run_duel`. Cards
are ints 0..31 and card sets 32-bit masks, as in C++.

## Tests

```bash
cpp/run_tests.sh                     # plain g++, no CMake needed
# or
cmake -B build -S cpp && cmake --build build -j
ctest --test-dir build --output-on-failure
```

- `test_rules` has one case per rule in RULES.md.
- `test_sampler` checks the sampler against brute force: on 600 late
  positions under four rule profiles, its worlds are exactly the deals that
  replay to the same public history; sampling is uniform; the real deal is
  never excluded.
- `test_search` checks that every ISMCTS determinization is consistent with
  the observer's knowledge, and runs a small ladder.
- `test_random` checks invariants over 2,000 random matches. It also
  cross-checks the engine against an independent port of Tanguy Pemeja's
  classic-French engine ([tpemeja/open_spiel](https://github.com/tpemeja/open_spiel/tree/belote-cpp-rework),
  Apache-2.0): with `Config::tanguyCompat()` every legal-move set and every
  final score must agree, over 50,000 random deals.

A random deal takes about 1.4 µs (44 ns per card) on a desktop CPU.
