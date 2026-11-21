# Belote bot

A bot for **Belgian Belote**, built the same way as the
[Manille bot](https://github.com/frigoow/manil-bot). This repository holds
the rules engine, the world sampler and the first search baselines; the
learned bots come next ([PLAN.md](PLAN.md)).

- [RULES.md](RULES.md): the rules the engine implements, with their sources
  and every house-rule choice as a config flag.
- [cpp/](cpp/): the C++ engine, sampler, ISMCTS, baseline agents and the
  duplicate-deal duel.
- [python/](python/): tests of the `belote_cpp` Python module.

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
| `cpp/agents/duel.h`, `cpp/tools/duel.cpp` | Evaluation harness: duplicate card-play (`play`), bidding-only (`bid`), full-deal (`deal`), mixed-partner (`mixed`), and full 101-point match (`match`) modes with mirrored swaps, paired 95% CIs, and config-version logging |

```bash
build/belote_duel --a rollout:32:ab4 --b rule --deals 1000 --threads 16
build/belote_duel --mode match --a rollout:32:ab4 --b rule --bidder rule --matches 200
```

Baseline ladder (`Config::belgian()`, rule bidder for every seat, A's margin per deal):

| A vs B | Units | Raw points | A ms/card |
|---|---:|---:|---:|
| rule vs random | +3.85 ± 0.24 | +38.9 ± 2.5 | 0.00 |
| ISMCTS-1000 vs rule | +2.64 ± 0.30 | +26.3 ± 3.1 | 3.9 |
| ISMCTS-1000 vs ISMCTS-100 | +1.39 ± 0.29 | +14.0 ± 3.0 | 4.0 |
| ISMCTS-5000 vs ISMCTS-1000 | +0.31 ± 0.17 | +3.1 ± 1.8 | 17.4 |
| rollout:32 vs rule | +2.77 ± 0.30 | +27.4 ± 3.1 | 0.19 |
| rollout:32:ab4 vs rollout:32 | +0.41 ± 0.27 | +4.0 ± 2.7 | 0.51 |
| rollout:32:ab4 vs ISMCTS-1000 | +0.00 ± 0.26 | −0.2 ± 2.7 | 0.53 |

Full 101-point matches (`--mode match`, 200 mirrored pairs = 400 matches, fixed rule bidder): `rollout:32:ab4` beats `rule` **75.25% ± 3.47 pp** (+24.5 ± 2.1 units/match).

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
