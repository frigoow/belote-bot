# Belote bot

A bot for **Belgian Belote**, built the same way as the
[Manille bot](https://github.com/frigoow/manil-bot). This repository holds
the rules first; the bots come next.

- [RULES.md](RULES.md): the rules the engine implements, with their sources
  and every house-rule choice as a config flag.
- [cpp/](cpp/): the C++ rules engine.

## The rules engine

`cpp/belote/` is the only implementation of the rules:

| File | What it covers |
|---|---|
| `card.h` | Card encoding (suit × 8 + rank, ranks 7 8 9 10 J Q K A), point and strength tables |
| `rules.h` | `Config`: the flags of RULES.md; `Config::belgian()` is the house ruleset |
| `toon.h` | Toon (sequences and carrés): finding the best split, comparing teams |
| `deal.h` | One deal: deal and blik, two bidding rounds, eight tricks, belot, scoring. `legalMask()` is the legal-move rule on its own, for samplers and play-outs |
| `match.h` | The race to 101: recorded units, points held over (*op den boek*), dealer rotation and redeals |

## Tests

```bash
cpp/run_tests.sh                     # plain g++, no CMake needed
# or
cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
```

- `test_rules` has one case per rule in RULES.md.
- `test_random` checks invariants over 2,000 random matches. It also
  cross-checks the engine against an independent port of Tanguy Pemeja's
  classic-French engine ([tpemeja/open_spiel](https://github.com/tpemeja/open_spiel/tree/belote-cpp-rework),
  Apache-2.0): with `Config::tanguyCompat()` every legal-move set and every
  final score must agree, over 50,000 random deals.

A random deal takes about 1.4 µs (44 ns per card) on a desktop CPU.
