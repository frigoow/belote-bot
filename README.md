# Belote Bot

[![Latest Release](https://img.shields.io/github/v/release/frigoow/belote-bot?label=release)](https://github.com/frigoow/belote-bot/releases/latest)
[![Platform](https://img.shields.io/badge/platforms-Android%20%7C%20Windows%20%7C%20macOS%20%7C%20Linux-blue)](https://github.com/frigoow/belote-bot/releases/latest)

Neural + search bots for **Bulgarian Bridge-Belot** (and a full C++ rules engine for **Belgian Belote**), built with the same single-GPU architecture as [manil-bot](https://github.com/frigoow/manil-bot) and powered by the shared [cardbot-framework](https://github.com/frigoow/cardbot-framework). Everything runs **100% offline** in native C++ on Android, Windows, macOS, and Linux.

> **Headline Result (100% end-to-end, zero external bot code):**
> **`58.75% ± 1.74 pp`** (**`+12.6 ± 1.9` game pts/game**, **`+61 Elo`**, `+10.0σ`, `0` fallbacks) over **2,000 mirrored 151-point games** against **unlimited-time Master** inside Nikolay Kostov's [BelotGameEngine](https://github.com/NikolayIT/BelotGameEngine/tree/master) — trained on a single consumer RTX 3060 (12 GB).

### Download the App

| Platform | Package |
|---|---|
| **Android** (arm64-v8a) | [Belot.apk](https://github.com/frigoow/belote-bot/releases/latest/download/Belot.apk) |
| **Windows** | [Belot-Windows.msi](https://github.com/frigoow/belote-bot/releases/latest/download/Belot-Windows.msi) |
| **macOS** (Apple Silicon) | [Belot-macOS.dmg](https://github.com/frigoow/belote-bot/releases/latest/download/Belot-macOS.dmg) |
| **Linux** (Ubuntu 22.04+) | [Belot-Linux.deb](https://github.com/frigoow/belote-bot/releases/latest/download/Belot-Linux.deb) |

See [`app/README.md`](app/README.md) for app build, screenshot testing, and release details.

---

## Reference Engine & Independent Benchmark

We used Nikolay Kostov's open-source [**BelotGameEngine**](https://github.com/NikolayIT/BelotGameEngine/tree/master) as our **reference implementation and independent referee** throughout development:

1. **Card-for-card Rules Oracle (`tools/kostov/oracle/`):** verifies our C++ Bulgarian Belot engine (`cpp/bg/`) against `BelotGameEngine` on legal moves, declarations, belote timing, hanging points, capot rules, and round/match scoring.
2. **Independent Arena (`tools/kostov/bridge/`):** runs mirrored 151-point whole-game matches (identical deals with teams swapped) refereed directly by `BelotGameEngine` against his strongest neural + search bot (**Master**, `ClaudePlayerProfiles.CreateMaster()`).
3. **Full-Budget Opponent:** in our benchmarks, we disable Master's wall-clock search timeouts (`SearchTimeLimitMilliseconds = 0`, `EndgameTimeLimitMilliseconds = 0`) so Master always finishes its full rollout and endgame node budget regardless of multi-threaded CPU load.

Our bots are built from scratch in C++ and PyTorch following our Manille architecture — not a port of Kostov's players.

---

## How They Play

The Bulgarian Belot bot combines an **unmasked counterfactual neural bidder** with a **3-specialist neural + search card player**. Every neural network runs in plain C++ with AVX2/FMA kernels (no PyTorch/ONNX runtime required on device).

```text
  5-card hand + auction record
               │
               ▼
       ┌──────────────┐
       │   BgBidNet   │  Unmasked neural Q-bidder (Pass, ♣ ♦ ♥ ♠, NT, AT, ×2, ×4)
       │  (0.14 ms)   │  Shared suit tower (rank s/3) + bid-history GRU
       └──────────────┘
               │
       Contract decided  ──►  Dispatch specialist trio:  All Trumps │ Suit │ No Trumps
               │
               ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ 1. Residual BeliefNet B_φ (0.14 ms)  ─►  Belief-DP weighted deal sampler   │
  │    Predicts logit deltas over exact combinatorial DealSampler::marginals() │
  │                                                                            │
  │ 2. World Weighting + Deduplication (192 worlds at Strong, 64 at Normal)    │
  │    Auction + FastPolicy play-history likelihoods; exact enum if <= 192     │
  │                                                                            │
  │ 3. CRN Candidate Racing + Root Q-GRU Tapered Blend                         │
  │    • Tricks 1–3: 6 µs FastPolicy rollouts + exact 3-trick AB handoff       │
  │    • Tricks 4–8: Exact alpha-beta solver (Zobrist TT + 0-pt pruning)       │
  └────────────────────────────────────────────────────────────────────────────┘
```

### 1. Neural Bidding (`BgBidNet`)
- **Architecture:** shared per-suit encoder (`4 × 16 → 4 × 48`) with explicit suit rank `s / 3.0` (preserving Bulgarian Belot's ordered bidding ladder `♣ < ♦ < ♥ < ♠`, where `S₄` suit permutation is illegal), a 64-dim global auction/seat encoder, a chronological bid-history GRU, and two residual trunk blocks predicting `Q(s, b)` for all 9 bid actions (`Pass`, `♣`, `♦`, `♥`, `♠`, `NT`, `AT`, `Double`, `Redouble`).
- **All-Action Paired Counterfactual Training:** at every auction node, our multithreaded C++ environment branches over **every legal bid** on the exact same 32-card deal and plays each branch out with our 3 specialist `FastPolicy` models (~25k full-deal rollouts/s). Trained over **4.5M fresh deals (`102.5M` counterfactual rollouts)** using **1-pass online streaming** (eliminating multi-epoch variance overfitting / `argmax` winner's curse), **EMA weights**, and a **paired-advantage auxiliary loss** on `Q(s, a) - Q(s, Pass)`.
- **Organically Natural Without Hand-Coded Masks:** trained with **zero artificial bidding mask** across a mixed table distribution (`50%` self-play, `25%` partnered with a natural bidder, `25%` vs aggressive-AT opponents). Without a single hard rule, `BgBidNet` organically converges to textbook conventions (`91.8%` J/9 on suit bids, `99.3%` Ace on NT, `99.2%` Jack on AT, selective `×2`/`×4` doubling) while remaining partner-compatible with both humans and Master.

### 2. Card Play (3-Specialist Hybrid Search)
Because **All Trumps**, **Suit**, and **No Trumps** have incompatible card rankings (`J/9` are top masters in AT and low cards in NT/plain suits), we train a **dedicated neural trio** for each contract regime:

- **`Q-GRU`:** Dueling action-as-input value network with a chronological play-history GRU and a training-only auxiliary `32 × 3` card-ownership head (discarded at export). Trained via GPU-resident BF16 Deep Monte Carlo with a 4-round FIFO rolling replay buffer (`20M` deals refit per specialist on our own `BgBidNet` auctions with doubles, normalizing doubled/redoubled card-play rewards by contract multiplier so card play optimizes exact winner-takes-all without `4×`/`16×` MSE gradient spikes).
- **Residual BeliefNet $B_\phi$:** predicts logit deltas on top of exact combinatorial `DealSampler::exactMarginals()` (zero-initialized output head so $B_\phi$ starts identical to exact uniform marginals and strictly improves on every trick `T1–T7`). Feeds directly into our Belief-DP weighted world sampler.
- **`FastPolicy`:** compact `256 × 128` C++ MLP (`~6 µs/call`) distilled from the converged `Q-GRU`, powering fast shared-world rollouts and soft play-history likelihoods.
- **Search & Exact Endgame:** draws Belief-DP weighted worlds (exact enumeration once `<= 192` consistent worlds remain), races candidate moves on shared worlds (CRN racing stops clear losers at `32` worlds), blends in root `Q-GRU` with late-deal tapering, and solves Trick 4 onward exactly with our Zobrist-TT alpha-beta solver (`0.014 ms` at 4 tricks left, `0.07 ms` at 5 tricks left).

---

## Results

All headline evaluations use **1,000 mirrored pairs (2,000 full 151-point games, seed `1`)** inside Kostov's `BelotGameEngine` with 95% paired confidence intervals.

### Progression Against Kostov's Master

| Milestone | Bidder | Opponent Budget | Games | Win Rate (95% CI) | Game-Point Margin |
|---|---|---|---:|---:|---:|
| Hand-crafted `rulePolicy` alone | Master | Default | 200 | `21.50% ± 5.07 pp` | `−46.7 ± 5.5` |
| 3-Spec `Q-GRU` alone (`0` search, `1 ms/card`) | Master | Default | 200 | `38.50% ± 4.80 pp` | `−14.5 ± 5.4` |
| Search + heuristic `rulePolicy` (Phase A) | Master | Default | 400 | `46.50% ± 3.15 pp` | `−2.9 ± 3.2` |
| Unified Neural Hybrid (`5M` deals) | Master | Default | 2,000 | `50.30% ± 1.56 pp` | `+0.1 ± 1.6` |
| **3-Specialist Hybrid (`5M` deals)** | Master | Default | 2,000 | `55.35% ± 1.50 pp` | `+8.5 ± 1.5` |
| **3-Specialist Hybrid (`5M`) + `BgBidNet`** | **Ours** | **Unlimited** | 2,000 | `54.35% ± 1.74 pp` | `+8.5 ± 1.9` |
| **3-Specialist Hybrid (`20M` refit)** | Master | **Unlimited** | 2,000 | `56.65% ± 1.56 pp` | `+11.3 ± 1.6` |
| **3-Specialist Hybrid (`20M` refit) + `BgBidNet`** | **Ours (100% E2E)** | **Unlimited** | 2,000 | `58.25% ± 1.68 pp` | `+12.2 ± 1.9` |
| **3-Specialist (`20M` + `Suit` $S_3$ suit-perm) + `BgBidNet` (`v1.0.0`)** | **Ours (100% E2E)** | **Unlimited** | **2,000** | **`58.75% ± 1.74 pp`** | **`+12.6 ± 1.9`** |

### Bidding & Compatibility Gates (`2,000` games each vs Unlimited-Time Master)

| Gate | Setup | Win Rate (95% CI) | Game-Point Margin | Fallbacks |
|---|---|---:|---:|---:|
| **Isolated Bidding** | `BgBidNet` + Master card play vs Master | **`51.15% ± 1.72 pp`** | **`+2.0 ± 1.9`** | `0` (`0.14 ms/bid`) |
| **Mixed Partner** | `(BgBidNet + Master)` bids + Master play vs Master | **`51.20% ± 1.63 pp`** | **`+0.1 ± 1.7`** | `0` |
| **Full End-to-End (`v1.0.0`)** | `BgBidNet` + `Suit`-$S_3$ / `AT`-`20M` / `NT`-`20M` Hybrid vs Master | **`58.75% ± 1.74 pp`** | **`+12.6 ± 1.9`** | `0` / `289,580` calls |

### Key Empirical Takeaways
- **Specialists beat unified models dramatically:** splitting `AT`, `Suit`, and `NT` jumped standalone `Q-GRU` from `19% → 38.5%` and full hybrid from `50.3% → 55.35%`.
- **Refitting specialists on our own bidder's auctions + doubles (`20M`):** jumped full end-to-end win rate by **`+3.9 pp`** (`54.35% → 58.25%`) and tightened $B_\phi$ belief cross-entropy across every contract (`AT 0.836`, `Suit 0.871`, `NT 0.894–0.901` vs exact uniform `DealSampler 0.975–1.049`).
- **Suit-permutation symmetry ($S_3$ vs $S_4$):** adding exact $S_3$ plain-suit permutation (trump pinned at slot `0`) on `Suit` improved `Q-GRU` (`+2.056 → +2.233`), $B_\phi$ (`0.871 → 0.864/0.868`), and `FastPolicy` (`+2.156 → +2.198`), lifting full E2E to **`58.75%`** (**`+12.6 pts/game`**, **`+61 Elo`**). By contrast, $S_4$ on `NT` scrambled Bulgarian Belot's ordered auction ladder (`♣ < ♦ < ♥ < ♠`) and hurt `NT` $B_\phi$, so `v1.0.0` keeps non-permuted `AT-20M` and `NT-20M` alongside `Suit`-$S_3$.
- **Selective exact endgame (`--exact-trick 4 --endgame 3`) beats both extremes:** turning exact AB off (`512` pure rollout worlds) loses `~6 pts/game` *and* runs slower, while pushing double-dummy AB too early inside rollouts (`--endgame 5`) drops to `57.85%` due to Trick-4 X-ray optimism bias.
- **1-pass streaming + EMA solves bidding RL collapse:** multi-epoch replay on noisy single-deal targets (`σ ≈ 21 pts`) caused `+9.6 pt` winner's-curse overbidding on high-variance `NT`/`AT`; 1-pass streaming on fresh counterfactual batches eliminated overfitting completely (`train RMSE 25.60` vs `val RMSE 25.70`).

---

## Repository Layout

| Path | Description |
|---|---|
| `cpp/bg/` | Bulgarian Bridge-Belot engine, exact `DealSampler`, Belief-DP, AVX2 neural inference (`BgBidModel`, `BgFastPolicyModel`, `BgBeliefModel`, `BgQGRUModel`), `searchCard`, exact AB solver, and multithreaded training environments (`BgBatchPlayEnv`, `BgBatchBidEnv`) |
| `cpp/belote/` | Belgian Belote rules engine ([`docs/RULES_BELGIAN.md`](docs/RULES_BELGIAN.md)), `WorldSampler`, and `belote_cpp` module |
| `python/bg/` | PyTorch BF16 trainers (`train_dmc.py`, `train_bid.py`), `S₃`/`S₄` suit-permutation augmentation (`suit_perm.py`), model export, and C++ parity tests |
| `app/` | Cross-platform Bulgarian Belot app (Android, Windows, macOS, Linux) built on [`cardbot-framework`](https://github.com/frigoow/cardbot-framework) |
| `tools/kostov/` | C# oracle (`oracle/`) and mirrored-match arena (`bridge/`) against `BelotGameEngine` |
| `docs/` | Full rules specifications ([`RULES_BULGARIAN.md`](docs/RULES_BULGARIAN.md), [`RULES_BELGIAN.md`](docs/RULES_BELGIAN.md)) |
| `cardbot-framework/` | Shared C++ search/NN/sampler kernels and Compose Multiplatform UI framework (git submodule) |

---

## Build, Test & Train

```bash
git clone --recurse-submodules https://github.com/frigoow/belote-bot.git
cd belote-bot

# C++ engines, tools, and unit tests
cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Python bindings (for training / evaluation)
python3.12 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/cmake -B build -S cpp -DBELOTE_PYTHON=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python \
    -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')
.venv/bin/cmake --build build -j

# Fetch the latest trained networks (belot-models-v3) for the app
app/fetch_models.sh
```

Training logs and binary checkpoints live in gitignored `runs/` and `checkpoints/`. Pre-trained network bundles are published as [`belot-models-v3`](https://github.com/frigoow/belote-bot/releases/tag/belot-models-v3).

## License

Licensed under the [Apache License 2.0](LICENSE) (including the attribution requirements in [NOTICE](NOTICE)).
