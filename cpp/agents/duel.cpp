#include "agents/duel.h"

#include "agents/agent.h"
#include "belote/match.h"
#include "search/belote_game.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace belote {

std::string configSummary(const Config& c) {
    auto tieStr = [](TieRule t) {
        switch (t) {
            case TieRule::kHanging: return "hanging";
            case TieRule::kTakersMake: return "takers_make";
            case TieRule::kTakersFail: return "takers_fail";
        }
        return "?";
    };
    auto scoreStr = [](ScoreMode s) {
        switch (s) {
            case ScoreMode::kComplement: return "complement";
            case ScoreMode::kUnits: return "units";
            case ScoreMode::kRaw: return "raw";
        }
        return "?";
    };
    std::ostringstream os;
    os << "target=" << c.match_target << ",margin=" << c.match_win_margin
       << ",score=" << scoreStr(c.score_mode) << ",tie=" << tieStr(c.tie_rule)
       << ",ronde=" << c.ronde_points
       << ",redeal_next=" << (c.redeal_next_dealer ? 1 : 0)
       << ",r2_blik=" << (c.round2_allow_blik_suit ? 1 : 0)
       << ",partner_over=" << (c.trump_lead_partner_must_overtrump ? 1 : 0)
       << ",undertrump=" << (c.must_undertrump ? 1 : 0)
       << ",belot_q_forf=" << (c.belot_queen_first_forfeits ? 1 : 0)
       << ",toon=" << (c.toon_enabled ? 1 : 0)
       << "/low=" << c.carre_low_points
       << "/opt=" << (c.toon_optional ? 1 : 0)
       << "/show=" << c.toon_show_trick
       << "/losers=" << (c.toon_losers_show ? 1 : 0);
    return os.str();
}

namespace {

using Clock = std::chrono::steady_clock;

struct SeatAgents {
    Agent* seat[kPlayers]{};
    int timing_slot[kPlayers]{-1, -1, -1, -1};  // 0 = A, 1 = B, -1 = partner/fixed
};

SeatAgents assignSeats(int a_team, int partner_slot, Agent& a, Agent& b, Agent* partner) {
    SeatAgents s;
    for (int p = 0; p < kPlayers; ++p) {
        const bool is_partner_seat = partner && ((p / 2) == partner_slot);
        if (is_partner_seat) {
            s.seat[p] = partner;
            s.timing_slot[p] = -1;
        } else if (teamOf(p) == a_team) {
            s.seat[p] = &a;
            s.timing_slot[p] = 0;
        } else {
            s.seat[p] = &b;
            s.timing_slot[p] = 1;
        }
    }
    return s;
}

struct Run {
    bool passed = false;
    double units = 0, raw = 0;
    double seconds[2] = {0, 0};
    long cards[2] = {0, 0};
    DealResult result{};
};

Run playDeal(const Config& cfg, const std::array<int, kCards>& deck, int dealer, int a_team,
             const SeatAgents& seats, Agent* fixed_bidder, Agent* fixed_player,
             std::vector<Bid>* record_bids = nullptr, const std::vector<Bid>* replay_bids = nullptr) {
    Run run;
    Deal d(cfg, dealer, deck);
    std::size_t bid_idx = 0;
    while (d.phase() == Phase::kBid1 || d.phase() == Phase::kBid2) {
        Bid chosen;
        if (replay_bids && bid_idx < replay_bids->size()) {
            chosen = (*replay_bids)[bid_idx++];
        } else {
            Agent& who = fixed_bidder ? *fixed_bidder : *seats.seat[d.currentPlayer()];
            chosen = who.chooseBid(d);
            if (record_bids) record_bids->push_back(chosen);
        }
        d.bid(chosen);
    }
    if (d.phase() == Phase::kDone) {
        run.passed = true;
        run.result = d.result();
        return run;
    }
    while (d.phase() == Phase::kPlay) {
        const int p = d.currentPlayer();
        Agent& who = fixed_player ? *fixed_player : *seats.seat[p];
        if (d.canSetDeclare(p)) d.setDeclare(p, who.chooseDeclare(d));
        const auto t0 = Clock::now();
        const int card = who.chooseCard(d);
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (!fixed_player) {
            const int slot = seats.timing_slot[p];
            if (slot >= 0) {
                run.seconds[slot] += dt;
                ++run.cards[slot];
            }
        }
        d.play(card);
    }
    run.result = d.result();
    run.units = unitMargin(d, a_team);
    run.raw = d.result().credited[a_team] - d.result().credited[1 - a_team];
    return run;
}

std::array<int, kCards> shuffledDeck(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    return deck;
}

void meanSe(const std::vector<double>& x, double& mean, double& se) {
    const double n = double(x.size());
    mean = n ? std::accumulate(x.begin(), x.end(), 0.0) / n : 0;
    double ss = 0;
    for (double v : x) ss += (v - mean) * (v - mean);
    se = n > 1 ? std::sqrt(ss / (n - 1) / n) : 0;
}

}  // namespace

DuelResult runDuel(const DuelOptions& o) {
    const auto start = Clock::now();
    std::vector<Run> first(std::size_t(o.deals)), second(std::size_t(o.deals));
    std::atomic<int> next{0};

    auto worker = [&](int w) {
        const std::uint64_t base = o.seed * 0x9E3779B97F4A7C15ull + std::uint64_t(w) * 7919;
        auto a = makeAgent(o.a, base + 1);
        auto b = makeAgent(o.b, base + 2);
        std::unique_ptr<Agent> fixed_bidder = o.bidder == "self" ? nullptr : makeAgent(o.bidder, base + 3);
        std::unique_ptr<Agent> fixed_player = o.player.empty() ? nullptr : makeAgent(o.player, base + 4);
        std::unique_ptr<Agent> partner = o.partner.empty() ? nullptr : makeAgent(o.partner, base + 5);

        for (int i; (i = next.fetch_add(1)) < o.deals;) {
            const std::uint64_t deck_seed = o.seed * 1000003ull + std::uint64_t(i);
            const auto deck = shuffledDeck(deck_seed);
            const int dealer = i % kPlayers;
            const int partner_slot = (i / kPlayers) & 1;
            const SeatAgents seats0 = assignSeats(0, partner_slot, *a, *b, partner.get());
            const SeatAgents seats1 = assignSeats(1, partner_slot, *a, *b, partner.get());

            auto seedRun = [&](std::uint64_t run_tag) {
                const std::uint64_t s = deck_seed * 0x9E3779B97F4A7C15ull + run_tag * 7919ull;
                a->reseed(s + 1);
                b->reseed(s + 2);
                if (fixed_bidder) fixed_bidder->reseed(s + 3);
                if (fixed_player) fixed_player->reseed(s + 4);
                if (partner) partner->reseed(s + 5);
            };

            if (fixed_bidder) {
                // Record the fixed bidder's auction so even stochastic fixed
                // bidders produce the identical contract on both mirrored runs.
                std::vector<Bid> bids;
                seedRun(1);
                first[std::size_t(i)] =
                    playDeal(o.config, deck, dealer, 0, seats0, fixed_bidder.get(), fixed_player.get(), &bids, nullptr);
                seedRun(2);
                second[std::size_t(i)] =
                    playDeal(o.config, deck, dealer, 1, seats1, fixed_bidder.get(), fixed_player.get(), nullptr, &bids);
            } else {
                seedRun(1);
                first[std::size_t(i)] =
                    playDeal(o.config, deck, dealer, 0, seats0, nullptr, fixed_player.get());
                seedRun(2);
                second[std::size_t(i)] =
                    playDeal(o.config, deck, dealer, 1, seats1, nullptr, fixed_player.get());
            }
        }
    };
    std::vector<std::thread> pool;
    for (int w = 0; w < std::max(1, o.threads); ++w) pool.emplace_back(worker, w);
    for (auto& t : pool) t.join();

    DuelResult r;
    r.config_summary = configSummary(o.config);
    std::vector<double> units, raw;
    double secs[2] = {0, 0};
    long cards[2] = {0, 0};
    for (int i = 0; i < o.deals; ++i) {
        const Run& x = first[std::size_t(i)];
        const Run& y = second[std::size_t(i)];
        // Only skip when BOTH runs all-passed. Under bidder=="self", an
        // asymmetric pass (one side bids, the other passes for 0 margin) is a
        // real bidding decision and must be kept to avoid selection bias.
        if (x.passed && y.passed) {
            ++r.skipped;
            continue;
        }
        units.push_back((x.units + y.units) / 2.0);
        raw.push_back((x.raw + y.raw) / 2.0);
        for (int s = 0; s < 2; ++s) {
            secs[s] += x.seconds[s] + y.seconds[s];
            cards[s] += x.cards[s] + y.cards[s];
        }
    }
    r.decks = int(units.size());
    meanSe(units, r.units, r.units_se);
    meanSe(raw, r.raw, r.raw_se);
    r.ms_per_card_a = cards[0] ? 1000.0 * secs[0] / cards[0] : 0;
    r.ms_per_card_b = cards[1] ? 1000.0 * secs[1] / cards[1] : 0;
    r.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return r;
}

MatchDuelResult runMatchDuel(const MatchDuelOptions& o) {
    const auto start = Clock::now();
    struct PairOutcome {
        double win_score = 0;     // average over the two mirrored matches: 0.0, 0.5, or 1.0
        double score_margin = 0;  // average final score margin for A
        double deals = 0;
        double seconds[2] = {0, 0};
        long cards[2] = {0, 0};
    };
    std::vector<PairOutcome> outcomes(std::size_t(o.matches));
    std::atomic<int> next{0};
    constexpr int kMaxDealsPerMatch = 500;

    auto worker = [&](int w) {
        const std::uint64_t base = o.seed * 0x9E3779B97F4A7C15ull + std::uint64_t(w) * 7919;
        auto a = makeAgent(o.a, base + 1);
        auto b = makeAgent(o.b, base + 2);
        std::unique_ptr<Agent> fixed_bidder = o.bidder == "self" ? nullptr : makeAgent(o.bidder, base + 3);
        std::unique_ptr<Agent> fixed_player = o.player.empty() ? nullptr : makeAgent(o.player, base + 4);
        std::unique_ptr<Agent> partner = o.partner.empty() ? nullptr : makeAgent(o.partner, base + 5);

        for (int i; (i = next.fetch_add(1)) < o.matches;) {
            const int first_dealer = i % kPlayers;
            const int partner_slot = (i / kPlayers) & 1;
            const std::uint64_t pair_seed = o.seed * 1000003ull + std::uint64_t(i) * 10007ull;

            PairOutcome po;
            for (int side = 0; side < 2; ++side) {
                const int a_team = side;
                const SeatAgents seats = assignSeats(a_team, partner_slot, *a, *b, partner.get());
                const std::uint64_t ms = pair_seed * 0x9E3779B97F4A7C15ull + std::uint64_t(side + 1) * 7919ull;
                a->reseed(ms + 1);
                b->reseed(ms + 2);
                if (fixed_bidder) fixed_bidder->reseed(ms + 3);
                if (fixed_player) fixed_player->reseed(ms + 4);
                if (partner) partner->reseed(ms + 5);

                Match m(o.config, first_dealer);
                int total_deals = 0;
                int decided_idx = 0;
                int redeal_attempt = 0;
                while (!m.finished() && total_deals < kMaxDealsPerMatch) {
                    // Key decks by (decided_idx, redeal_attempt) so a redeal in
                    // one mirrored match doesn't shift all subsequent decided
                    // deals onto different decks.
                    const std::uint64_t dseed =
                        pair_seed + std::uint64_t(decided_idx) * 1009ull + std::uint64_t(redeal_attempt);
                    const auto deck = shuffledDeck(dseed);
                    const Run run = playDeal(o.config, deck, m.dealer(), a_team, seats, fixed_bidder.get(),
                                             fixed_player.get());
                    for (int s = 0; s < 2; ++s) {
                        po.seconds[s] += run.seconds[s];
                        po.cards[s] += run.cards[s];
                    }
                    m.record(run.result);
                    ++total_deals;
                    if (run.passed) {
                        ++redeal_attempt;
                    } else {
                        ++decided_idx;
                        redeal_attempt = 0;
                    }
                }
                const double w_val = !m.finished() ? 0.5 : (m.winner() == a_team ? 1.0 : 0.0);
                po.win_score += w_val / 2.0;
                po.score_margin += double(m.scores()[a_team] - m.scores()[1 - a_team]) / 2.0;
                po.deals += double(m.dealsRecorded()) / 2.0;
            }
            outcomes[std::size_t(i)] = po;
        }
    };

    std::vector<std::thread> pool;
    for (int w = 0; w < std::max(1, o.threads); ++w) pool.emplace_back(worker, w);
    for (auto& t : pool) t.join();

    MatchDuelResult r;
    r.config_summary = configSummary(o.config);
    r.pairs = o.matches;
    std::vector<double> wins, margins;
    wins.reserve(outcomes.size());
    margins.reserve(outcomes.size());
    double deals_sum = 0;
    double secs[2] = {0, 0};
    long cards[2] = {0, 0};
    for (const PairOutcome& po : outcomes) {
        wins.push_back(po.win_score);
        margins.push_back(po.score_margin);
        deals_sum += po.deals;
        for (int s = 0; s < 2; ++s) {
            secs[s] += po.seconds[s];
            cards[s] += po.cards[s];
        }
    }
    meanSe(wins, r.win_rate, r.win_rate_se);
    meanSe(margins, r.score_margin, r.score_margin_se);
    r.mean_deals = outcomes.empty() ? 0 : deals_sum / double(outcomes.size());
    r.ms_per_card_a = cards[0] ? 1000.0 * secs[0] / cards[0] : 0;
    r.ms_per_card_b = cards[1] ? 1000.0 * secs[1] / cards[1] : 0;
    r.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return r;
}

}  // namespace belote
