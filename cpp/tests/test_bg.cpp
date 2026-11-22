// Bulgarian Belot engine (bg/bg.h) against Kostov's BelotGameEngine: replays
// the oracle log written by tools/kostov/oracle (seeded random games through his
// BelotMatch) and checks every decision's seat and legal options, every bid
// (forced passes included), the contract, the declarations and which of them
// scored, every trick, every deal's score and the game result.
//
// Usage: test_bg [oracle-file]   (default: tests/data/bg_oracle.txt next to this file)

#include "bg/bg.h"
#include "check.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace bg;

namespace {

struct RoundLog {
    int number = 0, first = 0;
    std::array<int, kCards> deal{};
    std::vector<std::pair<int, int>> bids;
    int contract = 0, contract_player = 0;
    std::vector<std::array<int, 4>> announces;  // seat type card scored
    std::vector<std::string> tricks;
    std::array<int, 7> result{};
};

std::vector<int> ints(std::istringstream& in) {
    std::vector<int> v;
    for (long long x; in >> x;) v.push_back(int(x));
    return v;
}

int replayGame(int game, int first, const std::vector<std::string>& decisions, const std::vector<RoundLog>& rounds,
               const std::vector<int>& final_line) {
    GameScore gs;
    gs.first = first;
    std::size_t next = 0;
    int checked = 0;
    for (const RoundLog& rl : rounds) {
        CHECK_EQ(rl.first, gs.first);
        Round r(rl.deal, gs.first, gs.hanging);
        while (!r.finished()) {
            if (next >= decisions.size()) {
                std::fprintf(stderr, "game %d round %d: engine asks for more decisions than the log has\n", game, rl.number);
                ++check::failures();
                return checked;
            }
            std::istringstream in(decisions[next++]);
            char kind;
            int round_no, seat;
            in >> kind >> round_no >> seat;
            CHECK_EQ(round_no, rl.number);
            CHECK_EQ(seat, r.toMove());
            ++checked;
            if (kind == 'B') {
                CHECK(r.decision() == Round::Decision::kBid);
                int avail, chosen;
                in >> avail >> chosen;
                CHECK_EQ(int(r.availableBids()), avail);
                r.bid(std::uint8_t(chosen));
            } else if (kind == 'A') {
                CHECK(r.decision() == Round::Decision::kAnnounce);
                int n;
                in >> n;
                std::vector<Announce> offered(static_cast<std::size_t>(n));
                for (Announce& a : offered) in >> a.type >> a.card;
                const auto& mine = r.offeredAnnounces();
                CHECK_EQ(mine.size(), offered.size());
                for (std::size_t i = 0; i < offered.size() && i < mine.size(); ++i) {
                    CHECK_EQ(mine[i].type, offered[i].type);
                    CHECK_EQ(mine[i].card, offered[i].card);
                }
                std::string bar;
                int k;
                in >> bar >> k;
                std::vector<Announce> chosen;
                for (int i = 0, idx; i < k && in >> idx;) {
                    chosen.push_back(offered[std::size_t(idx)]);
                    ++i;
                }
                r.declare(chosen);
            } else if (kind == 'P') {
                CHECK(r.decision() == Round::Decision::kPlay);
                long long legal;
                int card, belote;
                in >> legal >> card >> belote;
                CHECK_EQ(r.legalMoves(), Mask(legal));
                r.play(card, belote != 0);
            }
        }
        // The whole deal.
        CHECK_EQ(r.bids().size(), rl.bids.size());
        for (std::size_t i = 0; i < rl.bids.size() && i < r.bids().size(); ++i) {
            CHECK_EQ(r.bids()[i].first, rl.bids[i].first);
            CHECK_EQ(int(r.bids()[i].second), rl.bids[i].second);
        }
        CHECK_EQ(int(r.contract()), rl.contract);
        CHECK_EQ(r.contractPlayer(), rl.contract_player);
        CHECK_EQ(r.announces().size(), rl.announces.size());
        for (std::size_t i = 0; i < rl.announces.size() && i < r.announces().size(); ++i) {
            const Announce& a = r.announces()[i];
            CHECK_EQ(a.player, rl.announces[i][0]);
            CHECK_EQ(a.type, rl.announces[i][1]);
            CHECK_EQ(a.card, rl.announces[i][2]);
            CHECK_EQ(a.scored, rl.announces[i][3]);
        }
        CHECK_EQ(r.tricks().size(), rl.tricks.size());
        for (std::size_t t = 0; t < rl.tricks.size() && t < r.tricks().size(); ++t) {
            std::ostringstream mine;
            for (int i = 0; i < kPlayers; ++i) {
                const auto& pl = r.plays()[t * 4 + std::size_t(i)];
                mine << pl.first << ' ' << pl.second.first << ' ' << (pl.second.second ? 1 : 0) << ' ';
            }
            mine << r.trickWinners()[t];
            CHECK(mine.str() == rl.tricks[t]);
        }
        const RoundResult& res = r.result();
        const std::array<int, 7> got{rl.number, res.sn, res.ew, res.sn_total, res.ew_total, res.no_tricks ? 1 : 0, res.hanging};
        for (int i = 1; i < 7; ++i) CHECK_EQ(got[std::size_t(i)], rl.result[std::size_t(i)]);
        gs.record(r.contract(), res);
    }
    CHECK_EQ(next, decisions.size());
    CHECK_EQ(gs.winner, final_line[0]);
    CHECK_EQ(gs.sn, final_line[1]);
    CHECK_EQ(gs.ew, final_line[2]);
    CHECK_EQ(gs.hanging, final_line[3]);
    return checked;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/')) + "/data/bg_oracle.txt";
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "test_bg: cannot open %s\n", path.c_str());
        return 1;
    }
    int games = 0, decisions_checked = 0, rounds_checked = 0;
    std::map<int, int> contracts;
    std::string line;
    int game = -1, first = 0;
    std::vector<std::string> decisions;
    std::vector<RoundLog> rounds;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        char kind;
        ls >> kind;
        if (kind == 'G') {
            ls >> game >> first;
            decisions.clear();
            rounds.clear();
        } else if (kind == 'B' || kind == 'A' || kind == 'P') {
            decisions.push_back(line);
        } else if (kind == 'R') {
            RoundLog rl;
            ls >> rl.number >> rl.first;
            for (int& c : rl.deal) ls >> c;
            rounds.push_back(rl);
        } else if (kind == 'S') {
            int no, n;
            ls >> no >> n;
            for (int i = 0, s, t; i < n && ls >> s >> t; ++i) rounds.back().bids.push_back({s, t});
        } else if (kind == 'C') {
            int no;
            ls >> no >> rounds.back().contract >> rounds.back().contract_player;
            ++contracts[rounds.back().contract & 63];
        } else if (kind == 'N') {
            int no, n;
            ls >> no >> n;
            for (int i = 0; i < n; ++i) {
                std::array<int, 4> a{};
                ls >> a[0] >> a[1] >> a[2] >> a[3];
                rounds.back().announces.push_back(a);
            }
        } else if (kind == 'T') {
            int no;
            ls >> no;
            std::string rest;
            std::getline(ls, rest);
            rounds.back().tricks.push_back(rest.substr(rest.find_first_not_of(' ')));
        } else if (kind == 'E') {
            auto v = ints(ls);
            for (std::size_t i = 0; i < 7 && i < v.size(); ++i) rounds.back().result[i] = v[i];
        } else if (kind == 'W') {
            const auto v = ints(ls);
            decisions_checked += replayGame(game, first, decisions, rounds, v);
            rounds_checked += int(rounds.size());
            ++games;
        }
    }
    std::printf("bg oracle: %d games, %d deals, %d decisions replayed (contracts:", games, rounds_checked, decisions_checked);
    for (const auto& [c, n] : contracts) std::printf(" %d:%d", c, n);
    std::printf(")\n");
    return checkResult("test_bg");
}
