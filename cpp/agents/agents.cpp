#include "agents/agent.h"
#include "agents/rule_agent.h"

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace belote {

// ---------------------------------------------------------------- bidding

double ruleHandValue(Mask cards, int trump) {
    // Trump honours, length and belot; side aces and guarded tens.
    static constexpr double kTrumpValue[kRanks] = {0.4, 0.4, 2.5, 1.1, 3.5, 0.5, 0.5, 1.4};  // 7 8 9 10 J Q K A
    double v = 0;
    const Mask trumps = cards & suitMask(trump);
    for (Mask m = trumps; m; m &= m - 1) v += kTrumpValue[rankOf(lowestCard(m))];
    const int n = popcount(trumps);
    if (n >= 4) v += 0.5 * (n - 3);
    const Mask kq = bit(cardOf(trump, kK)) | bit(cardOf(trump, kQ));
    if ((trumps & kq) == kq) v += 1.0;
    for (int s = 0; s < kSuits; ++s) {
        if (s == trump) continue;
        const Mask side = cards & suitMask(s);
        const bool ace = side & bit(cardOf(s, kA));
        const bool ten = side & bit(cardOf(s, k10));
        if (ace) v += 1.0;
        if (ten) v += ace ? 0.6 : (popcount(side) >= 2 ? 0.3 : 0.0);
    }
    return v;
}

Bid ruleBid(const Deal& d, double threshold) {
    const Mask cards = d.hand(d.currentPlayer()) | bit(d.blik());  // the taker gets the blik
    const int blik_suit = suitOf(d.blik());
    if (d.phase() == Phase::kBid1)
        return ruleHandValue(cards, blik_suit) >= threshold ? Bid::take() : Bid::pass();
    Bid best = Bid::pass();
    double best_value = threshold;
    for (const Bid& b : d.legalBids()) {
        if (b.kind != Bid::kChoose) continue;
        const double v = ruleHandValue(cards, b.suit);
        if (v >= best_value) {
            best_value = v;
            best = b;
        }
    }
    return best;
}

namespace {

bool strictlyBeatsAnnouncement(const Announcement& opp, const Announcement& mine) {
    if (int(opp.kind) > int(mine.kind)) return true;
    if (opp.kind == ToonKind::kCarre && mine.kind == ToonKind::kCarre)
        return kTrumpStrength[opp.carre_rank] > kTrumpStrength[mine.carre_rank];
    return false;
}

}  // namespace

bool smartChooseDeclare(const Deal& d) {
    const int me = d.currentPlayer();
    if (!d.canSetDeclare(me) || d.toons(me).empty()) return true;

    const Trick& t = d.trick();
    bool partner_played = false;
    const int partner = partnerOf(me);
    for (int i = 0; i < t.n; ++i)
        if (t.players[i] == partner) partner_played = true;
    if (!partner_played) return true;  // partner may still hold a winning toon

    const Announcement my_ann = announcementOf(d.toons(me), d.trump());
    const bool partner_ann = d.announced(partner);

    for (int i = 0; i < t.n; ++i) {
        const int p = t.players[i];
        if (teamOf(p) == teamOf(me) || !d.announced(p)) continue;
        const Announcement& opp_ann = d.announcement(p);
        if (strictlyBeatsAnnouncement(opp_ann, my_ann) &&
            (!partner_ann || strictlyBeatsAnnouncement(opp_ann, d.announcement(partner)))) {
            return false;  // zero chance to win the toon comparison; avoid leaking
        }
    }
    return true;
}

// ---------------------------------------------------------------- random

Bid RandomAgent::chooseBid(const Deal& d) {
    const auto bids = d.legalBids();
    if (std::bernoulli_distribution(p_take_)(rng_))
        return bids[1 + std::uniform_int_distribution<std::size_t>(0, bids.size() - 2)(rng_)];
    return Bid::pass();
}

int RandomAgent::chooseCard(const Deal& d) {
    Mask m = d.legalMoves();
    int k = std::uniform_int_distribution<int>(0, popcount(m) - 1)(rng_);
    while (k--) m &= m - 1;
    return lowestCard(m);
}

// ---------------------------------------------------------------- rule play

namespace {

// Global order of cards for "highest": any trump above any plain card.
int power(int card, int trump) { return cardStrength(card, trump) + (suitOf(card) == trump ? 8 : 0); }

// The cheapest card of `m`: fewest points, then weakest.
int cheapest(Mask m, int trump) {
    int best = kNoCard;
    for (; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (best == kNoCard || cardPoints(c, trump) < cardPoints(best, trump) ||
            (cardPoints(c, trump) == cardPoints(best, trump) && power(c, trump) < power(best, trump)))
            best = c;
    }
    return best;
}

int richest(Mask m, int trump) {
    int best = kNoCard;
    for (; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (best == kNoCard || cardPoints(c, trump) > cardPoints(best, trump) ||
            (cardPoints(c, trump) == cardPoints(best, trump) && power(c, trump) < power(best, trump)))
            best = c;
    }
    return best;
}

int strongest(Mask m, int trump) {
    int best = kNoCard;
    for (; m; m &= m - 1) {
        const int c = lowestCard(m);
        if (best == kNoCard || power(c, trump) > power(best, trump)) best = c;
    }
    return best;
}

bool wouldWin(const Trick& t, int card, int player, int trump) {
    Trick u = t;
    u.cards[u.n] = card;
    u.players[u.n] = player;
    ++u.n;
    return trickWinnerIndex(u, trump) == t.n;
}

}  // namespace

int ruleChooseCard(const Deal& d) {
    const int me = d.currentPlayer();
    const int trump = d.trump();
    Mask legal = d.legalMoves();
    if (popcount(legal) == 1) return lowestCard(legal);

    // Belot guard under belot_queen_first_forfeits: when holding both trump K
    // and Q, K is always legal whenever Q is, so mask out Q to preserve the 20.
    if (d.config().belot_queen_first_forfeits) {
        const Mask k_bit = bit(cardOf(trump, kK));
        const Mask q_bit = bit(cardOf(trump, kQ));
        if ((d.hand(me) & (k_bit | q_bit)) == (k_bit | q_bit) && (legal & k_bit)) legal &= ~q_bit;
    }

    const Trick& t = d.trick();
    const Mask plain = legal & ~suitMask(trump);

    if (t.n == 0) {
        // Takers pull trumps with a strong holding; then side aces; then cheap plain cards.
        const Mask my_trumps = legal & suitMask(trump);
        if (teamOf(me) == d.takers() && my_trumps &&
            ((my_trumps & bit(cardOf(trump, kJ))) || popcount(my_trumps) >= 3))
            return strongest(my_trumps, trump);
        for (Mask m = plain; m; m &= m - 1)
            if (rankOf(lowestCard(m)) == kA) return lowestCard(m);
        return cheapest(plain ? plain : legal, trump);
    }

    const int w = trickWinnerIndex(t, trump);
    const bool partner_winning = t.players[w] == partnerOf(me);
    const bool last = t.n == kPlayers - 1;
    Mask winners = 0;
    for (Mask m = legal; m; m &= m - 1)
        if (wouldWin(t, lowestCard(m), me, trump)) winners |= bit(lowestCard(m));

    if (partner_winning) {
        // Give partner points when the trick is safe (last to play, or
        // partner holds it with a trump); otherwise stay cheap.
        if (last || suitOf(t.cards[w]) == trump) return richest(plain ? plain : legal, trump);
        return cheapest(legal, trump);
    }
    if (winners) {
        int points = 0;
        for (int i = 0; i < t.n; ++i) points += cardPoints(t.cards[i], trump);
        if (last) return cheapest(winners, trump);
        // Not last: win high in the led suit; ruff only when the trick is worth it.
        const Mask plain_winners = winners & ~suitMask(trump);
        if (plain_winners) return strongest(plain_winners, trump);
        if (points >= 10) return cheapest(winners, trump);
    }
    return cheapest(legal, trump);
}

// ---------------------------------------------------------------- factory

std::unique_ptr<Agent> makeAgent(const std::string& spec, std::uint64_t seed) {
    if (auto rule_agent = makeRuleAgent(spec, seed)) return rule_agent;
    if (spec == "random") return std::make_unique<RandomAgent>(seed);
    if (spec == "rule") return std::make_unique<RuleAgent>();
    if (spec.rfind("ismcts:", 0) == 0) {
        IsmctsOptions o;
        const std::string rest = spec.substr(7);
        const auto colon = rest.find(':');
        o.iterations = unsigned(std::stoul(rest.substr(0, colon)));
        if (colon != std::string::npos) o.threads = unsigned(std::stoul(rest.substr(colon + 1)));
        return std::make_unique<IsmctsAgent>(o, seed);
    }
    if (spec.rfind("rollout:", 0) == 0) {
        RolloutOptions o;
        const std::string rest = spec.substr(8);
        const auto colon = rest.find(':');
        o.worlds = std::stoi(rest.substr(0, colon));
        if (colon != std::string::npos) {
            const std::string tail = rest.substr(colon + 1);
            if (tail.rfind("ab", 0) == 0) o.ab_handoff_trick = std::stoi(tail.substr(2));
            else throw std::invalid_argument("unknown rollout option: " + tail);
        }
        return std::make_unique<RolloutAgent>(o, seed);
    }
    auto loadCached = [](const std::string& path) -> std::shared_ptr<const FastPolicyModel> {
        static std::mutex mu;
        struct Entry {
            std::filesystem::file_time_type mtime;
            std::shared_ptr<const FastPolicyModel> model;
        };
        static std::unordered_map<std::string, Entry> cache;
        std::error_code ec;
        const auto mt = std::filesystem::last_write_time(path, ec);
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(path);
        if (it != cache.end() && !ec && it->second.mtime == mt) return it->second.model;
        auto m = FastPolicyModel::load(path);
        cache[path] = Entry{mt, m};
        return m;
    };
    auto loadBidCached = [](const std::string& path) -> std::shared_ptr<const BidModel> {
        static std::mutex mu;
        struct Entry {
            std::filesystem::file_time_type mtime;
            std::shared_ptr<const BidModel> model;
        };
        static std::unordered_map<std::string, Entry> cache;
        std::error_code ec;
        const auto mt = std::filesystem::last_write_time(path, ec);
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(path);
        if (it != cache.end() && !ec && it->second.mtime == mt) return it->second.model;
        auto m = std::make_shared<const BidModel>(BidModel::load(path));
        cache[path] = Entry{mt, m};
        return m;
    };
    if (spec.rfind("bid:", 0) == 0) {
        // bid:<bundle.bin>[:s3=<worlds>[:m=<margin>][:<fast.bin>]][+<play_spec>]
        const auto plus = spec.find('+');
        const std::string bid_part = (plus == std::string::npos) ? spec.substr(4) : spec.substr(4, plus - 4);
        const std::string play_spec = (plus == std::string::npos) ? "rule" : spec.substr(plus + 1);
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (start <= bid_part.size()) {
            const auto pos = bid_part.find(':', start);
            if (pos == std::string::npos) {
                parts.push_back(bid_part.substr(start));
                break;
            }
            parts.push_back(bid_part.substr(start, pos - start));
            start = pos + 1;
        }
        if (parts.empty() || parts[0].empty()) throw std::invalid_argument("bid spec needs bid:<bundle.bin>");
        int s3_worlds = 0;
        double s3_margin = 0.04;
        std::string fast_path;
        for (std::size_t i = 1; i < parts.size(); ++i) {
            if (parts[i].rfind("s3=", 0) == 0) s3_worlds = std::stoi(parts[i].substr(3));
            else if (parts[i].rfind("m=", 0) == 0) s3_margin = std::stod(parts[i].substr(2));
            else fast_path = parts[i];
        }
        // If s3 is requested without an explicit fast_path, infer from fast:/hybrid: play_spec.
        if (s3_worlds > 0 && fast_path.empty()) {
            if (play_spec.rfind("fast:", 0) == 0) {
                fast_path = play_spec.substr(5);
            } else if (play_spec.rfind("hybrid:", 0) == 0) {
                const auto c2 = play_spec.find(':', 7);
                fast_path = play_spec.substr(7, c2 == std::string::npos ? std::string::npos : c2 - 7);
            }
        }
        std::shared_ptr<const FastPolicyModel> sp = fast_path.empty() ? nullptr : loadCached(fast_path);
        return std::make_unique<BidWrapperAgent>(loadBidCached(parts[0]), makeAgent(play_spec, seed), std::move(sp),
                                                 s3_worlds, s3_margin, spec, seed);
    }
    if (spec.rfind("fast:", 0) == 0) {
        const std::string path = spec.substr(5);
        return std::make_unique<FastPolicyAgent>(loadCached(path), spec);
    }
    if (spec.rfind("hybrid:", 0) == 0) {
        // hybrid:<path>:<worlds>[:ab<trick>][:nowt]
        std::vector<std::string> parts;
        std::size_t start = 7;
        while (start <= spec.size()) {
            const auto pos = spec.find(':', start);
            if (pos == std::string::npos) {
                parts.push_back(spec.substr(start));
                break;
            }
            parts.push_back(spec.substr(start, pos - start));
            start = pos + 1;
        }
        if (parts.size() < 2) throw std::invalid_argument("hybrid spec needs hybrid:<path>:<worlds>");
        HybridOptions o;
        o.worlds = std::stoi(parts[1]);
        for (std::size_t i = 2; i < parts.size(); ++i) {
            if (parts[i].rfind("ab", 0) == 0) o.ab_handoff_trick = std::stoi(parts[i].substr(2));
            else if (parts[i] == "noab") o.ab_handoff_trick = -1;
            else if (parts[i].rfind("rab", 0) == 0) o.rollout_ab_trick = std::stoi(parts[i].substr(3));
            else if (parts[i] == "norab") o.rollout_ab_trick = -1;
            else if (parts[i] == "nowt") o.weight_play_history = false;
            else if (parts[i] == "race") o.race = true;
            else if (parts[i] == "me") o.use_match_equity = true;
            else if (parts[i] == "auct") o.weight_auction = true;
            else if (parts[i] == "hauct") o.weight_fast_auction = true;
            else if (parts[i].rfind("at", 0) == 0) {
                o.weight_auction = true;
                o.auction_lik_temp = std::stod(parts[i].substr(2));
            } else if (parts[i].rfind("z", 0) == 0) {
                o.race = true;
                o.race_z = std::stod(parts[i].substr(1));
            } else if (parts[i].rfind("t", 0) == 0) {
                o.play_lik_temp = std::stod(parts[i].substr(1));
            } else throw std::invalid_argument("unknown hybrid option: " + parts[i]);
        }
        return std::make_unique<HybridAgent>(loadCached(parts[0]), o, spec, seed);
    }
    throw std::invalid_argument("unknown agent: " + spec);
}

}  // namespace belote
