// belote_cpp: the C++ engine, sampler, search and harness for Python.
//
// Cards are ints 0..31 (suit * 8 + rank, ranks 7 8 9 10 J Q K A); card sets
// are 32-bit masks (ints), with mask_cards()/cards_mask() to convert. The
// engine stays the only implementation of the rules: Python drives it, it
// never re-implements legality or scoring.

#include "agents/agent.h"
#include "agents/duel.h"
#include "belote/batch_env.h"
#include "belote/card.h"
#include "belote/deal.h"
#include "belote/encoder.h"
#include "belote/match.h"
#include "belote/observation.h"
#include "belote/rules.h"
#include "belote/sampler.h"
#include "belote/toon.h"
#include "search/endgame.h"
#include "search/ismcts.h"
#include "search/rollout.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>

namespace py = pybind11;
using namespace belote;

namespace {

std::vector<int> maskCards(Mask m) {
    std::vector<int> v;
    for (; m; m &= m - 1) v.push_back(lowestCard(m));
    return v;
}

Mask cardsMask(const std::vector<int>& cards) {
    Mask m = 0;
    for (int c : cards) {
        if (c < 0 || c >= kCards) throw py::value_error("card out of range");
        m |= bit(c);
    }
    return m;
}

std::array<Mask, kPlayers> handsFrom(const std::vector<Mask>& hands) {
    if (hands.size() != kPlayers) throw py::value_error("need 4 hands");
    return {hands[0], hands[1], hands[2], hands[3]};
}

}  // namespace

PYBIND11_MODULE(belote_cpp, m) {
    m.doc() = "Belgian Belote engine, world sampler, ISMCTS and duplicate-deal harness";

    // ------------------------------------------------------------ cards
    m.attr("CARDS") = kCards;
    m.attr("PLAYERS") = kPlayers;
    m.attr("TRICKS") = kTricks;
    m.attr("NO_CARD") = kNoCard;
    m.attr("NO_SUIT") = kNoSuit;
    m.def("suit_of", &suitOf);
    m.def("rank_of", &rankOf);
    m.def("card_of", &cardOf, py::arg("suit"), py::arg("rank"));
    m.def("team_of", &teamOf);
    m.def("partner_of", &partnerOf);
    m.def("card_name", &cardName);
    m.def("parse_card", &parseCard);
    m.def("parse_cards", &parseCards, "Space-separated names -> mask");
    m.def("mask_string", &maskString);
    m.def("mask_cards", &maskCards, "Mask -> sorted list of cards");
    m.def("cards_mask", &cardsMask, "List of cards -> mask");
    m.def("card_points", &cardPoints, py::arg("card"), py::arg("trump"));
    m.def("card_strength", &cardStrength, py::arg("card"), py::arg("trump"));

    // ------------------------------------------------------------ rules
    py::enum_<TieRule>(m, "TieRule")
        .value("HANGING", TieRule::kHanging)
        .value("TAKERS_MAKE", TieRule::kTakersMake)
        .value("TAKERS_FAIL", TieRule::kTakersFail);
    py::enum_<ScoreMode>(m, "ScoreMode")
        .value("COMPLEMENT", ScoreMode::kComplement)
        .value("UNITS", ScoreMode::kUnits)
        .value("RAW", ScoreMode::kRaw);
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_static("belgian", &Config::belgian)
        .def_static("wikibooks", &Config::wikibooks)
        .def_static("tanguy_compat", &Config::tanguyCompat)
        .def_readwrite("redeal_next_dealer", &Config::redeal_next_dealer)
        .def_readwrite("round2_allow_blik_suit", &Config::round2_allow_blik_suit)
        .def_readwrite("trump_lead_partner_must_overtrump", &Config::trump_lead_partner_must_overtrump)
        .def_readwrite("must_undertrump", &Config::must_undertrump)
        .def_readwrite("toon_enabled", &Config::toon_enabled)
        .def_readwrite("carre_low_points", &Config::carre_low_points)
        .def_readwrite("toon_optional", &Config::toon_optional)
        .def_readwrite("toon_show_trick", &Config::toon_show_trick)
        .def_readwrite("toon_losers_show", &Config::toon_losers_show)
        .def_readwrite("belot_queen_first_forfeits", &Config::belot_queen_first_forfeits)
        .def_readwrite("tie_rule", &Config::tie_rule)
        .def_readwrite("score_mode", &Config::score_mode)
        .def_readwrite("ronde_points", &Config::ronde_points)
        .def_readwrite("match_target", &Config::match_target)
        .def_readwrite("match_win_margin", &Config::match_win_margin);

    // ------------------------------------------------------------ toon
    py::enum_<ToonKind>(m, "ToonKind")
        .value("DERDE", ToonKind::kDerde)
        .value("VIERDE", ToonKind::kVierde)
        .value("VIJFDE", ToonKind::kVijfde)
        .value("CARRE", ToonKind::kCarre);
    py::class_<Toon>(m, "Toon")
        .def_readonly("carre", &Toon::carre)
        .def_readonly("suit", &Toon::suit)
        .def_readonly("rank", &Toon::rank)
        .def_readonly("length", &Toon::length)
        .def_readonly("cards", &Toon::cards)
        .def_readonly("points", &Toon::points)
        .def_property_readonly("kind", [](const Toon& t) { return kindOf(t); });
    py::class_<Announcement>(m, "Announcement")
        .def_readonly("kind", &Announcement::kind)
        .def_readonly("carre_rank", &Announcement::carre_rank);
    m.def("find_toons", &findToons, py::arg("hand"), py::arg("config"));
    m.def("announcement_of", &announcementOf, py::arg("toons"), py::arg("trump"));

    // ------------------------------------------------------------ deal
    py::enum_<Phase>(m, "Phase")
        .value("BID1", Phase::kBid1)
        .value("BID2", Phase::kBid2)
        .value("PLAY", Phase::kPlay)
        .value("DONE", Phase::kDone);
    py::enum_<Outcome>(m, "Outcome")
        .value("NONE", Outcome::kNone)
        .value("ALL_PASSED", Outcome::kAllPassed)
        .value("MADE", Outcome::kMade)
        .value("NAT", Outcome::kNat)
        .value("HANGING", Outcome::kHanging);

    py::class_<Bid> bid(m, "Bid");
    py::enum_<Bid::Kind>(bid, "Kind").value("PASS", Bid::kPass).value("TAKE", Bid::kTake).value("CHOOSE", Bid::kChoose);
    bid.def_static("pass_", &Bid::pass)
        .def_static("take", &Bid::take)
        .def_static("choose", &Bid::choose, py::arg("suit"))
        .def_readonly("kind", &Bid::kind)
        .def_readonly("suit", &Bid::suit)
        .def("__eq__", &Bid::operator==)
        .def("__repr__", [](const Bid& b) {
            if (b.kind == Bid::kPass) return std::string("Bid.pass_()");
            if (b.kind == Bid::kTake) return std::string("Bid.take()");
            return "Bid.choose(" + std::to_string(b.suit) + ")";
        });

    py::class_<Trick>(m, "Trick")
        .def(py::init<>())
        .def_readwrite("cards", &Trick::cards)
        .def_readwrite("players", &Trick::players)
        .def_readwrite("n", &Trick::n);
    m.def("trick_winner_index", &trickWinnerIndex, py::arg("trick"), py::arg("trump"));
    m.def("legal_mask", &legalMask, py::arg("hand"), py::arg("trick"), py::arg("trump"), py::arg("player"),
          py::arg("config"));

    py::class_<DealResult>(m, "DealResult")
        .def_readonly("outcome", &DealResult::outcome)
        .def_readonly("takers", &DealResult::takers)
        .def_readonly("trump", &DealResult::trump)
        .def_readonly("card_points", &DealResult::card_points)
        .def_readonly("toon_points", &DealResult::toon_points)
        .def_readonly("belot_points", &DealResult::belot_points)
        .def_readonly("totals", &DealResult::totals)
        .def_readonly("credited", &DealResult::credited)
        .def_readonly("held", &DealResult::held)
        .def_readonly("toon_team", &DealResult::toon_team)
        .def_readonly("belot_team", &DealResult::belot_team)
        .def_readonly("ronde_team", &DealResult::ronde_team);

    py::class_<Deal>(m, "Deal")
        .def(py::init<const Config&, int, const std::array<int, kCards>&>(), py::arg("config"), py::arg("dealer"),
             py::arg("deck"))
        .def_static(
            "from_play",
            [](const Config& cfg, int dealer, int taker, int trump, const std::vector<Mask>& hands) {
                return Deal::fromPlay(cfg, dealer, taker, trump, handsFrom(hands));
            },
            py::arg("config"), py::arg("dealer"), py::arg("taker"), py::arg("trump"), py::arg("hands"))
        .def("copy", [](const Deal& d) { return Deal(d); })
        .def_property_readonly("config", &Deal::config)
        .def_property_readonly("phase", &Deal::phase)
        .def_property_readonly("dealer", &Deal::dealer)
        .def_property_readonly("current_player", &Deal::currentPlayer)
        .def("hand", &Deal::hand)
        .def_property_readonly("blik", &Deal::blik)
        .def("legal_bids", &Deal::legalBids)
        .def("bid", &Deal::bid)
        .def_property_readonly("taker", &Deal::taker)
        .def_property_readonly("takers", &Deal::takers)
        .def_property_readonly("trump", &Deal::trump)
        .def_property_readonly("bids_made", &Deal::bidsMade)
        .def("toons", &Deal::toons)
        .def("can_set_declare", &Deal::canSetDeclare)
        .def("set_declare", &Deal::setDeclare)
        .def("declares", &Deal::declares)
        .def("announced", &Deal::announced)
        .def("announcement", &Deal::announcement)
        .def("shown", &Deal::shown)
        .def_property_readonly("belot_holder", &Deal::belotHolder)
        .def_property_readonly("belot_announced", &Deal::belotAnnounced)
        .def_property_readonly("belot_announcer", &Deal::belotAnnouncer)
        .def_property_readonly("belot_card", &Deal::belotCard)
        .def("legal_moves", &Deal::legalMoves, "Mask of legal cards")
        .def("legal_cards", [](const Deal& d) { return maskCards(d.legalMoves()); })
        .def("play", &Deal::play)
        .def_property_readonly("trick", &Deal::trick)
        .def_property_readonly("tricks_played", &Deal::tricksPlayed)
        .def("completed_trick", &Deal::completedTrick)
        .def("trick_winner", &Deal::trickWinner)
        .def_property_readonly("card_points", &Deal::cardPoints)
        .def_property_readonly("result", &Deal::result);

    // ------------------------------------------------------------ match
    py::class_<RecordedDeal>(m, "RecordedDeal")
        .def_readonly("scores", &RecordedDeal::scores)
        .def_readonly("held", &RecordedDeal::held);
    m.def("recorded_points", &recordedPoints, py::arg("raw"), py::arg("config"));
    m.def("record_deal", &recordDeal, py::arg("result"), py::arg("config"));
    py::class_<Match>(m, "Match")
        .def(py::init<const Config&, int>(), py::arg("config"), py::arg("first_dealer") = 0)
        .def_property_readonly("config", &Match::config)
        .def_property_readonly("dealer", &Match::dealer)
        .def_property_readonly("scores", &Match::scores)
        .def_property_readonly("pot", &Match::pot)
        .def_property_readonly("finished", &Match::finished)
        .def_property_readonly("winner", &Match::winner)
        .def_property_readonly("deals_recorded", &Match::dealsRecorded)
        .def("record", &Match::record);

    // ------------------------------------------------------------ observation / sampler
    py::class_<Play>(m, "Play").def_readonly("player", &Play::player).def_readonly("card", &Play::card);
    py::class_<Observation>(m, "Observation")
        .def_readonly("config", &Observation::cfg)
        .def_readonly("observer", &Observation::observer)
        .def_readonly("dealer", &Observation::dealer)
        .def_readonly("taker", &Observation::taker)
        .def_readonly("trump", &Observation::trump)
        .def_readonly("blik", &Observation::blik)
        .def_readonly("bids_made", &Observation::bids_made)
        .def_readonly("hand", &Observation::hand)
        .def_readonly("declare_self", &Observation::declare_self)
        .def_readonly("plays", &Observation::plays)
        .def_readonly("announced", &Observation::announced)
        .def_readonly("announcement", &Observation::announcement)
        .def_readonly("shown", &Observation::shown)
        .def_readonly("shown_cards", &Observation::shown_cards)
        .def_readonly("shown_points", &Observation::shown_points)
        .def_readonly("belot_announcer", &Observation::belot_announcer)
        .def_readonly("belot_card", &Observation::belot_card)
        .def("played", &Observation::played)
        .def("played_by", &Observation::playedBy)
        .def("hand_size", &Observation::handSize)
        .def("current_player", &Observation::currentPlayer);
    m.def("observe", &observe, py::arg("deal"), py::arg("observer"));
    py::class_<Constraints>(m, "Constraints")
        .def_readonly("excluded", &Constraints::excluded)
        .def_readonly("pinned", &Constraints::pinned);
    m.def("infer_constraints", &inferConstraints, py::arg("observation"));

    // ------------------------------------------------------------ encoder
    m.attr("ENCODER_VERSION") = kEncoderVersion;
    m.attr("STATE_DIM") = kStateDim;
    m.attr("PLY_DIM") = kPlyDim;
    m.attr("MAX_PLIES") = kMaxPlies;
    m.attr("ACTION_DIM") = kActionDim;
    m.def("canon_suit", &canonSuit, py::arg("suit"), py::arg("trump"));
    m.def("raw_suit", &rawSuit, py::arg("csuit"), py::arg("trump"));
    m.def("canon_card", &canonCard, py::arg("card"), py::arg("trump"));
    m.def("raw_card", &rawCard, py::arg("ccard"), py::arg("trump"));
    m.def("rel_seat", &relSeat, py::arg("player"), py::arg("observer"));
    m.def(
        "encode_state",
        [](const Observation& obs) {
            py::array_t<float> arr(kStateDim);
            encodeState(obs, arr.mutable_data());
            return arr;
        },
        py::arg("observation"));
    m.def(
        "encode_history",
        [](const Observation& obs) {
            py::array_t<float> arr({kMaxPlies, kPlyDim});
            const int n = encodeHistory(obs, arr.mutable_data());
            return py::make_tuple(arr, n);
        },
        py::arg("observation"));
    m.def(
        "encode_action",
        [](const Observation& obs, int card) {
            py::array_t<float> arr(kActionDim);
            encodeAction(obs, card, arr.mutable_data());
            return arr;
        },
        py::arg("observation"), py::arg("card"));
    m.def(
        "encode_observation",
        [](const Observation& obs) {
            py::array_t<float> state(kStateDim);
            py::array_t<float> hist({kMaxPlies, kPlyDim});
            encodeState(obs, state.mutable_data());
            const int n = encodeHistory(obs, hist.mutable_data());
            return py::make_tuple(state, hist, n);
        },
        py::arg("observation"));

    // ------------------------------------------------------------ batched self-play env
    m.attr("MAX_LEGAL") = kMaxLegal;
    py::class_<BatchPlayEnv>(m, "BatchPlayEnv")
        .def(py::init<int, const Config&, std::uint64_t, double, double>(), py::arg("num_envs"),
             py::arg("config") = Config::belgian(), py::arg("seed") = 1, py::arg("bid_threshold") = 5.0,
             py::arg("bid_jitter") = 0.8)
        .def_property_readonly("num_envs", &BatchPlayEnv::numEnvs)
        .def("reset_all", &BatchPlayEnv::resetAll, py::arg("seed"))
        .def("deal", &BatchPlayEnv::deal, py::arg("env"), py::return_value_policy::reference_internal)
        .def(
            "observe_batch",
            [](const BatchPlayEnv& env, bool include_history, bool include_beliefs) {
                const int B = env.numEnvs();
                const int hb = include_history ? B : 0;
                const int bb = include_beliefs ? B : 0;
                py::array_t<float> state(std::vector<py::ssize_t>{B, kStateDim});
                py::array_t<float> hist(std::vector<py::ssize_t>{hb, kMaxPlies, kPlyDim});
                py::array_t<std::int32_t> n_plies(B);
                py::array_t<float> legal_actions(std::vector<py::ssize_t>{B, kMaxLegal, kActionDim});
                py::array_t<std::int32_t> legal_cards(std::vector<py::ssize_t>{B, kMaxLegal});
                py::array_t<std::int32_t> n_legal(B);
                py::array_t<std::int32_t> current_team(B);
                py::array_t<std::int8_t> belief_targets(std::vector<py::ssize_t>{bb, kCards});
                py::array_t<std::uint8_t> belief_allowed(std::vector<py::ssize_t>{bb, kCards, 3});
                float* s_ptr = state.mutable_data();
                float* h_ptr = include_history ? hist.mutable_data() : nullptr;
                std::int32_t* np_ptr = n_plies.mutable_data();
                float* la_ptr = legal_actions.mutable_data();
                std::int32_t* lc_ptr = legal_cards.mutable_data();
                std::int32_t* nl_ptr = n_legal.mutable_data();
                std::int32_t* ct_ptr = current_team.mutable_data();
                std::int8_t* bt_ptr = include_beliefs ? belief_targets.mutable_data() : nullptr;
                std::uint8_t* ba_ptr = include_beliefs ? belief_allowed.mutable_data() : nullptr;
                {
                    py::gil_scoped_release release;
                    env.observeBatch(s_ptr, h_ptr, np_ptr, la_ptr, lc_ptr, nl_ptr, ct_ptr, bt_ptr, ba_ptr);
                }
                py::dict out;
                out["state"] = state;
                if (include_history) out["history"] = hist;
                out["n_plies"] = n_plies;
                out["legal_actions"] = legal_actions;
                out["legal_cards"] = legal_cards;
                out["n_legal"] = n_legal;
                out["current_team"] = current_team;
                if (include_beliefs) {
                    out["belief_targets"] = belief_targets;
                    out["belief_allowed"] = belief_allowed;
                }
                return out;
            },
            py::arg("include_history") = true, py::arg("include_beliefs") = true)
        .def(
            "step_batch",
            [](BatchPlayEnv& env, py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> action_idx) {
                const int B = env.numEnvs();
                if (action_idx.ndim() != 1 || action_idx.shape(0) != B)
                    throw py::value_error("action_idx must have shape (num_envs,)");
                py::array_t<std::uint8_t> done(B);
                py::array_t<float> reward_t0(B);
                const std::int32_t* act_ptr = action_idx.data();
                std::uint8_t* done_ptr = done.mutable_data();
                float* rew_ptr = reward_t0.mutable_data();
                {
                    py::gil_scoped_release release;
                    env.stepBatch(act_ptr, done_ptr, rew_ptr);
                }
                return py::make_tuple(done, reward_t0);
            },
            py::arg("action_idx"));

    py::class_<World>(m, "World")
        .def(py::init<>())
        .def(py::init([](const std::vector<Mask>& hands) { return World{handsFrom(hands)}; }))
        .def_readwrite("hands", &World::hands);
    py::class_<WorldSampler>(m, "WorldSampler")
        .def(py::init<const Observation&>(), py::arg("observation"))
        .def_property_readonly("observation", &WorldSampler::observation)
        .def("count", &WorldSampler::count)
        .def("admits", &WorldSampler::admits)
        .def("needs_check", &WorldSampler::needsCheck)
        .def("consistent", &WorldSampler::consistent)
        .def(
            "sample",
            [](const WorldSampler& s, std::uint64_t seed, int n, int max_tries) {
                std::mt19937_64 rng(seed);
                std::vector<World> out;
                for (int i = 0; i < n; ++i)
                    if (auto w = s.sample(rng, max_tries)) out.push_back(*w);
                return out;
            },
            py::arg("seed"), py::arg("n") = 1, py::arg("max_tries") = 2000,
            "n worlds drawn uniformly from the consistent worlds (fewer if rejection gives up)")
        .def("enumerate", &WorldSampler::enumerate, py::arg("max_worlds"))
        .def("replay", &WorldSampler::replay);

    // ------------------------------------------------------------ search
    py::class_<IsmctsOptions>(m, "IsmctsOptions")
        .def(py::init([](unsigned iterations, unsigned threads) { return IsmctsOptions{iterations, threads}; }),
             py::arg("iterations") = 1000, py::arg("threads") = 1)
        .def_readwrite("iterations", &IsmctsOptions::iterations)
        .def_readwrite("threads", &IsmctsOptions::threads);
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("best_move", &SearchResult::best_move)
        .def_readonly("visits", &SearchResult::visits)
        .def_readonly("total_visits", &SearchResult::total_visits)
        .def_readonly("value", &SearchResult::value);
    m.def("ismcts_search", &ismctsSearch, py::arg("deal"), py::arg("options") = IsmctsOptions{},
          py::call_guard<py::gil_scoped_release>());

    py::enum_<EndgameObjective>(m, "EndgameObjective")
        .value("UNITS", EndgameObjective::kUnits)
        .value("RAW", EndgameObjective::kRaw);
    py::class_<EndgameOptions>(m, "EndgameOptions")
        .def(py::init([](EndgameObjective objective, int tt_log2) { return EndgameOptions{objective, tt_log2}; }),
             py::arg("objective") = EndgameObjective::kUnits, py::arg("tt_log2") = 18)
        .def_readwrite("objective", &EndgameOptions::objective)
        .def_readwrite("tt_log2", &EndgameOptions::tt_log2);
    py::class_<EndgameResult>(m, "EndgameResult")
        .def_readonly("best_move", &EndgameResult::best_move)
        .def_readonly("value", &EndgameResult::value)
        .def_readonly("move_values", &EndgameResult::move_values)
        .def_readonly("nodes", &EndgameResult::nodes);
    m.def("solve_endgame", &solveEndgame, py::arg("deal"), py::arg("options") = EndgameOptions{},
          py::call_guard<py::gil_scoped_release>());

    py::class_<RolloutOptions>(m, "RolloutOptions")
        .def(py::init([](int worlds, int ab_handoff_trick, std::uint64_t seed) {
                 return RolloutOptions{worlds, ab_handoff_trick, seed};
             }),
             py::arg("worlds") = 32, py::arg("ab_handoff_trick") = -1, py::arg("seed") = 0)
        .def_readwrite("worlds", &RolloutOptions::worlds)
        .def_readwrite("ab_handoff_trick", &RolloutOptions::ab_handoff_trick)
        .def_readwrite("seed", &RolloutOptions::seed);
    py::class_<RolloutResult>(m, "RolloutResult")
        .def_readonly("best_move", &RolloutResult::best_move)
        .def_readonly("value", &RolloutResult::value)
        .def_readonly("worlds_used", &RolloutResult::worlds_used)
        .def_readonly("exact_enumeration", &RolloutResult::exact_enumeration)
        .def_readonly("move_values", &RolloutResult::move_values);
    m.def("rollout_search", &rolloutSearch, py::arg("deal"), py::arg("options") = RolloutOptions{},
          py::call_guard<py::gil_scoped_release>());

    // ------------------------------------------------------------ agents / harness
    py::class_<Agent>(m, "Agent")
        .def_property_readonly("name", &Agent::name)
        .def("choose_bid", &Agent::chooseBid)
        .def("choose_declare", &Agent::chooseDeclare)
        .def("choose_card", &Agent::chooseCard, py::call_guard<py::gil_scoped_release>());
    m.def("make_agent", &makeAgent, py::arg("spec"), py::arg("seed") = 0,
          "random | rule | ismcts:<iterations>[:<threads>] | rollout:<worlds>[:ab<trick>]");
    m.def("rule_hand_value", &ruleHandValue, py::arg("cards"), py::arg("trump"));
    m.def("rule_bid", &ruleBid, py::arg("deal"), py::arg("threshold") = 5.0);
    m.def("rule_choose_card", &ruleChooseCard, py::arg("deal"));
    m.def("config_summary", &configSummary, py::arg("config"));

    py::class_<DuelOptions>(m, "DuelOptions")
        .def(py::init<>())
        .def_readwrite("a", &DuelOptions::a)
        .def_readwrite("b", &DuelOptions::b)
        .def_readwrite("bidder", &DuelOptions::bidder)
        .def_readwrite("player", &DuelOptions::player)
        .def_readwrite("partner", &DuelOptions::partner)
        .def_readwrite("deals", &DuelOptions::deals)
        .def_readwrite("seed", &DuelOptions::seed)
        .def_readwrite("threads", &DuelOptions::threads)
        .def_readwrite("config", &DuelOptions::config);
    py::class_<DuelResult>(m, "DuelResult")
        .def_readonly("decks", &DuelResult::decks)
        .def_readonly("skipped", &DuelResult::skipped)
        .def_readonly("units", &DuelResult::units)
        .def_readonly("units_se", &DuelResult::units_se)
        .def_readonly("raw", &DuelResult::raw)
        .def_readonly("raw_se", &DuelResult::raw_se)
        .def_readonly("ms_per_card_a", &DuelResult::ms_per_card_a)
        .def_readonly("ms_per_card_b", &DuelResult::ms_per_card_b)
        .def_readonly("seconds", &DuelResult::seconds)
        .def_readonly("config_summary", &DuelResult::config_summary);
    m.def("run_duel", &runDuel, py::arg("options"), py::call_guard<py::gil_scoped_release>());

    py::class_<MatchDuelOptions>(m, "MatchDuelOptions")
        .def(py::init<>())
        .def_readwrite("a", &MatchDuelOptions::a)
        .def_readwrite("b", &MatchDuelOptions::b)
        .def_readwrite("bidder", &MatchDuelOptions::bidder)
        .def_readwrite("player", &MatchDuelOptions::player)
        .def_readwrite("partner", &MatchDuelOptions::partner)
        .def_readwrite("matches", &MatchDuelOptions::matches)
        .def_readwrite("seed", &MatchDuelOptions::seed)
        .def_readwrite("threads", &MatchDuelOptions::threads)
        .def_readwrite("config", &MatchDuelOptions::config);
    py::class_<MatchDuelResult>(m, "MatchDuelResult")
        .def_readonly("pairs", &MatchDuelResult::pairs)
        .def_readonly("win_rate", &MatchDuelResult::win_rate)
        .def_readonly("win_rate_se", &MatchDuelResult::win_rate_se)
        .def_readonly("score_margin", &MatchDuelResult::score_margin)
        .def_readonly("score_margin_se", &MatchDuelResult::score_margin_se)
        .def_readonly("mean_deals", &MatchDuelResult::mean_deals)
        .def_readonly("ms_per_card_a", &MatchDuelResult::ms_per_card_a)
        .def_readonly("ms_per_card_b", &MatchDuelResult::ms_per_card_b)
        .def_readonly("seconds", &MatchDuelResult::seconds)
        .def_readonly("config_summary", &MatchDuelResult::config_summary);
    m.def("run_match_duel", &runMatchDuel, py::arg("options"), py::call_guard<py::gil_scoped_release>());
}
