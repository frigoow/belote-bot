#pragma once

// The rule flags of RULES.md ("Config flags"). Config::belgian() is the
// ruleset of belote-rebelote.be and the default; Config::wikibooks() is the
// nl.wikibooks profile the engine started from; Config::tanguyCompat()
// reproduces the classic-French engine in tpemeja/open_spiel
// (belote-cpp-rework), which the tests cross-check against.

namespace belote {

enum class TieRule {
    kHanging,     // defenders score, takers' points held over (op den boek)
    kTakersMake,  // a tie counts as made
    kTakersFail,  // a tie counts as nat (Tanguy's engine)
};

enum class ScoreMode {
    kComplement,  // defenders' total rounded to the nearest ten (5 up), in units;
                  // takers get the rest of the deal's units (16 + bonuses / 10)
    kUnits,       // each team's total rounded to the nearest ten, 5 up, in units
    kRaw,         // points recorded as-is
};

struct Config {
    // §1 / §2 trump choice
    bool redeal_next_dealer = true;        // false: same dealer deals once more first
    bool round2_allow_blik_suit = false;

    // §4 playing a card
    bool trump_lead_partner_must_overtrump = false;
    bool must_undertrump = true;

    // §5 declarations
    bool toon_enabled = true;               // false: no toon at all (Tanguy)
    int carre_low_points = 0;               // four 7s / four 8s (0 = not a toon)
    bool toon_optional = true;              // players may decline to declare
    int toon_show_trick = 3;                // trick in which toon is shown
    bool toon_losers_show = false;          // true: a beaten toon is shown too
    bool belot_queen_first_forfeits = false; // true: queen first forfeits the 20

    // §6 / §7 scoring and match
    TieRule tie_rule = TieRule::kHanging;
    ScoreMode score_mode = ScoreMode::kComplement;
    int ronde_points = 252;                 // card points when one team takes all 8 tricks
    int match_target = 101;
    int match_win_margin = 2;               // lead needed once past the target (0 = any lead)

    static Config belgian() { return Config{}; }

    static Config wikibooks() {
        Config c;
        c.belot_queen_first_forfeits = true;
        c.score_mode = ScoreMode::kUnits;
        c.match_win_margin = 0;
        return c;
    }

    static Config tanguyCompat() {
        Config c;
        c.match_win_margin = 0;
        c.trump_lead_partner_must_overtrump = true;
        c.toon_enabled = false;
        c.belot_queen_first_forfeits = false;
        c.tie_rule = TieRule::kTakersFail;
        c.score_mode = ScoreMode::kRaw;
        return c;
    }
};

}  // namespace belote
