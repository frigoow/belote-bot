#pragma once

// The rule flags of RULES.md ("Config flags"). Config::belgian() is the house
// ruleset; Config::tanguyCompat() reproduces the classic-French engine in
// tpemeja/open_spiel (belote-cpp-rework), which the tests cross-check against.

namespace belote {

enum class TieRule {
    kHanging,     // defenders score, takers' points held over (op den boek)
    kTakersMake,  // a tie counts as made
    kTakersFail,  // a tie counts as nat (Tanguy's engine)
};

enum class ScoreMode {
    kUnits,  // totals rounded to the nearest ten, 5 up, written in units
    kRaw,    // points recorded as-is
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
    bool belot_queen_first_forfeits = true; // false: K/Q order doesn't matter

    // §6 / §7 scoring and match
    TieRule tie_rule = TieRule::kHanging;
    ScoreMode score_mode = ScoreMode::kUnits;
    int ronde_points = 252;                 // card points when one team takes all 8 tricks
    int match_target = 101;
    int match_win_margin = 0;               // 0: any lead wins once past the target

    static Config belgian() { return Config{}; }

    static Config tanguyCompat() {
        Config c;
        c.trump_lead_partner_must_overtrump = true;
        c.toon_enabled = false;
        c.belot_queen_first_forfeits = false;
        c.tie_rule = TieRule::kTakersFail;
        c.score_mode = ScoreMode::kRaw;
        return c;
    }
};

}  // namespace belote
