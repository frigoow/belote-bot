#pragma once

#include "bg/view.h"

namespace bg {

// The acting seat's hand, declarations by type, and the public play only.
int rulePolicy(const Round& r);

// Natural call on the acting seat's initial five cards.
std::uint8_t naturalBid(const Round& r);

// Natural call with optional All-Trumps / No-Trumps opening bonuses (in tenths of
// a bidding unit) and optional suppression of Double/Redouble (for clean Manille-style
// single-hand specialist training).
std::uint8_t naturalBidTuned(const Round& r, int at_bonus, int nt_bonus = 0, bool allow_double = true);

// Log P(public auction | current sampled hands), omitting the observer's
// calls. Each seat used the same initial five cards throughout the auction.
double auctionLogLik(const PublicState& ps, const Hands& hands);

}  // namespace bg
