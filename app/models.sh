#!/usr/bin/env bash
# Copies the Belot networks from this repository's checkpoints/ (gitignored)
# into the Android assets, under the names app/cpp/belot_api.cpp loads. The
# default set is the one measured at 54.35% +- 1.74 pp of 2000 games against
# Kostov's Master (runs/bg_bridge_e2e_bid_dbl_5mspec_1000p.log). Override the
# directories to ship other exports, e.g. SUIT=bg_models_suit_20m app/models.sh
set -euo pipefail
cd "$(dirname "$0")"
SRC="${CHECKPOINTS:-../checkpoints}"   # or another checkout's: CHECKPOINTS=/path/to/checkpoints
DEST=belot-app/src/main/assets/models
BID="${BID:-bg_models_bid_unmasked_dbl}"; SUIT="${SUIT:-bg_models_suit}"; NT="${NT:-bg_models_nt_latest}"; AT="${AT:-bg_models_at}"
mkdir -p "$DEST"
copy() { cp "$SRC/$1" "$DEST/$2"; echo "  $2 <- $1"; }
copy "$BID/bg_bid.bin" belot_bid.bin
for pair in suit:$SUIT nt:$NT at:$AT; do
    fam=${pair%%:*}; dir=${pair#*:}
    copy "$dir/bg_fast_policy.bin" "belot_${fam}_fast.bin"
    copy "$dir/bg_belief.bin"      "belot_${fam}_belief.bin"
    copy "$dir/bg_q_gru.bin"       "belot_${fam}_q.bin"
done
echo "Models in app/$DEST"
