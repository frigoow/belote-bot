#!/usr/bin/env bash
# Downloads the Belot networks the releases ship (belot-models-vN of this
# repository: private, so gh must be logged in, or GH_TOKEN set in CI) into
# the Android assets. app/models.sh copies them from checkpoints/ instead.
#   app/fetch_models.sh                                   # belot-models-v1
#   BELOT_MODELS_TAG=belot-models-v2 app/fetch_models.sh
set -euo pipefail
cd "$(dirname "$0")"
TAG="${BELOT_MODELS_TAG:-belot-models-v1}"
DEST=belot-app/src/main/assets/models
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
gh release download "$TAG" --repo frigoow/belote-bot --pattern belot-app-models.zip --dir "$TMP"
mkdir -p "$DEST"
unzip -o -q "$TMP/belot-app-models.zip" -d "$DEST"
echo "Models in app/$DEST:"; ls -1 "$DEST"
