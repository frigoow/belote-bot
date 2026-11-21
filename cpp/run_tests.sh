#!/usr/bin/env bash
# Builds the engine, sampler and search and runs their tests with plain g++
# (no CMake needed). Also builds the belote_duel tool.
# Usage: cpp/run_tests.sh [build-dir]   (default: cpp/build-sh)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/build-sh}"
mkdir -p "$out"
cxx="${CXX:-g++}"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -pthread -I "$here" -isystem "$here/third_party/ismcsolver/include")
sources=("$here"/belote/*.cpp "$here"/search/*.cpp "$here"/agents/*.cpp)

"$cxx" "${flags[@]}" "${sources[@]}" "$here/tools/duel.cpp" -o "$out/belote_duel"
status=0
for test in rules random sampler search; do
    "$cxx" "${flags[@]}" "${sources[@]}" "$here/tests/test_$test.cpp" -o "$out/test_$test"
    "$out/test_$test" || status=1
done
exit $status
