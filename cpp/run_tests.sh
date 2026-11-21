#!/usr/bin/env bash
# Builds the engine and runs its tests with plain g++ (no CMake needed).
# Usage: cpp/run_tests.sh [build-dir]   (default: cpp/build-sh)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/build-sh}"
mkdir -p "$out"
cxx="${CXX:-g++}"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -I "$here")
sources=("$here"/belote/*.cpp)

status=0
for test in rules random; do
    "$cxx" "${flags[@]}" "${sources[@]}" "$here/tests/test_$test.cpp" -o "$out/test_$test"
    "$out/test_$test" || status=1
done
exit $status
