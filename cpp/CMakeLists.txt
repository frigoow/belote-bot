cmake_minimum_required(VERSION 3.15)
project(belote CXX)

# The Belgian Belote rules engine (docs/RULES_BELGIAN.md), the world sampler, ISMCTS,
# baselines, the duplicate-deal duel and their tests.
#   cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   ctest --test-dir build --output-on-failure
# The Python module (belote_cpp) as well:
#   cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release -DBELOTE_PYTHON=ON \
#         -DPython_EXECUTABLE=$(which python) \
#         -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
# Without CMake: cpp/run_tests.sh

option(BELOTE_PYTHON "Build the belote_cpp Python module (needs pybind11)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()
enable_testing()
find_package(Threads REQUIRED)

if(MSVC)
    set(BELOTE_WARNINGS /W4)
else()
    set(BELOTE_WARNINGS -Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND BELOTE_WARNINGS -march=native)
    endif()
endif()

# The engine, what a player knows (observation) and the world sampler.
add_library(belote_core STATIC
    belote/card.cpp
    belote/toon.cpp
    belote/deal.cpp
    belote/match.cpp
    belote/observation.cpp
    belote/sampler.cpp
    belote/encoder.cpp
)
target_include_directories(belote_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(belote_core PRIVATE ${BELOTE_WARNINGS})

# ISMCTS (third_party/ismcsolver, MIT), endgame AB, rollouts, agents and harness.
add_library(belote_search STATIC
    search/belote_game.cpp
    search/bid_model.cpp
    search/endgame.cpp
    search/fast_policy.cpp
    search/ismcts.cpp
    search/rollout.cpp
    agents/agents.cpp
    agents/rule_agent.cpp
    agents/duel.cpp
    belote/batch_env.cpp
)
target_include_directories(belote_search SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ismcsolver/include)
target_link_libraries(belote_search PUBLIC belote_core Threads::Threads)
target_compile_options(belote_search PRIVATE ${BELOTE_WARNINGS})

add_executable(belote_duel tools/duel.cpp)
target_link_libraries(belote_duel PRIVATE belote_search)
add_executable(belote_rule_review tools/rule_review.cpp)
target_link_libraries(belote_rule_review PRIVATE belote_search)

# Bulgarian Bridge-Belot (Kostov's BelotGameEngine rules), cross-checked by test_bg.
add_library(belote_bg STATIC
    bg/bg.cpp
    bg/view.cpp
    bg/rule_policy.cpp
    bg/agent.cpp
    bg/encoder.cpp
    bg/batch_env.cpp
    bg/neural.cpp
    bg/match_equity.cpp
    bg/bid_model.cpp
)
# The game-independent bot code (network kernels, ...): cardbot-framework's C++,
# the git submodule at the repository root (git submodule update --init).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cardbot-framework/cpp ${CMAKE_BINARY_DIR}/cardbot)
target_link_libraries(belote_bg PUBLIC belote_core cardbot Threads::Threads)
target_compile_options(belote_bg PRIVATE ${BELOTE_WARNINGS})
add_executable(belote_test_bg tests/test_bg.cpp)
target_link_libraries(belote_test_bg PRIVATE belote_bg)
add_test(NAME bg COMMAND belote_test_bg)
add_executable(belote_test_bg_agent tests/test_bg_agent.cpp)
target_link_libraries(belote_test_bg_agent PRIVATE belote_bg)
add_test(NAME bg_agent COMMAND belote_test_bg_agent)
add_executable(belote_test_bg_rule_policy tests/test_bg_rule_policy.cpp)
target_link_libraries(belote_test_bg_rule_policy PRIVATE belote_bg)
add_test(NAME bg_rule_policy COMMAND belote_test_bg_rule_policy)
add_executable(bg_agent tools/bg_agent.cpp)
target_link_libraries(bg_agent PRIVATE belote_bg)
# Belot through cardbot-framework's agents and duels (bg/framework.h)
add_executable(bg_framework_duel tools/bg_framework_duel.cpp)
target_link_libraries(bg_framework_duel PRIVATE belote_bg)
add_executable(bg_duel tools/bg_duel.cpp)
target_link_libraries(bg_duel PRIVATE belote_bg)
add_executable(bg_collect_search tools/bg_collect_search.cpp)
target_link_libraries(bg_collect_search PRIVATE belote_bg)

foreach(test rules random sampler search rule_agent)
    add_executable(belote_test_${test} tests/test_${test}.cpp)
    target_link_libraries(belote_test_${test} PRIVATE belote_search)
    add_test(NAME ${test} COMMAND belote_test_${test})
endforeach()

if(BELOTE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(belote_cpp python/bindings.cpp)
    target_link_libraries(belote_cpp PRIVATE belote_search)
    pybind11_add_module(bg_cpp python/bg_bindings.cpp)
    target_link_libraries(bg_cpp PRIVATE belote_bg)
endif()
