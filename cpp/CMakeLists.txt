cmake_minimum_required(VERSION 3.15)
project(belote CXX)

# The Belgian Belote rules engine (RULES.md), the world sampler, ISMCTS,
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
endif()
