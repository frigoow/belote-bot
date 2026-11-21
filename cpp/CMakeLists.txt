cmake_minimum_required(VERSION 3.15)
project(belote CXX)

# The Belgian Belote rules engine (RULES.md) and its tests.
#   cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   ctest --test-dir build --output-on-failure
# Without CMake: cpp/run_tests.sh

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()

add_library(belote_core STATIC
    belote/card.cpp
    belote/toon.cpp
    belote/deal.cpp
    belote/match.cpp
)
target_include_directories(belote_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
    target_compile_options(belote_core PRIVATE /O2 /W4)
else()
    target_compile_options(belote_core PRIVATE -O3 -Wall -Wextra)
endif()

foreach(test rules random)
    add_executable(belote_test_${test} tests/test_${test}.cpp)
    target_link_libraries(belote_test_${test} PRIVATE belote_core)
    add_test(NAME ${test} COMMAND belote_test_${test})
endforeach()
