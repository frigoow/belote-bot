#pragma once

// A minimal test harness: CHECK records a failure and carries on, so one run
// reports every broken rule. main() returns checkResult().

#include <cstdio>

namespace check {
inline int& failures() { static int n = 0; return n; }
inline int& checks() { static int n = 0; return n; }
}  // namespace check

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++check::checks();                                                       \
        if (!(cond)) {                                                           \
            ++check::failures();                                                 \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        ++check::checks();                                                       \
        const auto check_a_ = (a);                                               \
        const auto check_b_ = (b);                                               \
        if (!(check_a_ == check_b_)) {                                           \
            ++check::failures();                                                 \
            std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s == %s (%lld vs %lld)\n", \
                         __FILE__, __LINE__, #a, #b, (long long)check_a_,        \
                         (long long)check_b_);                                   \
        }                                                                        \
    } while (0)

inline int checkResult(const char* name) {
    std::printf("%s: %d checks, %d failed\n", name, check::checks(), check::failures());
    return check::failures() == 0 ? 0 : 1;
}
