// test_rng.cpp — validate the RNG against the Fortran oracle.
//
// Oracle values are lifted directly from tests.f90::test_random_number_sequence
// (seed = 10, n = 10). If these pass to 1e-6, the generator is bit-faithful and
// every downstream regression test that depends on the random stream is trustworthy.
#include "../src/rng.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using bdsim::Rng;

    // ---- Test 1: known sequence for seed = 10 ----
    const double expected[10] = {
        0.226473569869995,      0.357589989900589,      0.217218622565269,
        0.688211202621460,      8.224623650312424e-002, 0.238840281963348,
        0.956106066703796,      0.993510127067566,      0.167083948850632,
        0.341053307056427};
    const double tol = 1e-6;

    Rng rng(10);
    double x[10];
    rng.ran_1(10, x);

    int failures = 0;
    for (int i = 0; i < 10; ++i) {
        double diff = std::fabs(x[i] - expected[i]);
        if (diff > tol) {
            std::printf("  [FAIL] i=%d  got %.15f  expected %.15f  (|d|=%.3e)\n",
                        i, x[i], expected[i], diff);
            ++failures;
        }
    }
    std::printf("Test 1 (known sequence, seed=10): %s\n",
                failures == 0 ? "PASS" : "FAIL");

    // ---- Test 2: same seed reproduces the same stream ----
    Rng a(10), b(10);
    double xa[10], xb[10];
    a.ran_1(10, xa);
    b.ran_1(10, xb);
    int mism = 0;
    for (int i = 0; i < 10; ++i)
        if (xa[i] != xb[i]) ++mism;
    std::printf("Test 2 (reproducible for same seed): %s\n",
                mism == 0 ? "PASS" : "FAIL");
    failures += mism;

    // ---- Test 3: save/restore replays an identical stream (variance reduction) ----
    Rng c(10);
    double warm[3];
    c.ran_1(3, warm);           // advance
    Rng::State snap = c.save();
    double s1[5], s2[5];
    c.ran_1(5, s1);
    c.restore(snap);
    c.ran_1(5, s2);
    int vr = 0;
    for (int i = 0; i < 5; ++i)
        if (s1[i] != s2[i]) ++vr;
    std::printf("Test 3 (save/restore replay): %s\n", vr == 0 ? "PASS" : "FAIL");
    failures += vr;

    std::printf("\n%s\n", failures == 0 ? "ALL RNG TESTS PASSED" : "RNG TESTS FAILED");
    return failures == 0 ? 0 : 1;
}
