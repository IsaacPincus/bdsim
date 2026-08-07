// test_spring.cpp — the semi-implicit solver vs Fortran oracles.
#include "../src/spring.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

using bdsim::Spring;
using bdsim::solve_implicit_r;

static int checks = 0, failures = 0;
static void expect(double got, double want, double tol, const char* what) {
    ++checks;
    double d = std::fabs(got - want);
    if (d > tol) {
        std::printf("  [FAIL] %-42s got %.15g want %.15g (|d|=%.3e)\n", what, got, want, d);
        ++failures;
    }
}

int main() {
    const double td = 1e-6;
    using S = bdsim::SpringSolution;

    S a = solve_implicit_r(Spring::WLCbounded, 0.001 / 4.0, 5.0, 0.8);
    expect(a.length,       0.999354532073543,   td,       "WLCb g=5  s=0.8 length");
    expect(a.force_factor, 1.601291769648808e4, td * 1e4, "WLCb g=5  s=0.8 force");

    S b = solve_implicit_r(Spring::WLCbounded, 0.001 / 4.0, 0.0, 0.8);
    expect(b.length,       0.601489231399178,    td,       "WLCb g=0  s=0.8 length");
    expect(b.force_factor, -4.000000000000779e3, td * 1e4, "WLCb g=0  s=0.8 force");

    S c = solve_implicit_r(Spring::WLCbounded, 0.001 / 4.0, 1.0e5, 0.8);
    expect(c.length,       0.999995917496677,   td,       "WLCb g=1e5 s=0.8 length");
    expect(c.force_factor, 3.999976329984242e8, td * 1e8, "WLCb g=1e5 s=0.8 force");

    S d = solve_implicit_r(Spring::WLCbounded, 0.2 / 4.0,
                           2.0 * std::numeric_limits<double>::epsilon(), 0.0);
    expect(d.length, 0.0, td, "WLCb tiny-gama s=0 length");

    {   // WLC-bounded reduces to WLC when sigma = 0
        S wb = solve_implicit_r(Spring::WLCbounded, 0.2 / 4.0, 1.0, 0.0);
        S w  = solve_implicit_r(Spring::WLC,        0.2 / 4.0, 1.0, 0.0);
        expect(wb.length, w.length, td, "WLCb == WLC length (s=0)");
        expect(wb.force_factor, w.force_factor, td, "WLCb == WLC force  (s=0)");
    }

    {   // FENE length == FENE-Fraenkel length (natural length 0), swept grid
        int mism = 0;
        for (int i = 1; i <= 100; ++i)
            for (int j = 1; j <= 100; ++j) {
                double sqrtb = i * 0.1, gama = j * 0.5;
                double gstar = gama / sqrtb;
                S f  = solve_implicit_r(Spring::FENE,         0.05, gstar, 0.0);
                S ff = solve_implicit_r(Spring::FENEFraenkel, 0.05, gstar, 0.0);
                if (std::fabs(f.length - ff.length) > td) ++mism;
            }
        ++checks;
        if (mism) { std::printf("  [FAIL] FENE vs FF sweep: %d mismatches\n", mism); ++failures; }
    }

    std::printf("\nSpring solver: %d checks, %d failures -> %s\n",
                checks, failures, failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
