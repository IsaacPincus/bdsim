// test_flow.cpp — simplified flow: constant + linear interpolation + clamping.
#include "../src/flow.hpp"
#include <cmath>
#include <cstdio>

using namespace bdsim;
static int checks = 0, failures = 0;
static void expect(double got, double want, const char* what) {
    ++checks;
    if (std::fabs(got - want) > 1e-12) {
        std::printf("  [FAIL] %-24s got %.15g want %.15g\n", what, got, want);
        ++failures;
    }
}

int main() {
    // constant shear
    Mat3 S{}; S[0][1] = 2.5;
    Flow fc = Flow::constant(S);
    expect(fc.kappa(0.0)[0][1], 2.5, "const at t=0");
    expect(fc.kappa(9.9)[0][1], 2.5, "const at t=9.9");

    // time-varying uniaxial rate 0 -> 2 over t in [0,1]
    auto uni = [](dp g) { Mat3 K{}; K[0][0]=g; K[1][1]=-g/2; K[2][2]=-g/2; return K; };
    Flow ft = Flow::time_varying({0.0, 1.0}, {uni(0.0), uni(2.0)});
    expect(ft.kappa(0.0)[0][0], 0.0, "ramp at t=0");
    expect(ft.kappa(0.5)[0][0], 1.0, "ramp at t=0.5 (interp)");
    expect(ft.kappa(0.5)[1][1], -0.5, "ramp yy at t=0.5");
    expect(ft.kappa(1.0)[0][0], 2.0, "ramp at t=1");
    // clamp beyond ends
    expect(ft.kappa(-3.0)[0][0], 0.0, "clamp low");
    expect(ft.kappa(5.0)[0][0], 2.0, "clamp high");

    std::printf("\nFlow: %d checks, %d failures -> %s\n",
                checks, failures, failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
