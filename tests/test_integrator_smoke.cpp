// test_integrator_smoke.cpp — determinism / exercise of the FENE + flow(kappa)
// + Gauss-EV paths through the integrator at hstar=0. NOT a correctness oracle
// (the Fortran flow+EV fixtures all use HI); it locks reproducibility and makes
// sure those code paths run, pending Phase-2 bit-exact validation.
#include "../src/integrator.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace bdsim;

static std::vector<Vec3> run(bool with_ev, dp hstar = 0.0,
                             DelSMethod method = DelSMethod::Chebyshev) {
    const int N = 6;
    PhysParams phys;
    phys.number_of_beads = N;
    phys.spring.type = Spring::FENE;
    phys.spring.sqrtb = 50.0;
    phys.spring.natural_length = 0.0;
    phys.hstar = hstar;
    phys.hi_method = method;
    if (with_ev) { phys.ev.type = EV::Gauss; phys.ev.zstar = 1.0; phys.ev.dstar = 1.0; }
    Mat3 K{}; K[0][1] = 2.0;            // simple shear, rate 2
    phys.flow = Flow::constant(K);

    SimParams sim;
    sim.time_start = 0.0; sim.time_end = 0.5; sim.dt = 0.01;
    sim.implicit_loop_tol = 1e-6;

    std::vector<Vec3> R(N);
    for (int i = 0; i < N; ++i) R[i] = {0.3 * i, 0.1 * i, -0.2 * i};
    Rng rng(7); rng.reset(7);
    time_integrate_chain(R, phys, sim, rng);
    return R;
}

int main() {
    int failures = 0;
    auto a = run(false);
    auto b = run(false);
    for (size_t i = 0; i < a.size(); ++i)
        for (int d = 0; d < 3; ++d) if (a[i][d] != b[i][d]) ++failures;
    std::printf("FENE+shear deterministic: %s\n", failures == 0 ? "PASS" : "FAIL");

    auto c = run(true);   // just needs to run and stay finite
    int bad = 0;
    for (auto& v : c) for (int d = 0; d < 3; ++d) if (!std::isfinite(v[d])) ++bad;
    std::printf("FENE+shear+GaussEV finite: %s\n", bad == 0 ? "PASS" : "FAIL");
    failures += bad;

    // HI (hstar>0): Cholesky reproducible; both methods finite.
    auto ch1 = run(false, 0.15, DelSMethod::Cholesky);
    auto ch2 = run(false, 0.15, DelSMethod::Cholesky);
    int hidiff = 0;
    for (size_t i = 0; i < ch1.size(); ++i)
        for (int d = 0; d < 3; ++d) if (ch1[i][d] != ch2[i][d]) ++hidiff;
    std::printf("HI Cholesky deterministic: %s\n", hidiff == 0 ? "PASS" : "FAIL");
    failures += hidiff;

    auto cheb = run(false, 0.15, DelSMethod::Chebyshev);
    int hibad = 0;
    for (auto& v : cheb) for (int d = 0; d < 3; ++d) if (!std::isfinite(v[d])) ++hibad;
    std::printf("HI Chebyshev finite: %s\n", hibad == 0 ? "PASS" : "FAIL");
    failures += hibad;

    std::printf("\nIntegrator smoke: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
