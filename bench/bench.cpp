// bench.cpp — timing of the integrator across chain sizes and HI methods.
//
// Usage:  bench [seconds_per_case]     (default 0.5; use e.g. 20 for good stats)
//
// Reports, per case, the achieved rate as microseconds per timestep and
// nanoseconds per bead-timestep (the size-independent figure of merit).
#include "integrator.hpp"
#include "model.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace bdsim;
using clk = std::chrono::steady_clock;

static Vec3Field random_chain(int n, double bond, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<double> nd(0.0, bond);
    Vec3Field R(n);
    for (int i = 1; i < n; ++i)
        R[i] = {R[i-1][0] + nd(g), R[i-1][1] + nd(g), R[i-1][2] + nd(g)};
    return R;
}

struct Case { const char* label; int N; double hstar; DelSMethod method;
              Spring spring; double sqrtb; double natlen; double init_bond; };

// Integrate `nsteps` steps once; return elapsed seconds.
static double run_steps(const PhysParams& phys, double dt, long nsteps,
                        Vec3Field R, Rng& rng) {
    SimParams sim;
    sim.dt = dt;
    sim.time_start = 0.0;
    sim.time_end = nsteps * dt;
    sim.implicit_loop_tol = 1e-4;
    auto t0 = clk::now();
    time_integrate_chain(R, phys, sim, rng);
    auto t1 = clk::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const double budget = (argc > 1) ? std::atof(argv[1]) : 0.5;  // seconds per case
    const double dt = 0.01;

    // FENE (sqrtb 50), and a stiff FENE-Fraenkel (sqrtb 3, natural length 10).
    const Spring FE = Spring::FENE, FF = Spring::FENEFraenkel;
    const auto CH = DelSMethod::Cholesky;
    const auto CB = DelSMethod::Chebyshev;
    const std::vector<Case> cases = {
        {"FENE free",       10,  0.0, CH, FE, 50.0, 0.0,  1.0},
        {"FENE free",       50,  0.0, CH, FE, 50.0, 0.0,  1.0},
        {"FENE free",       100, 0.0, CH, FE, 50.0, 0.0,  1.0},
        {"FENE free",       200, 0.0, CH, FE, 50.0, 0.0,  1.0},
        {"FENE HI Chol",    50,  0.2, CH, FE, 50.0, 0.0,  1.0},
        {"FENE HI Cheb",    50,  0.2, CB, FE, 50.0, 0.0,  1.0},
        {"FENE HI Cheb",    100, 0.2, CB, FE, 50.0, 0.0,  1.0},
        {"stiff-FF free",   10,  0.0, CH, FF,  3.0, 10.0, 10.0},
        {"stiff-FF free",   50,  0.0, CH, FF,  3.0, 10.0, 10.0},
        {"stiff-FF free",   100, 0.0, CH, FF,  3.0, 10.0, 10.0},
        {"stiff-FF HI Cheb", 50, 0.2, CB, FF,  3.0, 10.0, 10.0},
    };

    std::printf("%-15s %5s %10s %12s %14s\n",
                "case", "N", "steps", "us/step", "ns/bead-step");
    std::printf("%s\n", std::string(60, '-').c_str());

    for (const auto& c : cases) {
        PhysParams phys;
        phys.number_of_beads = c.N;
        phys.spring.type = c.spring;
        phys.spring.sqrtb = c.sqrtb;
        phys.spring.natural_length = c.natlen;
        phys.hstar = c.hstar;
        phys.hi_method = c.method;

        Rng rng(12345);
        Vec3Field R = random_chain(c.N, c.init_bond, 7);

        // calibrate: how many steps fit in the budget?
        const long cal = 20;
        double tcal = run_steps(phys, dt, cal, R, rng);
        long nsteps = static_cast<long>(cal * budget / tcal);
        if (nsteps < cal) nsteps = cal;

        double secs = run_steps(phys, dt, nsteps, R, rng);
        double us_per_step = 1e6 * secs / nsteps;
        double ns_per_bead_step = 1e9 * secs / (double(nsteps) * c.N);
        std::printf("%-15s %5d %10ld %12.2f %14.2f\n",
                    c.label, c.N, nsteps, us_per_step, ns_per_bead_step);
    }
    return 0;
}
