// sens_equiv.cpp — the exact run in inputc.dat, for a like-for-like timing
// against the Fortran ./sens:
//   N=51 FENE-Fraenkel (sqrtb=14.16, sigma=0), shear rate 1, HI hstar=0.2
//   (Chebyshev), dt=0.001, t=0..1 (1000 steps), 2 trajectories, tol=1e-5.
#include "integrator.hpp"
#include "model.hpp"

#include <chrono>
#include <cstdio>
#include <random>

using namespace bdsim;

int main() {
    PhysParams phys;
    phys.number_of_beads = 51;
    phys.spring.type = Spring::FENEFraenkel;
    phys.spring.sqrtb = 14.16;
    phys.spring.natural_length = 0.0;
    phys.hstar = 0.2;
    phys.hi_method = DelSMethod::Chebyshev;
    phys.ncheb_multiplier = 1.0;
    phys.fd_err_max = 0.0025;
    Mat3 K{}; K[0][1] = 1.0;                 // shear, gdots = 1
    phys.flow = Flow::constant(K);

    SimParams sim;
    sim.time_start = 0.0;
    sim.time_end = 1.0;                        // 1000 steps at dt=0.001
    sim.dt = 0.001;
    sim.implicit_loop_tol = 1e-5;
    sim.update_center_of_mass = true;

    const int n_traj = 2;
    std::mt19937 g(1);
    std::normal_distribution<double> nd(0.0, 1.0);

    Rng rng(12345);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_traj; ++t) {
        Vec3Field R(51);                       // random-walk initial config
        for (int i = 1; i < 51; ++i)
            R[i] = {R[i-1][0] + nd(g), R[i-1][1] + nd(g), R[i-1][2] + nd(g)};
        time_integrate_chain(R, phys, sim, rng);
        std::printf("  traj %d done, bead[25] = (%.4f, %.4f, %.4f)\n",
                    t, R[25][0], R[25][1], R[25][2]);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("2 trajectories x 1000 steps, N=51, HI Chebyshev: %.3f s wall\n", secs);
    return 0;
}
