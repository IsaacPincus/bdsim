// test_external_force.cpp — user-applied bead forces.
//
// Checks the bookkeeping (which bead, what value, time interpolation, negative
// indexing, zero net force for a stretch) and then the physics: a Hookean dumbbell
// pulled by equal and opposite forces has an exactly known mean extension, so the
// applied force can be verified through the integrator rather than just at the API.
#include "../src/external_force.hpp"
#include "../src/integrator.hpp"

#include <cmath>
#include <cstdio>

using namespace bdsim;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%-52s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    // ---- constant force on one bead ----
    {
        ExternalForce ef;
        ef.add_constant(2, Vec3{1.0, -2.0, 0.5});
        const Vec3Field F = ef.force(5, 0.0);
        bool ok = std::fabs(F[2][0] - 1.0) < 1e-15 && std::fabs(F[2][1] + 2.0) < 1e-15
               && std::fabs(F[2][2] - 0.5) < 1e-15;
        for (int i = 0; i < 5; ++i)
            if (i != 2) ok = ok && norm2(F[i]) == 0.0;
        check(ok, "constant force lands on the right bead only");
    }

    // ---- negative index counts from the end ----
    {
        ExternalForce ef;
        ef.add_constant(-1, Vec3{3.0, 0.0, 0.0});
        const Vec3Field F = ef.force(4, 0.0);
        check(std::fabs(F[3][0] - 3.0) < 1e-15 && norm2(F[0]) == 0.0,
              "negative bead index addresses the last bead");
    }

    // ---- time interpolation, clamped outside the table ----
    {
        ExternalForce ef;
        ef.add_time_varying(0, {0.0, 1.0, 2.0},
                            {Vec3{0, 0, 0}, Vec3{10, 0, 0}, Vec3{20, 0, 0}});
        const dp a = ef.force(2, 0.5)[0][0];      // halfway to 10
        const dp b = ef.force(2, 1.5)[0][0];      // halfway 10 -> 20
        const dp lo = ef.force(2, -5.0)[0][0];    // clamped low
        const dp hi = ef.force(2, 99.0)[0][0];    // clamped high
        check(std::fabs(a - 5.0) < 1e-12 && std::fabs(b - 15.0) < 1e-12
              && lo == 0.0 && std::fabs(hi - 20.0) < 1e-12,
              "time-varying force interpolates and clamps");
    }

    // ---- a stretch applies no net force ----
    {
        ExternalForce ef;
        ef.add_stretch(0, -1, Vec3{0.0, 0.0, 7.0});
        const Vec3Field F = ef.force(6, 0.0);
        Vec3 net{0, 0, 0};
        for (const auto& f : F) net = net + f;
        check(std::sqrt(norm2(net)) < 1e-14 && std::fabs(F[5][2] - 7.0) < 1e-14
              && std::fabs(F[0][2] + 7.0) < 1e-14,
              "stretch is equal and opposite (zero net force)");
    }

    // ---- physics: Hookean dumbbell under a constant stretching force ----
    // For a Hookean dumbbell (H = 1) pulled by +/- f the connector obeys the linear
    // SDE  dQ = -(1/2)(Q - f zhat) dt + noise,  so <Q_z> = f exactly.
    //
    // Measuring that mean directly is noise-limited (the connector's stationary
    // standard deviation is 1, with a correlation time of 2, so a long run still
    // leaves a sizeable error on the mean). But the equation is LINEAR, so two runs
    // driven by the *same* random stream differ by a purely deterministic amount:
    // Q_f(t) - Q_0(t) relaxes to exactly f. Differencing at fixed seed therefore
    // cancels the noise completely and tests the applied force to high precision.
    {
        auto mean_Qz = [](dp f) {
            PhysParams phys;
            phys.spring.type = Spring::Hook;
            phys.spring.sqrtb = 1.0e6;          // effectively infinitely extensible
            phys.number_of_beads = 2;
            phys.hstar = 0.0;
            if (f != 0.0) phys.external.add_stretch(0, -1, Vec3{0.0, 0.0, f});

            Vec3Field R = {Vec3{0, 0, 0}, Vec3{0, 0, 1}};   // same start for every f
            Rng rng(7);                                      // same stream for every f
            SimParams sim;
            sim.dt = 2.0e-3;
            sim.update_center_of_mass = false;

            dp sum = 0.0;
            int n = 0;
            for (int step = 0; step < 40000; ++step) {
                sim.time_start = step * sim.dt;
                sim.time_end = sim.time_start;
                time_integrate_chain(R, phys, sim, rng);
                if (step > 10000) { sum += R[1][2] - R[0][2]; ++n; }
            }
            return sum / n;
        };

        const dp base = mean_Qz(0.0);
        for (dp f : {0.5, 2.0, 5.0}) {
            const dp d = mean_Qz(f) - base;
            std::printf("    f = %.1f : <Q_z>(f) - <Q_z>(0) = %.6f  (exact %.1f)\n", f, d, f);
            check(std::fabs(d - f) < 1e-3 * f,
                  "forced extension matches the exact linear-response result");
        }
    }

    std::printf("\nExternal force: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
