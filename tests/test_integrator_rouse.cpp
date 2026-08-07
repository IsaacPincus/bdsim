// test_integrator_rouse.cpp — end-to-end regression vs Fortran test_rouse_chain_eq.
// Two consecutive integrations on ONE continued RNG stream (no reseed between).
#include "../src/integrator.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace bdsim;

static int failures = 0;
static void compare(const std::vector<Vec3>& got, const std::vector<Vec3>& want,
                    const char* name) {
    int bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        for (int d = 0; d < 3; ++d) {
            double diff = std::fabs(got[i][d] - want[i][d]);
            if (diff > 1e-6) {
                if (bad < 4)
                    std::printf("  [FAIL] %s bead %zu comp %d got %.15g want %.15g (|d|=%.2e)\n",
                                name, i, d, got[i][d], want[i][d], diff);
                ++bad;
            }
        }
    std::printf("%-28s %s\n", name, bad == 0 ? "PASS" : "FAIL");
    if (bad) ++failures;
}

int main() {
    const int N = 10;

    PhysParams phys;
    phys.number_of_beads = N;
    phys.spring.type = Spring::Hook;
    phys.spring.sqrtb = 1000.0;
    phys.spring.natural_length = 0.0;
    phys.hstar = 0.0;
    phys.ev.type = EV::None;
    phys.bend.type = Bending::None;
    phys.flow = Flow();  // EQ => zero kappa

    SimParams sim;
    sim.time_start = 0.0;
    sim.time_end = 1.0;
    sim.dt = 0.1;
    sim.implicit_loop_tol = 1e-6;
    sim.update_center_of_mass = true;

    std::vector<Vec3> R = {
        {0,0,0},
        {-6.586250662803650e-02,-4.911733418703079e-02,2.791636288166050e-01},
        {-6.753685772418980e-01,3.519757017493250e-01,-6.340299546718600e-01},
        {-2.844688922166820e00,-2.239686943590640e00,-7.809645235538480e-01},
        {-1.818384915590290e00,-2.201230965554710e00,9.345296323299410e-01},
        {-3.210149317979810e00,-9.372534379363060e-01,-4.681020081043240e-01},
        {-1.490711003541950e00,-5.971195027232170e-01,-1.509506493806840e00},
        {-1.587433911859990e00,-1.017917685210700e00,-1.456502400338650e00},
        {-6.204553022980690e-01,-1.219155095517640e00,-7.125178799033161e-01},
        {-8.804024234414100e-01,-3.662307761609550e00,-7.348611745983360e-01}};

    std::vector<Vec3> ans1 = {
        {2.014655995894740e00,3.266955094952210e00,6.195918791485830e-01},
        {6.898958025934120e-01,8.753981552180370e-01,4.785034222762240e-01},
        {-3.793564324297950e-01,1.489672909282450e00,-6.390832004693100e-01},
        {-1.130337343526490e00,-6.146915212957660e-01,-5.621008935576799e-02},
        {-6.222945452675470e-01,-1.038001996615710e00,5.028959245456580e-01},
        {-2.989398757279950e-01,-8.209160875818450e-01,1.289920072746550e00},
        {2.453406088052380e-01,2.086869672299050e-01,4.316992884969550e-01},
        {-1.110128512089560e-01,-5.552336867493930e-01,-1.277926546344950e00},
        {-6.424329445947624e-02,-1.036741598612170e00,-1.699024281591250e00},
        {-1.915098093214250e-01,-2.620904430772130e00,2.403765459336170e-01}};

    std::vector<Vec3> ans2 = {
        {2.358466969676050e00,3.680650287690870e00,7.391286505833830e-01},
        {9.840637226840210e-02,1.553897668605920e00,-4.535516789623883e-02},
        {3.703216834718498e-03,1.100513005729840e00,-5.311631291023066e-02},
        {-1.762304275149680e00,-2.808383016377780e-02,2.665681695031340e-01},
        {-1.122405351207410e00,-4.065182692363470e-01,9.790414477264030e-01},
        {6.632472866300509e-02,-4.887170934587170e-01,9.806758704379520e-01},
        {-5.063576758472540e-01,-9.911910816941520e-01,1.377974053682720e-01},
        {2.248130548550660e-01,-9.103044958788740e-01,-1.223068999396410e00},
        {-2.639644443515610e-01,-1.153796153975820e00,-1.197611287878650e00},
        {1.905555228121980e00,-2.718416492074990e00,-5.090337805446200e-01}};

    Rng rng(5);
    rng.reset(5);   // reset_RNG_with_seed(5)

    time_integrate_chain(R, phys, sim, rng);
    compare(R, ans1, "rouse integration 1");

    // Second integration continues the SAME rng stream (no reseed), from ans1.
    R = ans1;
    time_integrate_chain(R, phys, sim, rng);
    compare(R, ans2, "rouse integration 2");

    std::printf("\nIntegrator (rouse): %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
