// test_ev.cpp — excluded-volume force vs Fortran oracle (test_EV_cutoff).
#include "../src/excluded_volume.hpp"
#include "../src/config.hpp"
#include <cmath>
#include <cstdio>
#include <limits>
using namespace bdsim;

static int checks = 0, failures = 0;
static void fail(const char* m) { std::printf("  [FAIL] %s\n", m); ++failures; }

int main() {
    const double eps = std::numeric_limits<double>::epsilon();

    {   // contour distance 10 on a 10-bead chain -> all forces zero
        Vec3Field R = {
            {0,0,0},
            {-8.7001417024242400e+00,5.3611027493752204e+00,2.0995067416916502e+00},
            {-9.1605814460059198e-01,1.1704273266182801e+01,-1.4985387607124800e+00},
            {-5.4793423363635396e+00,4.8982010793314998e+00,4.0063480623078496e+00},
            {-1.0438766944965399e+01,-3.6900277039248199e+00,1.7059173838192401e+00},
            {-7.3861146555559003e+00,-5.5679879113699000e+00,-8.5270109831673704e+00},
            {-1.1052344868863999e+01,-1.4308940267976500e+00,-1.6009484144943301e+01},
            {-8.7582187147024406e+00,-4.2391733059389702e+00,-2.6018313074301499e+01},
            {-1.5884767628153300e+01,3.9463324714686001e+00,-2.5335247046303198e+01},
            {-9.3183370137224504e+00,1.1169550080743599e+01,-2.8667718582544101e+01}};
        EVParams p; p.type = EV::LJ; p.zstar = 1.0; p.dstar = 10.0; p.contour_dist_for_EV = 10;
        ExcludedVolume ev(p, /*equilibration=*/false, (int)R.size());
        const ChainGeometry g = chain_geometry(R);
        const Vec3Field F = ev.force({R, g});
        ++checks;
        for (auto& v : F) for (int d = 0; d < 3; ++d)
            if (std::fabs(v[d]) > 1e-6) fail("Fev not zero for contour dist 10");
    }

    {   // end beads feel a y-force; interior x-forces vanish
        Vec3Field R = {{0,0,0},{1,0,0},{2.1,0,0},{3,0,0},{0,1.5,0}};
        const int N = (int)R.size();
        EVParams p; p.type = EV::LJ; p.zstar = 1.0; p.dstar = 1.0; p.contour_dist_for_EV = 4;
        ExcludedVolume ev(p, false, N);
        const ChainGeometry g = chain_geometry(R);
        const Vec3Field F = ev.force({R, g});
        ++checks;
        for (int nu = 0; nu < N; ++nu) {
            if (nu == 0 || nu == N-1) { if (std::fabs(F[nu][1]) <= eps) fail("end bead needs y-force"); }
            else { if (F[nu][0] != 0.0) fail("interior x-force should be zero"); }
        }
    }

    std::printf("\nExcluded volume: %d blocks, %d failures -> %s\n",
                checks, failures, failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
