// test_mobility.cpp — the free-draining diffusion (D = I) via Mobility/Diffusion.
#include "../src/mobility.hpp"
#include <cmath>
#include <cstdio>
using namespace bdsim;

int main() {
    int failures = 0;
    const Mobility mob(0.0, DelSMethod::Cholesky, 1.0, 0.0025);  // hstar=0 => identity

    Vec3Field F = {{1,-2,3},{0.5,0.25,-0.75},{-4,5,6}};
    const Diffusion d = mob.at(F);              // config only sets size here
    const Vec3Field DF = d.apply(F);
    for (size_t i = 0; i < F.size(); ++i) for (int k = 0; k < 3; ++k)
        if (DF[i][k] != F[i][k]) ++failures;
    std::printf("apply is identity: %s\n", failures == 0 ? "PASS" : "FAIL");

    Vec3Field R(4, Vec3{0,0,0});
    const Diffusion d4 = mob.at(R);
    Vec3Field a, b;
    { Rng r(11); a = d4.brownian_step(r, 0.01); }
    { Rng r(11); b = d4.brownian_step(r, 0.01); }
    int mism = 0;
    for (size_t i = 0; i < a.size(); ++i) for (int k = 0; k < 3; ++k) if (a[i][k] != b[i][k]) ++mism;
    std::printf("brownian reproducible: %s\n", mism == 0 ? "PASS" : "FAIL");
    failures += mism;

    Vec3Field R2(2, Vec3{0,0,0});
    const Diffusion d2 = mob.at(R2);
    Vec3Field s1, s2;
    { Rng r(3); s1 = d2.brownian_step(r, 1.0); }
    { Rng r(3); s2 = d2.brownian_step(r, 4.0); }
    int bad = 0;
    for (size_t i = 0; i < s1.size(); ++i) for (int k = 0; k < 3; ++k)
        if (std::fabs(s2[i][k] - 2.0 * s1[i][k]) > 1e-12) ++bad;  // sqrt(4/1)=2
    std::printf("brownian sqrt(dt) scaling: %s\n", bad == 0 ? "PASS" : "FAIL");
    failures += bad;

    std::printf("\nMobility: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
