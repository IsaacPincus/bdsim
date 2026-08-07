// test_spring_force.cpp — spring_force sanity (Hookean => Rouse connector force).
#include "../src/spring.hpp"
#include "../src/config.hpp"
#include <cmath>
#include <cstdio>
using namespace bdsim;
int main() {
    Vec3Field R = {{0,0,0},{1,0,0},{1,1,0},{2,1,0}};
    const int N = (int)R.size();
    SpringParams sp; sp.type = Spring::Hook; sp.sqrtb = 1000.0; sp.natural_length = 0.0;
    const Vec3Field Fs = spring_force(chain_geometry(R), sp);

    // For Hookean ff=1: F[i] = (R[i+1]-R[i]) - (R[i]-R[i-1]); ends one-sided.
    Vec3Field exp(N);
    auto q=[&](int a,int b){ return R[b]-R[a]; };
    exp[0]=q(0,1); exp[N-1]=(-1.0)*q(N-2,N-1);
    for (int i=1;i<N-1;++i) exp[i]=q(i,i+1)-q(i-1,i);
    int failures=0;
    for (int i=0;i<N;++i) for(int d=0;d<3;++d)
        if (std::fabs(Fs[i][d]-exp[i][d])>1e-6) { ++failures;
            std::printf("  [FAIL] bead %d comp %d got %.9g want %.9g\n",i,d,Fs[i][d],exp[i][d]); }
    std::printf("\nSpring force (Hookean): %d failures -> %s\n",
                failures, failures==0?"ALL PASSED":"FAILED");
    return failures==0?0:1;
}
