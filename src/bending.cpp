#include "bending.hpp"

namespace bdsim {

Vec3Field BendingForce::force(const ChainState& state) const {
    const ChainGeometry& g = state.geom;
    const int N = g.nbeads();
    Vec3Field F(N, Vec3{0.0, 0.0, 0.0});

    if (p_.type == Bending::None) return F;
    if (N == 2) return F;  // no interior angle

    const dp C = p_.stiffness;
    const std::vector<dp>& len = g.length;       // len[a] = |bond a| = dist(a, a+1)
    const Vec3Field u = unit_bonds(g);            // bond unit vectors  (N-1)
    const std::vector<dp> cs = bond_cosines(u);   // cos between bonds  (N-2)

    // Chain ends (same for all N >= 3).
    F[0]     = (C / len[0])     * (cs[0] * u[0] - u[1]);
    F[N - 1] = (C / len[N - 2]) * ((-cs[N - 3]) * u[N - 2] + u[N - 3]);

    if (N == 3) {
        F[1] = C * ((1.0 / len[1]) * (cs[0] * u[1] - u[0]) +
                    (1.0 / len[0]) * ((-cs[0]) * u[0] + u[1]));
        return F;
    }

    F[1] = C * ((1.0 / len[1]) * (cs[0] * u[1] - u[0]) +
                (1.0 / len[0]) * ((-cs[0]) * u[0] + u[1]) +
                (1.0 / len[1]) * (cs[1] * u[1] - u[2]));

    F[N - 2] = C * ((1.0 / len[N - 2]) * (cs[N - 3] * u[N - 2] - u[N - 3]) +
                    (1.0 / len[N - 3]) * ((-cs[N - 3]) * u[N - 3] + u[N - 2]) +
                    (1.0 / len[N - 3]) * ((-cs[N - 4]) * u[N - 3] + u[N - 4]));

    for (int m = 2; m <= N - 3; ++m) {
        F[m] = C * ((1.0 / len[m])     * (cs[m - 1] * u[m] - u[m - 1]) +
                    (1.0 / len[m - 1]) * ((-cs[m - 1]) * u[m - 1] + u[m]) +
                    (1.0 / len[m - 1]) * ((-cs[m - 2]) * u[m - 1] + u[m - 2]) +
                    (1.0 / len[m])     * (cs[m] * u[m] - u[m + 1]));
    }
    return F;
}

} // namespace bdsim
