// test_hi.cpp — RPY diffusion tensor and the square-root methods, vs the Fortran
// oracle in test_HI_and_chevyshev (N=4, hstar=0.2, delts=0.1, given X_0).
#include "../src/hydrodynamics.hpp"
#include "../src/linalg.hpp"

#include <cmath>
#include <cstdio>

using namespace bdsim;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%-40s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    const dp delts = 0.1, hstar = 0.2;
    Vec3Field R = {
        {0.0, 0.0, 0.0},
        {-8.7001417024242400e+00, 5.3611027493752204e+00, 2.0995067416916502e+00},
        {-9.1605814460059198e-01, 1.1704273266182801e+01, -1.4985387607124800e+00},
        {-5.4793423363635396e+00, 4.8982010793314998e+00, 4.0063480623078496e+00}};
    Vec3Field X0 = {
        {-0.653795716820965, 2.25469853815588, 0.165603302292486},
        {-0.587509005201644, -0.745084985707505, -0.157459965970764},
        {-1.88644397151742, 3.16619, 1.23661201418787},
        {-0.958635132046656, -0.253142128732732, 1.43910244145363}};

    const Matrix D = rpy_diffusion_matrix(R, hstar);

    // --- RPY tensor block (0,1) and self block (1,1) ---
    const double blk01[3][3] = {
        {0.043185241172496, -0.010895545687992, -0.004266896699380},
        {-0.010895545687992, 0.032217586367374, 0.002629298741189},
        {-0.004266896699380, 0.002629298741189, 0.026533338541433}};
    bool ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (std::fabs(D(i, 3 + j) - blk01[i][j]) > 1e-6) ok = false;
    check(ok, "RPY tensor block (0,1)");

    ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (std::fabs(D(3 + i, 3 + j) - (i == j ? 1.0 : 0.0)) > 1e-12) ok = false;
    check(ok, "RPY self block (1,1) = identity");

    // --- Cholesky:  B.X0 with B the lower Cholesky factor ---
    const Vec3Field chol_expected = {
        {-0.206748358961671, 0.712998281762464, 0.052368362328964},
        {-0.202521038852097, -0.210272818508389, -0.045641895488899},
        {-0.611043311268904, 1.022458498065220, 0.391180874642083},
        {-0.351959054078799, -0.035305408161349, 0.461910033760980}};
    const Matrix L = cholesky_lower(D);
    const Vec3Field chol = unflatten(lower_matvec(L, flatten(X0)));
    ok = true;
    for (size_t m = 0; m < R.size(); ++m)
        for (int d = 0; d < 3; ++d)
            if (std::fabs(std::sqrt(delts) * chol[m][d] - chol_expected[m][d]) > 1e-5) ok = false;
    check(ok, "Cholesky delS");

    // --- Chebyshev: verify it is a genuine symmetric square root of D ---
    // (1) fluctuation-dissipation: |B.X0|^2 == X0.D.X0
    // (2) squares back to D: B.(B.X0) == D.X0  (with the same Chebyshev operator)
    const ChebyshevResult cr = chebyshev_sqrt_times(D, X0, /*mult=*/1.0, /*fd_tol=*/1e-6);
    std::printf("  (chebyshev: %d terms, fd_error = %.3e)\n", cr.nterms, cr.fd_error);
    check(cr.fd_error < 1e-4, "Chebyshev fluctuation-dissipation");

    const Vec3Field BBx0 = chebyshev_sqrt_times(D, cr.value, 1.0, 1e-6).value;
    const Vec3Field Dx0  = unflatten(matvec(D, flatten(X0)));
    double maxdiff = 0.0;
    for (size_t m = 0; m < R.size(); ++m)
        for (int d = 0; d < 3; ++d)
            maxdiff = std::max(maxdiff, std::fabs(BBx0[m][d] - Dx0[m][d]));
    std::printf("  (B(B.X0) vs D.X0 max diff = %.3e)\n", maxdiff);
    check(maxdiff < 1e-4, "Chebyshev squares back to D");
    (void)delts;

    std::printf("\nHI: %s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
