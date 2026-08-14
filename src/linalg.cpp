#include "linalg.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace bdsim {

namespace {
// A Cholesky factorisation only exists for a positive-definite matrix. The RPY
// diffusion tensor is positive definite for any physical configuration, so a
// non-positive pivot means the configuration itself has already gone wrong --
// typically two beads driven on top of each other, which makes D numerically
// singular. Reporting that is far more useful than std::sqrt of a negative
// number, which returns NaN silently and contaminates everything downstream.
[[noreturn]] void not_positive_definite(int index, dp pivot) {
    throw std::runtime_error(
        "cholesky_lower: matrix is not positive definite (pivot " +
        std::to_string(index) + " = " + std::to_string(pivot) + "). For the "
        "RPY tensor this means the configuration is degenerate -- usually two "
        "beads at essentially the same position after a diverging step. Check "
        "for an earlier corrector non-convergence warning: that is where the "
        "trajectory actually failed.");
}
}  // namespace

// The dense kernels (matvec / Cholesky / triangular apply / dot) have two
// backends. The default is hand-rolled and dependency-free. Building with
// BDSIM_USE_LAPACK routes them through BLAS/LAPACK (dgemv / dpotrf / dtrmv /
// ddot) instead -- much faster for the HI path at large N. Callers are
// unchanged: cholesky_lower + lower_matvec are a matched pair, and the lower
// Cholesky factor is unique, so both backends give identical results.
#ifdef BDSIM_USE_LAPACK

extern "C" {
void dgemv_(const char*, const int*, const int*, const double*, const double*,
            const int*, const double*, const int*, const double*, double*, const int*);
void dpotrf_(const char*, const int*, double*, const int*, int*);
void dtrmv_(const char*, const char*, const char*, const int*, const double*,
            const int*, double*, const int*);
double ddot_(const int*, const double*, const int*, const double*, const int*);
}

std::vector<dp> matvec(const Matrix& M, const std::vector<dp>& x) {
    std::vector<dp> y(M.n);
    const char trans = 'T';           // row-major A == col-major A^T; 'T' -> A x
    const int n = M.n, inc = 1;
    const double one = 1.0, zero = 0.0;
    dgemv_(&trans, &n, &n, &one, M.a.data(), &n, x.data(), &inc, &zero, y.data(), &inc);
    return y;
}

Matrix cholesky_lower(const Matrix& M) {
    Matrix L = M;                     // dpotrf factors in place
    const char uplo = 'L';
    const int n = M.n;
    int info = 0;
    dpotrf_(&uplo, &n, L.a.data(), &n, &info);
    // info > 0: the leading minor of that order is not positive definite.
    if (info > 0) not_positive_definite(info - 1, L(info - 1, info - 1));
    if (info < 0) throw std::runtime_error(
        "cholesky_lower: dpotrf rejected argument " + std::to_string(-info));
    return L;
}

std::vector<dp> lower_matvec(const Matrix& L, const std::vector<dp>& x) {
    std::vector<dp> y = x;            // dtrmv overwrites its vector
    const char uplo = 'L', trans = 'N', diag = 'N';
    const int n = L.n, inc = 1;
    dtrmv_(&uplo, &trans, &diag, &n, L.a.data(), &n, y.data(), &inc);
    return y;
}

dp dot(const std::vector<dp>& a, const std::vector<dp>& b) {
    const int n = static_cast<int>(a.size()), inc = 1;
    return ddot_(&n, a.data(), &inc, b.data(), &inc);
}

#else  // ---- hand-rolled backend (default, dependency-free) ----

std::vector<dp> matvec(const Matrix& M, const std::vector<dp>& x) {
    std::vector<dp> y(M.n, 0.0);
    for (int i = 0; i < M.n; ++i) {
        dp s = 0.0;
        for (int j = 0; j < M.n; ++j) s += M(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

Matrix cholesky_lower(const Matrix& M) {
    const int n = M.n;
    Matrix L(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            dp sum = M(i, j);
            for (int k = 0; k < j; ++k) sum -= L(i, k) * L(j, k);
            if (i == j) {
                if (!(sum > 0.0)) not_positive_definite(i, sum);
                L(i, j) = std::sqrt(sum);
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }
    return L;
}

std::vector<dp> lower_matvec(const Matrix& L, const std::vector<dp>& x) {
    std::vector<dp> y(L.n, 0.0);
    for (int i = 0; i < L.n; ++i) {
        dp s = 0.0;
        for (int j = 0; j <= i; ++j) s += L(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

dp dot(const std::vector<dp>& a, const std::vector<dp>& b) {
    dp s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

#endif

std::vector<dp> flatten(const Vec3Field& f) {
    std::vector<dp> v(3 * f.size());
    for (size_t mu = 0; mu < f.size(); ++mu)
        for (int i = 0; i < 3; ++i) v[3 * mu + i] = f[mu][i];
    return v;
}

Vec3Field unflatten(const std::vector<dp>& v) {
    Vec3Field f(v.size() / 3);
    for (size_t mu = 0; mu < f.size(); ++mu)
        for (int i = 0; i < 3; ++i) f[mu][i] = v[3 * mu + i];
    return f;
}

} // namespace bdsim
