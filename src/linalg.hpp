// linalg.hpp — minimal dense linear algebra for the hydrodynamic mobility.
//
// Deliberately hand-rolled and dependency-free for now: a small symmetric dense
// matrix, matrix-vector product, and Cholesky factor. These are the operations
// the RPY diffusion tensor needs. They can later be dispatched to BLAS/LAPACK or
// Eigen for speed without changing callers (they'd back the same functions).
#pragma once

#include "vec.hpp"

#include <vector>

namespace bdsim {

// Dense n x n matrix, row-major.
struct Matrix {
    int n = 0;
    std::vector<dp> a;

    Matrix() = default;
    explicit Matrix(int n_) : n(n_), a(static_cast<size_t>(n_) * n_, 0.0) {}

    dp&       operator()(int i, int j)       { return a[static_cast<size_t>(i) * n + j]; }
    dp        operator()(int i, int j) const { return a[static_cast<size_t>(i) * n + j]; }
};

// y = M x   (full matrix).
std::vector<dp> matvec(const Matrix& M, const std::vector<dp>& x);

// Lower Cholesky factor L with  M = L L^T  (M symmetric positive definite).
Matrix cholesky_lower(const Matrix& M);

// y = L x   using only the lower triangle of L.
std::vector<dp> lower_matvec(const Matrix& L, const std::vector<dp>& x);

dp dot(const std::vector<dp>& a, const std::vector<dp>& b);

// Bridge between the per-bead field and the flat 3N vector the matrices act on.
// Flat index of bead mu, component i is 3*mu + i (matches the tensor row order).
std::vector<dp> flatten(const Vec3Field& f);
Vec3Field       unflatten(const std::vector<dp>& v);

} // namespace bdsim
