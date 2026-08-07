// vec.hpp — tiny fixed-size vector/matrix helpers for the 3D bead math.
// Kept deliberately small and header-only so the compiler fully inlines the
// hot-loop arithmetic (important for matching/beating the Fortran).
#pragma once

#include "types.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace bdsim {

using Vec3 = std::array<dp, 3>;
// A per-bead field of 3-vectors (forces, positions, displacements, ...).
using Vec3Field = std::vector<Vec3>;
// Row-major 3x3: M[i][j] is row i, column j.
using Mat3 = std::array<std::array<dp, 3>, 3>;

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Vec3 operator*(dp s, const Vec3& a) {
    return {s * a[0], s * a[1], s * a[2]};
}
inline dp dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline dp norm2(const Vec3& a) { return dot(a, a); }
inline dp norm(const Vec3& a) { return std::sqrt(norm2(a)); }

// Matrix-vector product K * v.
inline Vec3 matvec(const Mat3& K, const Vec3& v) {
    return {K[0][0] * v[0] + K[0][1] * v[1] + K[0][2] * v[2],
            K[1][0] * v[0] + K[1][1] * v[1] + K[1][2] * v[2],
            K[2][0] * v[0] + K[2][1] * v[1] + K[2][2] * v[2]};
}


// ---- whole-field arithmetic (per-bead), so integrator code reads like the SDE.
// These allocate a result field; fine for clarity, poolable later if needed.
inline Vec3Field operator+(const Vec3Field& a, const Vec3Field& b) {
    Vec3Field r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}
inline Vec3Field operator-(const Vec3Field& a, const Vec3Field& b) {
    Vec3Field r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}
inline Vec3Field operator*(dp s, const Vec3Field& a) {
    Vec3Field r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = s * a[i];
    return r;
}

} // namespace bdsim
