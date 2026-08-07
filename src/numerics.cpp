#include "numerics.hpp"

#include <algorithm>

namespace bdsim {

// Bracketed bisection on the cubic, used as a guaranteed-in-range fallback.
static dp bisect_cubic(const std::array<dp, 4>& c, dp lo, dp hi) {
    auto f = [&](dp x) { return ((c[3] * x + c[2]) * x + c[1]) * x + c[0]; };
    dp flo = f(lo), fhi = f(hi);
    if (flo == 0.0) return lo;
    if (fhi == 0.0) return hi;
    if (flo * fhi > 0.0)                       // no sign change: clamp to nearer end
        return (std::fabs(flo) < std::fabs(fhi)) ? lo : hi;
    for (int it = 0; it < 200; ++it) {
        const dp mid = 0.5 * (lo + hi);
        const dp fm = f(mid);
        if (fm == 0.0 || (hi - lo) < 1e-15) return mid;
        if (flo * fm < 0.0) { hi = mid; fhi = fm; }
        else                { lo = mid; flo = fm; }
    }
    return 0.5 * (lo + hi);
}

dp find_roots_cubic(const std::array<dp, 4>& coeff, dp lower_bound, dp upper_bound) {
    // Fortran indices coeff(1..4) -> coeff[0..3].
    const dp a = coeff[2] / coeff[3];
    const dp b = coeff[1] / coeff[3];
    const dp c = coeff[0] / coeff[3];

    const dp Q = (a * a - 3.0 * b) / 9.0;
    const dp R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;

    dp x;
    if (R * R < Q * Q * Q) {
        // three real roots; pick the one inside the bracket
        const dp theta = std::acos(R / std::sqrt(Q * Q * Q));
        for (int i = -1; i <= 1; ++i) {
            x = -2.0 * std::sqrt(Q) *
                    std::cos((theta + static_cast<dp>(i) * PI * 2.0) / 3.0) -
                a / 3.0;
            if (x >= lower_bound && x <= upper_bound) return x;
        }
        x = -2.0 * std::sqrt(Q) * std::cos((theta + PI * 2.0) / 3.0) - a / 3.0;
    } else {
        // one real root
        const dp Au =
            -std::copysign(1.0, R) *
            std::cbrt(std::fabs(R) + std::sqrt(R * R - Q * Q * Q));
        const dp Bu = (Au == 0.0) ? 0.0 : Q / Au;
        x = (Au + Bu) - a / 3.0;
    }
    // The physical root is guaranteed to lie in [lower, upper]. The closed-form
    // expressions can lose precision for extreme coefficients (very large gama)
    // and return a spurious out-of-range root; in that case fall back to a
    // bracketed bisection so the finite-extensibility bound is never violated.
    if (x >= lower_bound && x <= upper_bound) return x;
    return bisect_cubic(coeff, lower_bound, upper_bound);
}

dp polish_poly_root(const std::array<dp, 4>& c, dp x0, dp atol) {
    constexpr int n = 4;      // cubic
    constexpr int IMAX = 30;

    dp x = x0;
    for (int iter = 0; iter < IMAX; ++iter) {
        dp p  = c[n - 1] * x + c[n - 2];   // c(n)*x + c(n-1)
        dp p1 = c[n - 1];                   // c(n)
        for (int i = n - 3; i >= 0; --i) {  // i = n-2 .. 1 (Fortran) -> n-3..0
            p1 = p + p1 * x;
            p  = c[i] + p * x;
        }

        if (std::fabs(p1) == 0.0) {
            p1 = 6.0 * c[3];      // f'''
            p1 = 6.0 * p / p1;
            x -= std::copysign(1.0, p1) * std::cbrt(std::fabs(p1));
        } else {
            x -= p / p1;
        }

        if (std::fabs(p) <= atol) break;
    }
    return x;
}

} // namespace bdsim
