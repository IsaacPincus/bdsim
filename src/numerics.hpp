// numerics.hpp — general numerical helpers ported from utils.f90.
// (find_roots_cubic, polish_poly_root, rtsafe.)
#pragma once

#include "types.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace bdsim {

// Analytic real root of a cubic in a bracket [lower, upper].
// coeff is {c0, c1, c2, c3} for c0 + c1 x + c2 x^2 + c3 x^3 (Fortran coeff(1..4)).
// Port of utils.f90::find_roots_cubic.
dp find_roots_cubic(const std::array<dp, 4>& coeff, dp lower_bound, dp upper_bound);

// Newton-polish a root of the same cubic; returns the polished root.
dp polish_poly_root(const std::array<dp, 4>& c, dp x0, dp atol);

// Safeguarded Newton (bisection fallback) root finder on [lo, hi].
// funcd(x, f, df) fills value and derivative. Port of utils.f90::rtsafe.
template <typename Funcd>
dp rtsafe(Funcd funcd, dp lower_bound, dp upper_bound, dp root_accuracy) {
    constexpr int MAXIT = 10000;

    dp x1 = lower_bound, x2 = upper_bound, xacc = root_accuracy;
    dp fl, fh, df;
    funcd(x1, fl, df);
    funcd(x2, fh, df);

    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
        std::fprintf(stderr, "root must be bracketed in rtsafe\n");
        std::abort();
    }
    if (fl == 0.0) return x1;
    if (fh == 0.0) return x2;

    dp xl, xh;
    if (fl < 0.0) { xl = x1; xh = x2; }
    else          { xh = x1; xl = x2; }

    dp rts = 0.5 * (x1 + x2);
    dp dxold = std::fabs(x2 - x1);
    dp dx = dxold;
    dp f;
    funcd(rts, f, df);

    for (int j = 0; j < MAXIT; ++j) {
        if (((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0.0 ||
            std::fabs(2.0 * f) > std::fabs(dxold * df)) {
            // bisection
            dxold = dx;
            dx = 0.5 * (xh - xl);
            rts = xl + dx;
            if (xl == rts) return rts;
        } else {
            dxold = dx;
            dx = f / df;
            dp temp = rts;
            rts -= dx;
            if (temp == rts) return rts;
        }
        if (std::fabs(dx) < xacc) return rts;
        funcd(rts, f, df);
        if (f < 0.0) xl = rts;
        else         xh = rts;
    }

    std::fprintf(stderr, "Max iterations exceeded in rtsafe\n");
    std::abort();
}

} // namespace bdsim
