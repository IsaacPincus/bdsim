#include "hydrodynamics.hpp"

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace bdsim {

Matrix rpy_diffusion_matrix(const Vec3Field& R, dp hstar) {
    const int N = static_cast<int>(R.size());
    Matrix D(3 * N);

    // self-mobility: identity blocks on the diagonal
    for (int mu = 0; mu < N; ++mu)
        for (int i = 0; i < 3; ++i) D(3 * mu + i, 3 * mu + i) = 1.0;

    if (hstar <= 0.0) return D;

    const dp RPI = std::sqrt(PI), TRPI = 2.0 * RPI, TPI = 2.0 * PI;
    const dp C1 = TPI / 3.0, C3 = 0.75 / RPI * 0.375, C7 = 0.09375 / RPI;
    const dp RPI3by4 = RPI * 0.75;

    for (int nu = 1; nu < N; ++nu) {
        for (int mu = 0; mu < nu; ++mu) {
            const Vec3 b = R[nu] - R[mu];       // bead-to-bead vector
            dp sq = norm2(b);
            if (sq < 1.0e-12) sq = 1.0e-12;
            const dp rs = std::sqrt(sq);

            const dp hsbyrs = hstar / rs, rsbyhs = rs / hstar;
            const dp hsbyrs2 = hsbyrs * hsbyrs;

            dp om1, om2;  // isotropic and dyadic weights of the RPY interaction
            if (rsbyhs >= TRPI) {  // well separated beads
                om1 = RPI3by4 * hsbyrs * (1.0 + C1 * hsbyrs2);
                om2 = RPI3by4 * (hsbyrs / (rs * rs)) * (1.0 - TPI * hsbyrs2);
            } else {               // overlapping spheres
                om1 = 1.0 - C3 * rsbyhs;
                om2 = C7 * rsbyhs / (rs * rs);
            }

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const dp val = om2 * b[i] * b[j] + (i == j ? om1 : 0.0);
                    D(3 * mu + i, 3 * nu + j) = val;
                    D(3 * nu + j, 3 * mu + i) = val;  // symmetric
                }
        }
    }
    return D;
}

namespace {

// Spectral bounds for the Chebyshev interval [lmin, lmax].
//
// The interval MUST contain the whole spectrum of D: a Chebyshev polynomial that
// approximates sqrt on [lmin, lmax] GROWS (like cosh) outside it, so any
// eigen-component whose eigenvalue is below lmin or above lmax is amplified rather
// than square-rooted, and the Brownian displacement B.dW diverges. The original
// Fixman estimate (a Rayleigh quotient of one alternating vector) lies inside
// [lambda_min, lambda_max] and badly overestimated lambda_min for stretched chains
// -- it left the smallest eigenvalues outside the interval, which was the cause of
// intermittent blow-ups.
//
// We instead estimate the extreme eigenvalues with a short Lanczos run (Ritz
// values converge to the extremes quickly), then pad outwards. Ritz values bracket
// the true spectrum from the inside, so the padding is what guarantees containment;
// the Cholesky fallback in Diffusion::brownian_step is the final safety net.
//
// Successive timesteps change the configuration very little, so the extremal
// eigenvectors are nearly unchanged. Seeding Lanczos with the previous step's
// extremal Ritz vectors (a "warm start") reaches the same accuracy in far fewer
// iterations -- see SpectralCache in the header.

constexpr int LANCZOS_COLD = 30;   // iterations from a cold start
constexpr int LANCZOS_WARM = 8;    // iterations when warm-started

// Deterministic, spectrally rich fallback seed.
std::vector<dp> default_seed(int n) {
    std::vector<dp> v(n);
    for (int i = 0; i < n; ++i) v[i] = std::sin(0.7654321 * (i + 1)) + 0.5;
    return v;
}

// Sturm sequence: number of eigenvalues of the symmetric tridiagonal (a, b) that
// are strictly less than x. (b has a.size()-1 entries.)
int sturm_count(const std::vector<dp>& a, const std::vector<dp>& b, dp x) {
    int count = 0;
    dp d = a[0] - x;
    if (d < 0.0) ++count;
    for (size_t i = 1; i < a.size(); ++i) {
        if (d == 0.0) d = 1.0e-300;
        d = (a[i] - x) - b[i - 1] * b[i - 1] / d;
        if (d < 0.0) ++count;
    }
    return count;
}

// Eigenvector of the symmetric tridiagonal (a, b) for an (approximate) eigenvalue
// lambda, by inverse iteration. The shift is nudged off lambda so the solve stays
// non-singular. O(m) per iteration.
std::vector<dp> tridiag_eigenvector(const std::vector<dp>& a, const std::vector<dp>& b,
                                    dp lambda) {
    const int m = static_cast<int>(a.size());
    const dp scale = std::max(std::fabs(lambda), 1.0);
    const dp shift = lambda - 1.0e-7 * scale;

    std::vector<dp> y(m, 1.0 / std::sqrt(static_cast<dp>(m)));
    std::vector<dp> c(m), d(m);
    for (int it = 0; it < 2; ++it) {
        // Thomas algorithm on (T - shift I) x = y
        dp beta0 = a[0] - shift;
        if (std::fabs(beta0) < 1.0e-300) beta0 = 1.0e-300;
        c[0] = (m > 1 ? b[0] / beta0 : 0.0);
        d[0] = y[0] / beta0;
        for (int i = 1; i < m; ++i) {
            dp beta = (a[i] - shift) - b[i - 1] * c[i - 1];
            if (std::fabs(beta) < 1.0e-300) beta = 1.0e-300;
            c[i] = (i < m - 1 ? b[i] / beta : 0.0);
            d[i] = (y[i] - b[i - 1] * d[i - 1]) / beta;
        }
        y[m - 1] = d[m - 1];
        for (int i = m - 2; i >= 0; --i) y[i] = d[i] - c[i] * y[i + 1];

        dp nrm = 0.0;
        for (dp v : y) nrm += v * v;
        nrm = std::sqrt(nrm);
        if (!(nrm > 0.0) || !std::isfinite(nrm)) return std::vector<dp>(m, 1.0 / std::sqrt((dp)m));
        for (dp& v : y) v /= nrm;
    }
    return y;
}

struct LanczosResult {
    dp rmin = 0.0, rmax = 0.0;
    std::vector<dp> vmin, vmax;   // extremal Ritz vectors (length n)
    int steps = 0;
};

// m-step Lanczos with full reorthogonalisation, started from `seed`.
// Returns the extreme Ritz values and their Ritz vectors.
LanczosResult lanczos_extremes(const Matrix& D, int m, const std::vector<dp>& seed) {
    const int n = D.n;
    m = std::min(m, n);

    std::vector<std::vector<dp>> V;
    V.reserve(m);
    std::vector<dp> v = (static_cast<int>(seed.size()) == n) ? seed : default_seed(n);
    { dp nv = std::sqrt(dot(v, v));
      if (!(nv > 0.0)) { v = default_seed(n); nv = std::sqrt(dot(v, v)); }
      for (auto& x : v) x /= nv; }

    std::vector<dp> alpha, beta;
    std::vector<dp> vprev(n, 0.0);
    dp b = 0.0;
    for (int j = 0; j < m; ++j) {
        V.push_back(v);
        std::vector<dp> w = matvec(D, v);
        const dp a = dot(v, w);
        alpha.push_back(a);
        for (int i = 0; i < n; ++i) w[i] -= a * v[i] + b * vprev[i];
        for (const auto& u : V) {                 // full reorthogonalisation
            const dp d = dot(w, u);
            for (int i = 0; i < n; ++i) w[i] -= d * u[i];
        }
        b = std::sqrt(dot(w, w));
        if (b < 1.0e-10) break;                   // invariant subspace found
        vprev = v;
        for (int i = 0; i < n; ++i) v[i] = w[i] / b;
        beta.push_back(b);
    }

    LanczosResult out;
    const int k = static_cast<int>(alpha.size());
    out.steps = k;
    if (k == 1) {
        out.rmin = out.rmax = alpha[0];
        out.vmin = out.vmax = V[0];
        return out;
    }
    std::vector<dp> bb(beta.begin(), beta.begin() + (k - 1));

    // Gershgorin bracket of the tridiagonal for the bisection interval.
    dp lo = alpha[0], hi = alpha[0];
    for (int i = 0; i < k; ++i) {
        const dp rad = (i > 0 ? std::fabs(bb[i - 1]) : 0.0) +
                       (i < k - 1 ? std::fabs(bb[i]) : 0.0);
        lo = std::min(lo, alpha[i] - rad);
        hi = std::max(hi, alpha[i] + rad);
    }

    dp a1 = lo, b1 = hi;                           // smallest: count(>=1) boundary
    for (int it = 0; it < 100; ++it) {
        const dp mid = 0.5 * (a1 + b1);
        if (sturm_count(alpha, bb, mid) >= 1) b1 = mid; else a1 = mid;
    }
    dp a2 = lo, b2 = hi;                           // largest: count(>=k) boundary
    for (int it = 0; it < 100; ++it) {
        const dp mid = 0.5 * (a2 + b2);
        if (sturm_count(alpha, bb, mid) >= k) b2 = mid; else a2 = mid;
    }
    out.rmin = 0.5 * (a1 + b1);
    out.rmax = 0.5 * (a2 + b2);

    // Lift the tridiagonal eigenvectors back to the full space: v = V^T y.
    const std::vector<dp> ymin = tridiag_eigenvector(alpha, bb, out.rmin);
    const std::vector<dp> ymax = tridiag_eigenvector(alpha, bb, out.rmax);
    out.vmin.assign(n, 0.0);
    out.vmax.assign(n, 0.0);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) {
            out.vmin[i] += ymin[j] * V[j][i];
            out.vmax[i] += ymax[j] * V[j][i];
        }
    return out;
}

}  // namespace

std::pair<dp, dp> spectral_bounds(const Matrix& D, SpectralCache* cache) {
    const bool can_warm =
        cache && static_cast<int>(cache->start.size()) == D.n;

    LanczosResult r = lanczos_extremes(D, can_warm ? LANCZOS_WARM : LANCZOS_COLD,
                                       can_warm ? cache->start : std::vector<dp>{});
    // A warm start that collapsed into an invariant subspace too early cannot be
    // trusted to have seen both extremes -- redo it cold.
    if (can_warm && r.steps < 4) {
        r = lanczos_extremes(D, LANCZOS_COLD, {});
        if (cache) ++cache->cold_runs;
    } else if (cache) {
        can_warm ? ++cache->warm_runs : ++cache->cold_runs;
    }

    if (cache) {
        // Seed the next step with a vector rich in BOTH extremal directions. Using
        // vmin alone would risk immediate Lanczos breakdown (if it is already an
        // near-exact eigenvector) and would then never see lambda_max; the small
        // deterministic admixture guarantees the Krylov space cannot degenerate.
        const int n = D.n;
        std::vector<dp> s(n);
        const std::vector<dp> u = default_seed(n);
        dp un = std::sqrt(dot(u, u));
        for (int i = 0; i < n; ++i)
            s[i] = r.vmin[i] + r.vmax[i] + 0.01 * u[i] / un;
        const dp sn = std::sqrt(dot(s, s));
        if (sn > 0.0 && std::isfinite(sn)) {
            for (dp& x : s) x /= sn;
            cache->start = std::move(s);
        } else {
            cache->start.clear();
        }
    }

    dp lmin = 0.5 * r.rmin;      // pad below the smallest Ritz value
    dp lmax = 1.02 * r.rmax;     // pad above the largest
    if (lmin < 1.0e-12) lmin = 1.0e-12;
    if (lmin >= lmax) lmin = lmax * 1.0e-6;
    return {lmin, lmax};
}

// Chebyshev coefficients for sqrt on the interval mapped to [-1,1] by (da, db).
std::vector<dp> sqrt_chebyshev_coefficients(int L, dp da, dp db) {
    std::vector<dp> a(L + 1, 0.0);
    std::vector<dp> xks(L + 1);
    for (int k = 0; k <= L; ++k)
        xks[k] = std::cos(PI * (k + 0.5) / (L + 1)) / da - db / da;
    for (int j = 0; j <= L; ++j) {
        for (int k = 0; k <= L; ++k)
            a[j] += std::sqrt(xks[k]) * std::cos(j * (k + 0.5) * PI / (L + 1));
        a[j] *= 2.0 / (L + 1);
    }
    a[0] /= 2.0;
    return a;
}

// One Chebyshev-series evaluation of D^{1/2}.x0 with `L` terms, on the spectral
// interval [lmin, lmax]. Uses the shifted operator D' = da D + db I (spectrum in
// [-1,1]) via the three-term recurrence.
std::vector<dp> chebyshev_series(const Matrix& D, const std::vector<dp>& x0, int L,
                                 dp lmin, dp lmax) {
    const dp da = 2.0 / (lmax - lmin);
    const dp db = -(lmax + lmin) / (lmax - lmin);
    const std::vector<dp> a = sqrt_chebyshev_coefficients(L, da, db);

    auto shifted = [&](const std::vector<dp>& v) {   // D' v = da (D v) + db v
        std::vector<dp> Dv = matvec(D, v);
        std::vector<dp> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = da * Dv[i] + db * v[i];
        return out;
    };

    std::vector<dp> result(x0.size());
    for (size_t i = 0; i < x0.size(); ++i) result[i] = a[0] * x0[i];

    std::vector<dp> t_prev = x0;        // T_0
    std::vector<dp> t_cur = shifted(x0);  // T_1
    for (size_t i = 0; i < result.size(); ++i) result[i] += a[1] * t_cur[i];

    for (int l = 2; l <= L; ++l) {
        std::vector<dp> t_next = shifted(t_cur);
        for (size_t i = 0; i < t_next.size(); ++i) t_next[i] = 2.0 * t_next[i] - t_prev[i];
        t_prev = std::move(t_cur);
        t_cur = std::move(t_next);
        for (size_t i = 0; i < result.size(); ++i) result[i] += a[l] * t_cur[i];
    }
    return result;
}

namespace {
}  // namespace

ChebyshevResult chebyshev_sqrt_times(const Matrix& D, const Vec3Field& X0,
                                     dp nterm_multiplier, dp fd_tol,
                                     SpectralCache* cache) {
    const std::vector<dp> x0 = flatten(X0);
    const auto [lmin, lmax] = spectral_bounds(D, cache);
    const std::vector<dp> Dx0 = matvec(D, x0);
    const dp x0Dx0 = dot(x0, Dx0);

    int L = static_cast<int>(std::sqrt(lmax / lmin) * nterm_multiplier + 0.5) + 1;
    L = std::max(2, std::min(L, MAXCHEB));

    std::vector<dp> result;
    dp fd_err = 0.0;
    while (true) {
        result = chebyshev_series(D, x0, L, lmin, lmax);
        fd_err = std::fabs(dot(result, result) - x0Dx0) / x0Dx0;
        if (fd_err <= fd_tol || L >= MAXCHEB) break;
        L = std::min(MAXCHEB, L + std::max(1, L / 4));  // grow until FD is satisfied
    }
    return {unflatten(result), L, fd_err};
}

} // namespace bdsim
