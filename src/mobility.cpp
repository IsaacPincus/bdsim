#include "mobility.hpp"

#include "hydrodynamics.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace bdsim {

namespace {
// Ottinger's non-Gaussian deviate X = (Y-1/2)[c1 (Y-1/2)^2 + c2], Y ~ U[0,1];
// correct 2nd and 4th moments at low cost.
constexpr dp c1 = 14.14855378;
constexpr dp c2 = 1.21569221;

// Uncorrelated noise field X0 (without the sqrt(dt) factor), from the ran_1 stream.
Vec3Field raw_noise(Rng& rng, int nbeads) {
    Vec3Field x(nbeads);
    dp u[3];
    for (int mu = 0; mu < nbeads; ++mu) {
        rng.ran_1(3, u);
        for (int d = 0; d < 3; ++d) {
            const dp y = u[d] - 0.5;
            x[mu][d] = (c1 * y * y + c2) * y;
        }
    }
    return x;
}
}  // namespace

Diffusion Diffusion::identity(int nbeads) {
    Diffusion d;
    d.identity_ = true;
    d.nbeads_ = nbeads;
    return d;
}

Diffusion Diffusion::rpy(const Vec3Field& R, dp hstar, DelSMethod method,
                         dp cheb_multiplier, dp fd_tol, SpectralCache* cache) {
    Diffusion d;
    d.identity_ = false;
    d.nbeads_ = static_cast<int>(R.size());
    d.method_ = method;
    d.cheb_multiplier_ = cheb_multiplier;
    d.fd_tol_ = fd_tol;
    d.cache_ = cache;
    d.D_ = rpy_diffusion_matrix(R, hstar);
    if (method == DelSMethod::Cholesky) d.L_ = cholesky_lower(d.D_);
    return d;
}

Vec3Field Diffusion::apply(const Vec3Field& F) const {
    if (identity_) return F;
    return unflatten(matvec(D_, flatten(F)));
}

Vec3Field Diffusion::brownian_step(Rng& rng, dp dt) const {
    const Vec3Field x0 = raw_noise(rng, nbeads_);
    const dp sqrt_dt = std::sqrt(dt);

    if (identity_) return sqrt_dt * x0;

    Vec3Field bx0;
    switch (method_) {
        case DelSMethod::Cholesky:
            bx0 = unflatten(lower_matvec(L_, flatten(x0)));
            break;
        case DelSMethod::Chebyshev: {
            const ChebyshevResult cr = chebyshev_sqrt_times(D_, x0, cheb_multiplier_, fd_tol_, cache_);
            if (cr.fd_error <= fd_tol_) {
                bx0 = cr.value;
            } else {
                // The Chebyshev series failed to satisfy fluctuation-dissipation
                // (the spectral bounds missed part of the spectrum, so the series
                // is diverging). Fall back to the exact Cholesky factor for this
                // step -- correctness over speed. Rare in practice.
                const Matrix Lfb = cholesky_lower(D_);
                bx0 = unflatten(lower_matvec(Lfb, flatten(x0)));
            }
            break;
        }
        case DelSMethod::ExactSqrt:
            std::fprintf(stderr, "ExactSqrt square-root method not implemented\n");
            std::abort();
    }
    return sqrt_dt * bx0;
}

} // namespace bdsim
