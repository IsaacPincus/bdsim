// hydrodynamics.hpp — RPY diffusion tensor and the Chebyshev square root.
#pragma once

#include "linalg.hpp"
#include "vec.hpp"

#include <utility>
#include <vector>

namespace bdsim {

// RPY diffusion tensor (3N x 3N) for the given configuration.
Matrix rpy_diffusion_matrix(const Vec3Field& R, dp hstar);

struct ChebyshevResult {
    Vec3Field value;   // B . X0
    int nterms;
    dp fd_error;       // | <BX0,BX0> - <X0, D X0> | / <X0, D X0>
};

// Warm-start state for the spectral-bound estimation, carried between timesteps.
//
// The Chebyshev interval is built from Lanczos estimates of the extreme
// eigenvalues of D. Over one timestep the configuration -- and hence the
// spectrum and its extremal eigenvectors -- barely changes, so seeding Lanczos
// with the previous step's extremal Ritz vectors reaches the same accuracy in
// far fewer iterations than a cold start. `start` holds that seed; it is empty
// (or stale in size) on the first step of a run, which triggers a cold start.
//
// This is the only mutable state in the mobility path. It affects performance
// only: the fluctuation-dissipation check and the Cholesky fallback still police
// correctness, so a bad seed costs time, never accuracy.
struct SpectralCache {
    std::vector<dp> start;   // seed vector (empty => cold start)
    long cold_runs = 0;      // diagnostics
    long warm_runs = 0;
};

// Approximate D^{1/2} . X0 by a Chebyshev polynomial in D (Fixman's method):
// spectral bounds from a short Lanczos run, Chebyshev coefficients for sqrt on
// that interval, then the three-term recurrence applied to X0. The term count is
// grown until the fluctuation-dissipation identity is satisfied to `fd_tol`.
//
// `cache`, if non-null, is used to warm-start the bound estimation and is updated
// in place. Pass nullptr for a self-contained (cold) evaluation.
ChebyshevResult chebyshev_sqrt_times(const Matrix& D, const Vec3Field& X0,
                                     dp nterm_multiplier, dp fd_tol,
                                     SpectralCache* cache = nullptr);

// The spectral interval [lmin, lmax] the Chebyshev series is built on. Exposed so
// tests can assert the essential safety property: this interval must contain the
// whole spectrum of D, warm-started or not.
std::pair<dp, dp> spectral_bounds(const Matrix& D, SpectralCache* cache);

} // namespace bdsim
