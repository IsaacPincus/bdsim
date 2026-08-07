// mobility.hpp — the hydrodynamic mobility of the chain.
//
// The Ito SDE integrated by the code is
//     dR = [ K.R + (1/4) D.F ] dt* + (1/sqrt2) B.dW,      D = B.B^T.
//
// Mobility is the configuration-independent model (hstar + square-root method).
// Given a configuration R it produces a Diffusion: the actual operator D at that
// configuration, which can apply D to a force field and generate B.dW. D is
// built once per timestep and reused across the predictor/corrector.
//
//   * hstar = 0            -> D = I (free draining)
//   * hstar > 0, Cholesky  -> B = lower Cholesky factor of the RPY tensor
//   * hstar > 0, Chebyshev -> B.dW via Fixman's Chebyshev approximation of D^1/2
#pragma once

#include "hydrodynamics.hpp"
#include "linalg.hpp"
#include "rng.hpp"
#include "vec.hpp"

namespace bdsim {

class Diffusion {
public:
    static Diffusion identity(int nbeads);
    // `cache` (may be null) warm-starts the Chebyshev spectral-bound estimation
    // across timesteps. It is borrowed, not owned: it must outlive the Diffusion,
    // which is why it lives in the Mobility below rather than here.
    static Diffusion rpy(const Vec3Field& R, dp hstar, DelSMethod method,
                         dp cheb_multiplier, dp fd_tol,
                         SpectralCache* cache = nullptr);

    Vec3Field apply(const Vec3Field& F) const;       // D . F
    Vec3Field brownian_step(Rng& rng, dp dt) const;  // B . dW

private:
    bool identity_ = true;
    int nbeads_ = 0;
    DelSMethod method_ = DelSMethod::Cholesky;
    dp cheb_multiplier_ = 1.0, fd_tol_ = 0.0025;
    Matrix D_;   // full RPY tensor (apply, Chebyshev)
    Matrix L_;   // Cholesky factor (Cholesky Brownian step)
    SpectralCache* cache_ = nullptr;   // borrowed; see rpy() above
};

class Mobility {
public:
    Mobility(dp hstar, DelSMethod method, dp cheb_multiplier, dp fd_tol)
        : hstar_(hstar), method_(method),
          cheb_multiplier_(cheb_multiplier), fd_tol_(fd_tol) {}

    Diffusion at(const Vec3Field& R) const {
        if (hstar_ <= 0.0) return Diffusion::identity(static_cast<int>(R.size()));
        return Diffusion::rpy(R, hstar_, method_, cheb_multiplier_, fd_tol_, &cache_);
    }

    // Diagnostics: how many spectral-bound estimates were warm- vs cold-started.
    const SpectralCache& spectral_cache() const { return cache_; }

private:
    dp hstar_;
    DelSMethod method_;
    dp cheb_multiplier_, fd_tol_;
    // Carried between timesteps to warm-start the bound estimation. Mutable
    // because it is a pure performance cache: `at()` is logically const, and the
    // fluctuation-dissipation check plus the Cholesky fallback guarantee the
    // result is correct regardless of what the cache holds.
    mutable SpectralCache cache_;
};

} // namespace bdsim
