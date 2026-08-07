// types.hpp — core numeric types, constants, and enums.
// Ported from modules.f90 (Global_parameters_variables_and_types).
#pragma once

#include <cstdint>

namespace bdsim {

// Precision. In the Fortran, DBprec == DOBL == double, so the whole engine is
// double precision. (The one deliberate exception is the RNG scale factor, which
// is single precision in Fortran — see rng.hpp.)
using dp = double;

// Fortran's Selected_int_kind(9) -> 32-bit integer. Seeds/counters use this.
using i32 = std::int32_t;

// ---- Physical / numerical constants (modules.f90) ----
inline constexpr int Ndim = 3;              // spatial dimensions
inline constexpr int NProps = 21;           // number of scalar properties sampled
inline constexpr int MAXCHEB = 500;         // max Chebyshev terms (HI, Phase 2)

inline constexpr dp PI    = 3.14159265358979323846;
inline constexpr dp TINI  = 1e-25;
inline constexpr dp MYEPS = 1e-6;

// ---- Spring force laws ----
enum class Spring : int {
    Hook         = 1,
    FENE         = 2,
    ILC          = 3,  // inverse Langevin (Pade)
    WLC          = 4,  // worm-like chain (Marko-Siggia)
    Fraenkel     = 5,
    FENEFraenkel = 6,
    WLCbounded   = 7,
};

// ---- Excluded volume types ----
enum class EV : int {
    None        = 0,
    Gauss       = 1,
    LJ          = 2,
    SDK         = 3,
    SDKstickers = 4,
};

// ---- Bending potential ----
enum class Bending : int {
    None             = 0,
    OneMinusCosTheta = 1,
};

// ---- Hydrodynamic interaction options (Phase 2) ----
enum class DelSMethod : int { Chebyshev = 0, Cholesky = 1, ExactSqrt = 2 };
enum class EigMethod  : int { Fixman = 0, Exact = 1 };
enum class ChebUpdate : int { New = 0, AddOne = 1 };

} // namespace bdsim
