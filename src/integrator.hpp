// integrator.hpp — Brownian dynamics time integrator.
//
// Integrates the Ito SDE for the chain configuration
//
//     dR = [ K.R + (1/4) D.F ] dt* + (1/sqrt2) B.dW
//
// with a semi-implicit predictor-corrector step: the spring force is solved
// implicitly (bond-vector iteration); the non-spring forces (EV + bending) and
// the flow are explicit at predictor/corrector points. All physics lives in the
// PhysicalModel (see model.hpp); this file is the time-stepping scheme.
//
// Sampling / properties / output are intentionally NOT here -- the Python driver
// owns those and integrates in segments (the RNG stream carries across calls).
#pragma once

#include "model.hpp"
#include "rng.hpp"
#include "vec.hpp"

namespace bdsim {

struct SimParams {
    dp time_start = 0.0;
    dp time_end = 1.0;
    dp dt = 0.01;
    dp implicit_loop_tol = 1e-3;   // bond-solve convergence tolerance
    bool update_center_of_mass = true;
};

// Integrate `R` (N bead positions) in place from time_start to time_end.
// Randomness is drawn from `rng`, whose stream persists across calls.
void time_integrate_chain(Vec3Field& R, const PhysParams& phys, const SimParams& sim,
                          Rng& rng);

} // namespace bdsim
