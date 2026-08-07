// spring.hpp — spring force laws and the semi-implicit connector solve.
//
// Reduced units follow the Fortran: a bond of length Q is worked with r = Q/sqrtb
// and a reduced natural length q0 = Q0/sqrtb.
#pragma once

#include "config.hpp"
#include "vec.hpp"

namespace bdsim {

struct SpringParams {
    Spring type = Spring::Hook;
    dp sqrtb = 10.0;          // finite extensibility parameter
    dp natural_length = 0.0;  // Q0 / sigma

    // Natural length in reduced (per-sqrtb) units, as the solvers use it.
    dp reduced_natural_length() const { return natural_length / sqrtb; }
};

// Result of the implicit per-step solve: the reduced bond length and the force
// factor ff at that length.
struct SpringSolution {
    dp length;        // r  (reduced bond length Q/sqrtb)
    dp force_factor;  // ff (so the spring force is H * Q * ff)
};

// Force factor ff in  F = H * Q * ff, at reduced bond length q with reduced
// natural length q0. (Hookean is ff = 1.)
dp force_sans_hookean(Spring sptype, dp q, dp q0);

// Solve  r (1 + dtby4 * ff(r)) = gama  for the reduced bond length r.
SpringSolution solve_implicit_r(Spring sptype, dp dtby4, dp gama, dp natscl);

// Solve one bond of the implicit step: given the RHS Gamma for that bond, return
// the updated bond vector (same direction as Gamma, magnitude from the spring
// solve). Encapsulates the reduced-unit bookkeeping so callers need only the
// spring parameters. dt is the full timestep.
Vec3 solve_connector(const Vec3& gamma, dp dt, const SpringParams& sp);

// Spring force carried by each bond (length N-1).
Vec3Field connector_forces(const ChainGeometry& g, const SpringParams& sp);

// Spread bond forces onto beads: F_bead[nu] = F_bond[nu] - F_bond[nu-1]
// (length N, with the chain-end boundary terms).
Vec3Field bead_forces_from_connectors(const Vec3Field& bond_force);

// Net spring force on each bead (length N).
Vec3Field spring_force(const ChainGeometry& g, const SpringParams& sp);

} // namespace bdsim
