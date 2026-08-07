// model.hpp — the physical model of the problem.
//
// PhysParams is the user-facing description. PhysicalModel is the prepared model
// built from it: it owns the force objects, the flow, the mobility, and the
// chain size, and answers the questions the integrator asks (spring force,
// non-spring force, mobility, kappa). The integrator holds only a PhysicalModel
// and a SimParams -- all physics is here.
#pragma once

#include "bending.hpp"
#include "config.hpp"
#include "excluded_volume.hpp"
#include "external_force.hpp"
#include "flow.hpp"
#include "mobility.hpp"
#include "spring.hpp"

namespace bdsim {

struct PhysParams {
    SpringParams spring;
    EVParams ev;
    BendingParams bend;
    Flow flow;
    ExternalForce external;   // user-applied bead forces (default: none)
    int number_of_beads = 10;

    // Hydrodynamic interaction.
    dp hstar = 0.0;                                 // 0 => free draining
    DelSMethod hi_method = DelSMethod::Chebyshev;   // B.dW method when hstar > 0
    dp ncheb_multiplier = 1.0;                      // Chebyshev term-count multiplier
    dp fd_err_max = 0.0025;                         // Chebyshev fluctuation-dissipation tol

    bool equilibration = false;  // EV good-solvent flag
};

class PhysicalModel {
public:
    explicit PhysicalModel(const PhysParams& p)
        : nbeads_(p.number_of_beads),
          spring_(p.spring),
          flow_(p.flow),
          ev_(p.ev, p.equilibration, p.number_of_beads),
          bending_(p.bend),
          external_(p.external),
          mobility_(p.hstar, p.hi_method, p.ncheb_multiplier, p.fd_err_max) {}

    int nbeads() const { return nbeads_; }
    const SpringParams& spring() const { return spring_; }
    const Mobility& mobility() const { return mobility_; }
    Mat3 kappa(dp t) const { return flow_.kappa(t); }

    // Conservative intramolecular forces (excluded volume + bending). Kept free of
    // the external force so that it still means "the chain's own force" -- which is
    // what the Kramers stress and the Fortran cross-check compare against.
    Vec3Field non_spring_force(const ChainState& state) const {
        return ev_.force(state) + bending_.force(state);
    }

    // User-applied forces at time t (zero unless something was added).
    Vec3Field external_force(dp t) const {
        if (external_.empty()) return Vec3Field(nbeads_, Vec3{0.0, 0.0, 0.0});
        return external_.force(nbeads_, t);
    }
    bool has_external() const { return !external_.empty(); }

private:
    int nbeads_;
    SpringParams spring_;
    Flow flow_;
    ExcludedVolume ev_;
    BendingForce bending_;
    ExternalForce external_;
    Mobility mobility_;
};

} // namespace bdsim
