// excluded_volume.hpp — pairwise excluded-volume force as a force object.
//
// Potentials: None, Gauss, LJ, SDK. (SDK-with-stickers is deferred.)
//
// `equilibration` selects the good-solvent (repulsive-only) branch; it is fixed
// for a run, so it is baked into the object rather than passed per call. This is
// the old Fortran "inteq" flag, decoupled from the flow type.
#pragma once

#include "config.hpp"
#include "vec.hpp"

#include <vector>

namespace bdsim {

struct EVParams {
    EV type = EV::None;
    dp zstar = 0.0;               // dimensionless EV energy
    dp dstar = 0.0;               // dimensionless EV radius
    dp min_cutoff = 0.7;
    dp max_cutoff = 1.5;
    int contour_dist_for_EV = 1;  // skip bead pairs closer than this along the chain
    std::vector<dp> phi;          // SDK attractive strengths, row-major N*N (else zeros)
};

class ExcludedVolume {
public:
    ExcludedVolume(const EVParams& params, bool equilibration, int nbeads);

    // Excluded-volume force on each bead (length N), from the bead positions.
    Vec3Field force(const ChainState& state) const;

private:
    dp gauss_factor(dp r) const;
    dp lj_factor(dp r) const;
    dp sdk_factor(dp r, dp phi_pair) const;

    EV type_;
    int cd_;
    bool equilibration_;
    dp zsbyds5_ = 0, pt5bydssq_ = 0;
    dp LJa_ = 0, LJb_ = 0, SDKa_ = 0, SDKb_ = 0;
    dp Rmin_ = 0.7, Rcutg_ = 0, Rcutp_ = 1.5, RcutpSDK_ = 1.5;
    std::vector<dp> phi_;  // N*N attractive strengths (SDK)
};

} // namespace bdsim
