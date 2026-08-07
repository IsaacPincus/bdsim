// flow.hpp — simplified flow field.
//
// The Fortran had 12 flow-type cases plus harmonic traps. Per the port's design
// we collapse all of it to: supply the velocity-gradient tensor kappa directly.
// It is either constant in time, or given as a table of (time, kappa) samples
// that we linearly interpolate. Out-of-range times clamp to the end samples.
//
// Any of the old named flows is expressed by the caller as a kappa tensor, e.g.
// uniaxial extension at rate g -> diag(g, -g/2, -g/2); simple shear -> K[0][1]=g.
#pragma once

#include "vec.hpp"
#include <vector>

namespace bdsim {

class Flow {
public:
    Flow() : constant_(true), K0_{} {}  // default: zero (equilibrium)

    static Flow constant(const Mat3& K) {
        Flow f; f.constant_ = true; f.K0_ = K; return f;
    }
    // times must be strictly ascending; Ks[i] is kappa at times[i].
    static Flow time_varying(std::vector<dp> times, std::vector<Mat3> Ks) {
        Flow f; f.constant_ = false; f.times_ = std::move(times); f.Ks_ = std::move(Ks);
        return f;
    }

    Mat3 kappa(dp t) const;

    // Accessors (used for serialisation / pickling of the parameters).
    bool is_constant() const { return constant_; }
    const Mat3& constant_kappa() const { return K0_; }
    const std::vector<dp>& sample_times() const { return times_; }
    const std::vector<Mat3>& sample_tensors() const { return Ks_; }

private:
    bool constant_;
    Mat3 K0_;
    std::vector<dp> times_;
    std::vector<Mat3> Ks_;
};

} // namespace bdsim
