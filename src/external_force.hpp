// external_force.hpp — user-applied forces on individual beads.
//
// An external force is not part of the chain's own physics: it is whatever the
// experiment applies (optical tweezers, a magnetic bead, a pulling protocol). It
// therefore lives apart from the conservative intramolecular forces, and is added
// to the explicit part of the step -- exactly like excluded volume or bending, but
// depending on time rather than on configuration.
//
// Each entry is a force on one bead, either constant or given as a table of
// (time, vector) samples that is linearly interpolated and clamped outside the
// tabulated range -- the same convention as Flow, so a pulling protocol and a flow
// protocol are specified the same way.
//
// The canonical use is a stretching experiment: equal and opposite forces on the
// two ends, which leaves the net force zero so the chain does not drift.
#pragma once

#include "vec.hpp"

#include <vector>

namespace bdsim {

struct BeadForce {
    int bead = 0;                 // bead index; negative counts from the end (-1 = last)
    bool constant = true;
    Vec3 value{};                 // constant force
    std::vector<dp> times;        // time-varying samples (strictly ascending)
    std::vector<Vec3> values;
};

class ExternalForce {
public:
    ExternalForce() = default;

    // Constant force on a bead.
    void add_constant(int bead, const Vec3& F);

    // Time-varying force on a bead: values[i] applies at times[i], linearly
    // interpolated in between and held constant outside.
    void add_time_varying(int bead, std::vector<dp> times, std::vector<Vec3> values);

    // Equal and opposite forces (+F on `bead_plus`, -F on `bead_minus`): a pure
    // stretch with no net force on the chain.
    void add_stretch(int bead_minus, int bead_plus, const Vec3& F);

    bool empty() const { return entries_.empty(); }

    // Force on every bead at time t (length nbeads; zero where nothing is applied).
    Vec3Field force(int nbeads, dp t) const;

    const std::vector<BeadForce>& entries() const { return entries_; }
    void set_entries(std::vector<BeadForce> e) { entries_ = std::move(e); }

private:
    std::vector<BeadForce> entries_;
};

} // namespace bdsim
