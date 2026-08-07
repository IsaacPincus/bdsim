// bending.hpp — bending potential force (OneMinusCosTheta) as a force object.
#pragma once

#include "config.hpp"
#include "vec.hpp"

namespace bdsim {

struct BendingParams {
    Bending type = Bending::None;
    dp stiffness = 0.0;
};

// A bending force term. force(state) returns the per-bead bending force
// (length N); zero for a 2-bead chain or when the potential is off.
class BendingForce {
public:
    explicit BendingForce(BendingParams params) : p_(params) {}
    Vec3Field force(const ChainState& state) const;

private:
    BendingParams p_;
};

} // namespace bdsim
