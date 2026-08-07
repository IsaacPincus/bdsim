#include "external_force.hpp"

namespace bdsim {

namespace {

// Linear interpolation of a tabulated vector, clamped outside the table.
Vec3 sample(const std::vector<dp>& times, const std::vector<Vec3>& values, dp t) {
    const int n = static_cast<int>(times.size());
    if (n == 0) return Vec3{0.0, 0.0, 0.0};
    if (n == 1 || t <= times.front()) return values.front();
    if (t >= times.back()) return values.back();
    int i = 0;
    while (i < n - 1 && !(t >= times[i] && t < times[i + 1])) ++i;
    const dp w = (t - times[i]) / (times[i + 1] - times[i]);
    return values[i] + w * (values[i + 1] - values[i]);
}

}  // namespace

void ExternalForce::add_constant(int bead, const Vec3& F) {
    BeadForce e;
    e.bead = bead;
    e.constant = true;
    e.value = F;
    entries_.push_back(std::move(e));
}

void ExternalForce::add_time_varying(int bead, std::vector<dp> times,
                                     std::vector<Vec3> values) {
    BeadForce e;
    e.bead = bead;
    e.constant = false;
    e.times = std::move(times);
    e.values = std::move(values);
    entries_.push_back(std::move(e));
}

void ExternalForce::add_stretch(int bead_minus, int bead_plus, const Vec3& F) {
    add_constant(bead_plus, F);
    add_constant(bead_minus, (-1.0) * F);
}

Vec3Field ExternalForce::force(int nbeads, dp t) const {
    Vec3Field F(nbeads, Vec3{0.0, 0.0, 0.0});
    for (const BeadForce& e : entries_) {
        int i = e.bead < 0 ? nbeads + e.bead : e.bead;   // -1 => last bead
        if (i < 0 || i >= nbeads) continue;              // silently ignore out of range
        F[i] = F[i] + (e.constant ? e.value : sample(e.times, e.values, t));
    }
    return F;
}

} // namespace bdsim
