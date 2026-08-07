#include "config.hpp"

namespace bdsim {

namespace {
// Bond length with the Fortran's tiny-distance clamp (avoids a zero-length bond
// on a freshly initialised or degenerate configuration).
dp clamped_length(const Vec3& b) {
    dp sq = norm2(b);
    if (sq < 1.0e-12) sq = 1.0e-12;
    return std::sqrt(sq);
}
}

ChainGeometry chain_geometry(const Vec3Field& positions) {
    const int nbond = static_cast<int>(positions.size()) - 1;
    ChainGeometry g;
    g.bond.resize(nbond);
    g.length.resize(nbond);
    for (int i = 0; i < nbond; ++i) {
        g.bond[i] = positions[i + 1] - positions[i];
        g.length[i] = clamped_length(g.bond[i]);
    }
    return g;
}

ChainGeometry chain_geometry_from_bonds(Vec3Field bonds) {
    ChainGeometry g;
    g.length.resize(bonds.size());
    for (size_t i = 0; i < bonds.size(); ++i) g.length[i] = clamped_length(bonds[i]);
    g.bond = std::move(bonds);
    return g;
}

Vec3Field unit_bonds(const ChainGeometry& g) {
    Vec3Field u(g.bond.size());
    for (size_t i = 0; i < g.bond.size(); ++i) u[i] = (1.0 / g.length[i]) * g.bond[i];
    return u;
}

std::vector<dp> bond_cosines(const Vec3Field& unit) {
    std::vector<dp> cos(unit.size() >= 1 ? unit.size() - 1 : 0);
    for (size_t i = 0; i + 1 < unit.size(); ++i) cos[i] = dot(unit[i + 1], unit[i]);
    return cos;
}

} // namespace bdsim
