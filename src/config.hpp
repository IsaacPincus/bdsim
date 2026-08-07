// config.hpp — chain geometry: the adjacent bond vectors and their lengths,
// plus the derived unit bonds and bend cosines. (Replaces the Fortran
// "superdiagonal" bead-to-bead matrices; only adjacent bonds are ever needed by
// the spring and bending forces — excluded volume works directly from R.)
#pragma once

#include "vec.hpp"

namespace bdsim {

// Bonds between successive beads. For an N-bead chain there are N-1 bonds.
struct ChainGeometry {
    Vec3Field bond;          // bond[i] = R[i+1] - R[i]
    std::vector<dp> length;  // length[i] = |bond[i]|, clamped away from zero

    int nbeads() const { return static_cast<int>(bond.size()) + 1; }
};

// Geometry from bead positions.
ChainGeometry chain_geometry(const Vec3Field& positions);

// Geometry from a set of bond (connector) vectors directly — used by the
// implicit connector solve, which works in bond space.
ChainGeometry chain_geometry_from_bonds(Vec3Field bonds);

// A configuration the forces are evaluated at: bead positions and the derived
// adjacent-bond geometry. A non-owning view -- construct it transiently.
struct ChainState {
    const Vec3Field& R;
    const ChainGeometry& geom;
};

// Unit vector along each bond (length N-1).
Vec3Field unit_bonds(const ChainGeometry& g);

// Cosine of the angle between successive bonds (length N-2).
std::vector<dp> bond_cosines(const Vec3Field& unit);

} // namespace bdsim
