#include "spring.hpp"

#include "numerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace bdsim {

dp force_sans_hookean(Spring sptype, dp q, dp q0) {
    switch (sptype) {
        case Spring::Hook:
            return 1.0;
        case Spring::FENE:
            return 1.0 / (1.0 - q * q);
        case Spring::ILC:  // inverse Langevin, Pade
            return (3.0 - q * q) / (1.0 - q * q) / 3.0;
        case Spring::WLC:  // Marko-Siggia
            return 1.0 / (6.0 * q) * (4.0 * q + 1.0 / (1.0 - q) / (1.0 - q) - 1.0);
        case Spring::Fraenkel:
            return 1.0 * (1.0 - (q0 / q));
        case Spring::FENEFraenkel:
            return (1.0 - q0 / q) / (1.0 - (q - q0) * (q - q0));
        case Spring::WLCbounded: {
            const dp t = (q - q0) / (1.0 - q0);
            return (2.0 / (3.0 * q)) *
                   ((std::pow(1.0 - t, -2.0) - 1.0) / 4.0 + t -
                    q0 * (std::pow(1.0 + t, -2.0) - 1.0) / 4.0 + q0 * t);
        }
    }
    std::fprintf(stderr, "spring type not correctly set\n");
    return 1.0;
}

// ---------------------------------------------------------------------------
// The per-step implicit equation is   r ( 1 + (dt/4) f(r) ) = gama,
// where f = force_sans_hookean. Different force laws f give equations of very
// different character, so they are solved by three different techniques:
//
//   * closed form        (Hookean, Fraenkel)              -- direct formula
//   * cubic polynomial   (FENE, ILC, WLC, FENE-Fraenkel)  -- analytic root or
//                                                            guess + Newton polish
//   * transcendental     (modified WLC / WLCbounded)      -- rtsafe root find
//
// The helpers below isolate the cubic-family setup; solve_implicit_r dispatches.
// ---------------------------------------------------------------------------
namespace {

// Coefficients c0..c3 of the cubic  c0 + c1 r + c2 r^2 + c3 r^3 = 0  whose
// physical root is the implicit bond length, for each cubic-family law.
std::array<dp, 4> cubic_coefficients(Spring type, dp dtby4, dp gama, dp natscl) {
    std::array<dp, 4> c{};
    c[3] = 1.0;
    switch (type) {
        case Spring::FENE:
            c[0] = gama;
            c[1] = -(1.0 + dtby4);
            c[2] = -gama;
            break;
        case Spring::ILC: {
            const dp d = 3.0 + dtby4;
            c[0] = 3.0 * gama / d;
            c[1] = -(3.0 + 3.0 * dtby4) / d;
            c[2] = -3.0 * gama / d;
            break;
        }
        case Spring::WLC: {
            const dp d = 2.0 * (1.5 + dtby4);
            c[0] = -3.0 * gama / d;
            c[1] = 3.0 * (1.0 + dtby4 + 2.0 * gama) / d;
            c[2] = -1.5 * (4.0 + 3.0 * dtby4 + 2.0 * gama) / d;
            break;
        }
        case Spring::FENEFraenkel:
            c[0] = dtby4 * natscl + gama * (1.0 - natscl * natscl);
            c[1] = -1.0 + natscl * natscl - dtby4 + 2.0 * gama * natscl;
            c[2] = -(2.0 * natscl + gama);
            break;
        default:
            break;
    }
    return c;
}

// Large-gama asymptotic guess for the Newton-polished laws (r -> 1 as gama grows).
dp asymptotic_guess(Spring type, dp dtby4, dp gama) {
    switch (type) {
        case Spring::FENE: return 1.0 - dtby4 / 2.0 / gama;
        case Spring::ILC:  return 1.0 - dtby4 / 3.0 / gama;
        case Spring::WLC:  return 1.0 - std::sqrt(dtby4 / 6.0 / gama);
        default:           return gama / (1.0 + dtby4);
    }
}

// Root of the modified-WLC (bounded) implicit equation via rtsafe. The bracket
// is the physical range of the reduced bond length.
dp solve_wlc_bounded(dp dtby4, dp gama, dp sigma) {
    if (sigma > 1.0) {
        std::fprintf(stderr, "sigma/L cannot be greater than 1 for WLC bounded\n");
        std::abort();
    }
    const dp dt = dtby4 * 4.0;
    auto value_and_deriv = [dt, sigma, gama](dp x, dp& fval, dp& fderiv) {
        const dp s = sigma;
        fval = 6.0 * x / dt + 0.25 * (1.0 - s) * (1.0 - s) / ((1.0 - x) * (1.0 - x)) -
               0.25 + (x - s) / (1.0 - s) -
               s / 4.0 * (1.0 - s) * (1.0 - s) / ((1.0 + x - 2.0 * s) * (1.0 + x - 2.0 * s)) +
               s / 4.0 + s * (x - s) / (1.0 - s) - 6.0 * gama / dt;
        fderiv = 6.0 / dt + 1.0 / (1.0 - s) + s / (1.0 - s) +
                 (1.0 - s) * (1.0 - s) / (2.0 * std::pow(1.0 - x, 3.0)) +
                 ((1.0 - s) * (1.0 - s) * s) / (2.0 * std::pow(1.0 - 2.0 * s + x, 3.0));
    };
    const dp eps = std::numeric_limits<dp>::epsilon();
    return rtsafe(value_and_deriv, std::max(0.0, 2.0 * sigma - 1.0 + eps), 1.0 - eps, 1e-15);
}

}  // namespace

SpringSolution solve_implicit_r(Spring sptype, dp dtby4, dp gama, dp natscl) {
    switch (sptype) {
        // ---- closed form ----
        case Spring::Hook:
            return {gama / (1.0 + dtby4), 1.0};
        case Spring::Fraenkel: {
            const dp r = (gama + dtby4 * natscl) / (1.0 + dtby4);
            return {r, 1.0 - natscl / r};
        }
        // ---- transcendental ----
        case Spring::WLCbounded: {
            const dp r = solve_wlc_bounded(dtby4, gama, natscl);
            return {r, force_sans_hookean(sptype, r, natscl)};
        }
        // ---- cubic family: handled below ----
        case Spring::FENE:
        case Spring::ILC:
        case Spring::WLC:
        case Spring::FENEFraenkel:
            break;
    }

    const std::array<dp, 4> coeff = cubic_coefficients(sptype, dtby4, gama, natscl);

    dp r;
    if (sptype == Spring::FENEFraenkel) {
        // exactly one root lies in the physical bond range [max(sigma-1,0), sigma+1]
        const dp lo = std::max(natscl - 1.0, 0.0), hi = natscl + 1.0;
        r = find_roots_cubic(coeff, lo, hi);
        r = polish_poly_root(coeff, r, 1e-14);
        // Keep the bond strictly inside the finite-extensibility singularities at
        // r = q0 +/- 1 (where the force diverges), so the spring force stays finite
        // for any gama -- the semi-implicit step then never produces an unbounded
        // bond even when the Brownian forcing is momentarily huge.
        const dp margin = 1e-6;
        r = std::min(std::max(r, lo + margin), hi - margin);
    } else {
        // FENE / ILC / WLC: Hookean guess for small gama, asymptotic guess otherwise
        r = (gama < 1.0) ? gama / (1.0 + dtby4) : asymptotic_guess(sptype, dtby4, gama);
        // Newton-polish -- except WLC at large gama, where Newton wanders off the
        // physical root and the asymptotic guess is already good enough.
        if (!(sptype == Spring::WLC && gama > 100.0)) r = polish_poly_root(coeff, r, 1e-14);
        // These laws have their finite-extensibility singularity at reduced length 1;
        // keep r strictly inside [0, 1) so the force stays finite for any gama.
        const dp margin = 1e-6;
        r = std::min(std::max(r, 0.0), 1.0 - margin);
    }

    return {r, force_sans_hookean(sptype, r, natscl)};
}

Vec3 solve_connector(const Vec3& gamma, dp dt, const SpringParams& sp) {
    const dp sqrtb = sp.sqrtb;
    const dp q0 = sp.reduced_natural_length();
    const dp gmag = norm(gamma);
    dp r = solve_implicit_r(sp.type, dt / 4.0, gmag / sqrtb, q0).length;
    if (std::fabs(r - 1.0) < 1e-6) r = 1.0 - 1e-6;
    return (r * sqrtb / gmag) * gamma;
}

Vec3Field connector_forces(const ChainGeometry& g, const SpringParams& sp) {
    const dp sqrtb = sp.sqrtb;
    const dp q0 = sp.reduced_natural_length();
    Vec3Field Fbond(g.bond.size());
    for (size_t i = 0; i < g.bond.size(); ++i) {
        dp r = g.length[i] / sqrtb;
        if (std::fabs(r - 1.0) < MYEPS) r = 1.0 - MYEPS;
        Fbond[i] = force_sans_hookean(sp.type, r, q0) * g.bond[i];
    }
    return Fbond;
}

Vec3Field bead_forces_from_connectors(const Vec3Field& bond_force) {
    const int N = static_cast<int>(bond_force.size()) + 1;
    Vec3Field Fbead(N, Vec3{0.0, 0.0, 0.0});
    if (N == 1) return Fbead;
    Fbead[0] = bond_force[0];
    for (int nu = 1; nu < N - 1; ++nu) Fbead[nu] = bond_force[nu] - bond_force[nu - 1];
    Fbead[N - 1] = (-1.0) * bond_force[N - 2];
    return Fbead;
}

Vec3Field spring_force(const ChainGeometry& g, const SpringParams& sp) {
    return bead_forces_from_connectors(connector_forces(g, sp));
}

} // namespace bdsim
