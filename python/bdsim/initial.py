"""Initial chain configurations (replaces the Fortran InitialPos)."""
import numpy as np


def gaussian_chain(n: int, bond_std: float = 1.0, seed=None) -> np.ndarray:
    """A random walk of `n` beads: each bond is an isotropic Gaussian.

    Adequate as a starting configuration; the simulation equilibrates from here.
    Returns an (n, 3) float64 array with the centre of mass at the origin.
    """
    rng = np.random.default_rng(seed)
    bonds = rng.normal(scale=bond_std, size=(n - 1, 3))
    R = np.zeros((n, 3))
    R[1:] = np.cumsum(bonds, axis=0)
    R -= R.mean(axis=0)
    return R


# --------------------------------------------------------------------------
# FENE-Fraenkel equilibrium constructors (port of InitialPos.f90).
#
# The equilibrium bond-length distribution of a FENE-Fraenkel connector is
#     psi(Q) ~ Q^2 (1 - (Q - sigma)^2 / alpha)^(alpha/2),   alpha = dQ^2,
# supported on |Q - sigma| <= dQ (clamped at 0). We sample it by inverting the
# (trapezoidal) cumulative distribution, then attach directions -- either random
# on the unit sphere, or all along x.
# --------------------------------------------------------------------------

def _fene_fraenkel_bond_lengths(n_bonds, sigma, dQ, rng, nsteps=10000):
    alpha = dQ * dQ
    lo = max(0.0, sigma - dQ)
    hi = sigma + dQ
    Q = np.linspace(lo, hi, nsteps)
    Q[0] += 1e-7
    Q[-1] -= 1e-7
    psi = Q * Q * (1.0 - (Q - sigma) ** 2 / alpha) ** (alpha / 2.0)
    # cumulative trapezoidal integral, normalised to a CDF on [0, 1]
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (psi[:-1] + psi[1:]) * np.diff(Q))))
    cdf /= cdf[-1]
    u = rng.random(n_bonds)
    return np.interp(u, cdf, Q)     # inverse-CDF sampling


def _spherical_unit_vectors(n, rng):
    """`n` isotropically distributed unit vectors (Marsaglia's method)."""
    out = np.empty((n, 3))
    filled = 0
    while filled < n:
        m = 2 * (n - filled) + 8
        x = rng.uniform(-1.0, 1.0, size=(m, 2))
        s = x[:, 0] ** 2 + x[:, 1] ** 2
        keep = x[s < 1.0]
        sk = keep[:, 0] ** 2 + keep[:, 1] ** 2
        take = min(len(keep), n - filled)
        root = np.sqrt(1.0 - sk[:take])
        out[filled:filled + take, 0] = 2.0 * keep[:take, 0] * root
        out[filled:filled + take, 1] = 2.0 * keep[:take, 1] * root
        out[filled:filled + take, 2] = 1.0 - 2.0 * sk[:take]
        filled += take
    return out


def fene_fraenkel_chain(n: int, sigma: float, dQ: float, seed=None) -> np.ndarray:
    """FENE-Fraenkel equilibrium chain with randomly oriented bonds.

    `sigma` is the natural length, `dQ` the extensibility about it (bonds lie in
    [sigma - dQ, sigma + dQ]). Returns (n, 3) with bead 1 at the origin.
    """
    rng = np.random.default_rng(seed)
    Ql = _fene_fraenkel_bond_lengths(n - 1, sigma, dQ, rng)
    bonds = _spherical_unit_vectors(n - 1, rng) * Ql[:, None]
    R = np.zeros((n, 3))
    R[1:] = np.cumsum(bonds, axis=0)
    return R


def fene_fraenkel_chain_aligned_x(n: int, sigma: float, dQ: float, seed=None) -> np.ndarray:
    """FENE-Fraenkel equilibrium chain with all bonds along the x-axis."""
    rng = np.random.default_rng(seed)
    Ql = _fene_fraenkel_bond_lengths(n - 1, sigma, dQ, rng)
    R = np.zeros((n, 3))
    R[1:, 0] = np.cumsum(Ql)
    return R


# --------------------------------------------------------------------------
# General equilibrium constructor (arbitrary spring + bending potential).
#
# Builds a chain step by step so that both the spring-length distribution and
# the included-angle distribution match the underlying equilibrium. Spring
# lengths Q_i and bond angles theta_i are sampled from their 1D distributions
# (by inverse-CDF), then assembled: the first bond points isotropically; each
# subsequent bond is placed at angle theta_i to the previous one, with a random
# azimuth about it. This captures bending correlations (unlike a purely
# isotropic random walk).
# --------------------------------------------------------------------------

def inverse_cdf_sample(x, pdf, u):
    """Sample values with density proportional to `pdf` on grid `x`, given
    uniform deviates `u` in [0,1), via the (trapezoidal) inverse CDF."""
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * np.diff(x))))
    cdf /= cdf[-1]
    return np.interp(u, cdf, x)


def isotropic_directions(n, rng):
    """`n` isotropic unit vectors (cos(theta) uniform in [-1,1], azimuth uniform)."""
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def one_minus_cos_theta_angles(n, stiffness, rng):
    """Sample bond angles for the OneMinusCosTheta potential phi_b = C(1-cos theta).

    The equilibrium angle density is psi(theta) ~ sin(theta) exp(-C(1-cos theta));
    with u = cos(theta) this is exp(C u) on [-1,1], sampled analytically.
    """
    C = stiffness
    if C == 0.0:
        return np.arccos(rng.uniform(-1.0, 1.0, n))
    Y = rng.random(n)
    u = np.log(np.exp(-C) + Y * (np.exp(C) - np.exp(-C))) / C
    return np.arccos(np.clip(u, -1.0, 1.0))


def _rotation_z_to(u):
    """Rotation matrix R (Rodrigues) with R @ [0,0,1] == u (u a unit vector)."""
    c = u[2]                       # cos angle between z and u
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.array([-u[1], u[0], 0.0])   # z x u
    s2 = v @ v
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s2)


def equilibrium_chain(lengths, angles=None, seed=None):
    """Assemble a chain from sampled bond `lengths` (len n-1) and included
    `angles` between successive bonds (len n-2; None => isotropic).

    Returns (n, 3) with bead 1 at the origin.
    """
    lengths = np.asarray(lengths, dtype=float)
    n_bonds = len(lengths)
    rng = np.random.default_rng(seed)

    u = np.empty((n_bonds, 3))
    u[0] = isotropic_directions(1, rng)[0]
    for i in range(1, n_bonds):
        theta = np.pi / 2 if angles is None else float(angles[i - 1])
        az = rng.uniform(0.0, 2.0 * np.pi)
        base = np.array([np.sin(theta) * np.cos(az),
                         np.sin(theta) * np.sin(az),
                         np.cos(theta)])          # angle theta from local z
        u[i] = _rotation_z_to(u[i - 1]) @ base    # rotate local frame onto u[i-1]

    bonds = u * lengths[:, None]
    R = np.zeros((n_bonds + 1, 3))
    R[1:] = np.cumsum(bonds, axis=0)
    return R


def fene_fraenkel_bending_chain(n, sigma, dQ, stiffness, seed=None):
    """Equilibrium chain: FENE-Fraenkel bond lengths + OneMinusCosTheta bending."""
    rng = np.random.default_rng(seed)
    lengths = _fene_fraenkel_bond_lengths(n - 1, sigma, dQ, rng)
    angles = one_minus_cos_theta_angles(n - 2, stiffness, rng) if n > 2 else None
    # assemble with a fresh (seeded) rng so directions are reproducible
    return equilibrium_chain(lengths, angles, seed=seed)
