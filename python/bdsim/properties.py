"""Per-configuration chain properties, from an (N, 3) positions array."""
import numpy as np


def end_to_end_sq(R: np.ndarray) -> float:
    """Squared end-to-end distance ``|R_N - R_1|^2``."""
    d = R[-1] - R[0]
    return float(d @ d)


def gyration_tensor(R: np.ndarray) -> np.ndarray:
    """3x3 radius-of-gyration tensor about the centre of mass."""
    Rc = R - R.mean(axis=0)
    return (Rc.T @ Rc) / len(R)


def radius_of_gyration_sq(R: np.ndarray) -> float:
    """Squared radius of gyration = trace of the gyration tensor."""
    return float(np.trace(gyration_tensor(R)))


def stretch(R: np.ndarray) -> np.ndarray:
    """Extent (max - min) of the (centred) chain along each axis."""
    Rc = R - R.mean(axis=0)
    return Rc.max(axis=0) - Rc.min(axis=0)


# --------------------------------------------------------------------------
# Polymer stress (Kramers-Kirkwood) and viscosity.
#
# The polymer contribution to the stress tensor for one chain is
#     tau_ij = sum_mu (R_mu - R_c)_i * F_mu,j
# where F is the intramolecular force on each bead (spring, and optionally EV /
# bending). Pass whichever force you want to attribute the stress to -- e.g. the
# spring force alone, or the total force from bdsim.total_force.
# --------------------------------------------------------------------------

def kramers_stress(R: np.ndarray, F: np.ndarray) -> np.ndarray:
    """3x3 Kramers-Kirkwood stress tensor for a configuration and bead forces."""
    Rc = R - R.mean(axis=0)
    return Rc.T @ F


def shear_viscosity(R: np.ndarray, F: np.ndarray, rate: float) -> float:
    """Polymer shear viscosity contribution  -tau_xy / rate."""
    return -kramers_stress(R, F)[0, 1] / rate


def normal_stress_differences(R: np.ndarray, F: np.ndarray):
    """(N1, N2) = (tau_xx - tau_yy, tau_yy - tau_zz)."""
    s = kramers_stress(R, F)
    return s[0, 0] - s[1, 1], s[1, 1] - s[2, 2]
