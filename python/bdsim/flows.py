"""Velocity-gradient (kappa) tensors for common flows, as bdsim.Flow objects."""
import numpy as np
from ._bdsim import Flow


def equilibrium() -> Flow:
    """No flow (kappa = 0)."""
    return Flow()


def shear(rate: float) -> Flow:
    """Simple shear: v = (rate * y, 0, 0)."""
    K = np.zeros((3, 3))
    K[0, 1] = rate
    return Flow.constant(K)


def uniaxial_extension(rate: float) -> Flow:
    """Uniaxial extension: diag(rate, -rate/2, -rate/2)."""
    K = np.diag([rate, -rate / 2.0, -rate / 2.0])
    return Flow.constant(K)


def planar_extension(rate: float) -> Flow:
    """Planar extension: diag(rate, -rate, 0)."""
    K = np.diag([rate, -rate, 0.0])
    return Flow.constant(K)
