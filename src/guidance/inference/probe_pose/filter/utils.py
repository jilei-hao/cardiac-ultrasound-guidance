import numpy as np
from .se3 import skew


def make_diagonal_R(sigma_rot: float, sigma_trans: float) -> np.ndarray:
    """Create a diagonal 6x6 measurement noise covariance."""
    return np.diag([sigma_rot**2] * 3 + [sigma_trans**2] * 3)


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """Hat operator: ℝ⁶ → se(3) (4x4 matrix)."""
    w, v = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = skew(w)
    M[:3, 3] = v
    return M


def ad_se3(xi: np.ndarray) -> np.ndarray:
    """Small adjoint (ad) operator: ℝ⁶ → ℝ⁶ˣ⁶."""
    w, v = xi[:3], xi[3:]
    W = skew(w)
    V = skew(v)
    ad = np.zeros((6, 6))
    ad[:3, :3] = W
    ad[3:, :3] = V
    ad[3:, 3:] = W
    return ad
