"""
Minimal SE(3) Lie group implementation for the error-state EKF.

Tangent vector convention: ξ = [ω₁, ω₂, ω₃, v₁, v₂, v₃]
  - ω ∈ ℝ³: angular (rotation) component
  - v ∈ ℝ³: linear (translation) component
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from ℝ³."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _V_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Compute the V matrix used in SE(3) Exp map.

    V = I + (1-cos θ)/θ² [ω]× + (θ - sin θ)/θ³ [ω]×²

    For small θ, V ≈ I + 0.5 [ω]×
    """
    theta = np.linalg.norm(omega)
    W = skew(omega)

    if theta < 1e-10:
        return np.eye(3) + 0.5 * W

    W2 = W @ W
    t2 = theta * theta
    t3 = t2 * theta
    return np.eye(3) + ((1 - np.cos(theta)) / t2) * W + ((theta - np.sin(theta)) / t3) * W2


def _V_matrix_inv(omega: np.ndarray) -> np.ndarray:
    """
    Inverse of the V matrix, used in SE(3) Log map.

    V⁻¹ = I - 0.5 [ω]× + (1/θ² - (1+cos θ)/(2θ sin θ)) [ω]×²
    """
    theta = np.linalg.norm(omega)
    W = skew(omega)

    if theta < 1e-10:
        return np.eye(3) - 0.5 * W

    W2 = W @ W
    t2 = theta * theta
    half_theta = theta / 2.0
    coeff = (1.0 / t2) - (np.cos(half_theta)) / (2.0 * theta * np.sin(half_theta))
    return np.eye(3) - 0.5 * W + coeff * W2


class SE3:
    """
    An element of SE(3): rigid body transformation.

    Internally stores a 3x3 rotation matrix R and a 3x1 translation vector t.
    The 4x4 homogeneous form is: [[R, t], [0, 1]]
    """

    def __init__(self, R: np.ndarray, t: np.ndarray):
        """
        Args:
            R: 3x3 rotation matrix (assumed orthogonal, not re-normalized).
            t: 3-vector translation.
        """
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).ravel()

    @classmethod
    def identity(cls) -> SE3:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> SE3:
        """Construct from a 4x4 homogeneous matrix."""
        return cls(T[:3, :3], T[:3, 3])

    def as_matrix(self) -> np.ndarray:
        """Return 4x4 homogeneous matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> SE3:
        """T⁻¹ = [Rᵀ, -Rᵀ t]"""
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t)

    def compose(self, other: SE3) -> SE3:
        """self · other (left multiplication)."""
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def __mul__(self, other: SE3) -> SE3:
        return self.compose(other)

    def adj(self) -> np.ndarray:
        """
        6x6 Adjoint matrix Ad(T).

        Ad(T) = [[R, 0], [[t]× R, R]]

        Maps tangent vectors from one frame to another:
        ξ_world = Ad(T) · ξ_body
        """
        tR = skew(self.t) @ self.R
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = self.R
        Ad[3:, :3] = tR
        Ad[3:, 3:] = self.R
        return Ad

    def rotation(self) -> np.ndarray:
        """Return 3x3 rotation matrix."""
        return self.R.copy()

    def translation(self) -> np.ndarray:
        """Return translation 3-vector."""
        return self.t.copy()

    def __repr__(self) -> str:
        r = Rotation.from_matrix(self.R)
        euler = r.as_euler("xyz", degrees=True)
        return (
            f"SE3(t=[{self.t[0]:.3f}, {self.t[1]:.3f}, {self.t[2]:.3f}], "
            f"euler_xyz=[{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°])"
        )


def se3_exp(xi: np.ndarray) -> SE3:
    """
    Exponential map: se(3) → SE(3).

    Input: ξ = [ω₁, ω₂, ω₃, v₁, v₂, v₃] ∈ ℝ⁶
    Output: SE3 element

    Uses Rodrigues for rotation and the V matrix for translation:
        R = exp([ω]×)
        t = V(ω) · v
    """
    xi = np.asarray(xi, dtype=np.float64).ravel()
    omega = xi[:3]
    v = xi[3:]

    # Rotation via scipy (handles edge cases well)
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        R = np.eye(3) + skew(omega)
    else:
        R = Rotation.from_rotvec(omega).as_matrix()

    # Translation
    V = _V_matrix(omega)
    t = V @ v

    return SE3(R, t)


def se3_log(T: SE3) -> np.ndarray:
    """
    Logarithmic map: SE(3) → se(3).

    Output: ξ = [ω₁, ω₂, ω₃, v₁, v₂, v₃] ∈ ℝ⁶

    Uses scipy for rotation log and V⁻¹ for translation:
        ω = log(R)  (rotation vector)
        v = V⁻¹(ω) · t
    """
    # Rotation log via scipy
    omega = Rotation.from_matrix(T.R).as_rotvec()

    # Translation
    V_inv = _V_matrix_inv(omega)
    v = V_inv @ T.t

    return np.concatenate([omega, v])
