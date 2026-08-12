from __future__ import annotations

import numpy as np
from .se3 import SE3, se3_exp, se3_log
from .types import FilterState, PoseMeasurement


class PoseMeasurementModel:
    """
    Direct SE(3) pose measurement in body-centric coordinates.

    Innovation: δz = Log(T_nom⁻¹ · T_meas)   (right-error convention)
    Jacobian:   H = [I₆, 0₆ₓ₆]               (small-angle approximation)
    """

    def __init__(self, default_R: np.ndarray | None = None):
        """
        Args:
            default_R: Default 6x6 measurement noise covariance.
                       Diagonal ordering: [σ_ω², σ_ω², σ_ω², σ_v², σ_v², σ_v²]
        """
        if default_R is None:
            default_R = np.diag([0.01] * 3 + [0.005] * 3)
        self.default_R = default_R

    def update(self, state: FilterState, meas: PoseMeasurement) -> FilterState:
        """
        Kalman update with a pose measurement.

        Steps:
        1. Compute innovation in body frame (right error).
        2. Standard Kalman gain computation in ℝ¹².
        3. Inject correction into nominal state.

        Returns a new FilterState.
        """
        # --- Innovation (body-frame, right convention) ---
        T_err = state.T.inverse() * meas.T
        dz = se3_log(T_err)  # ℝ⁶

        # --- Measurement Jacobian ---
        H = np.zeros((6, 12))
        H[:6, :6] = np.eye(6)

        # --- Measurement noise ---
        R = meas.R if meas.R is not None else self.default_R

        # --- Standard Kalman update in ℝ¹² ---
        S = H @ state.P @ H.T + R
        # K = P Hᵀ S⁻¹  (use solve for numerical stability)
        K = state.P @ H.T @ np.linalg.inv(S)
        dx = K @ dz

        # Joseph form for numerical stability
        I_KH = np.eye(12) - K @ H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        # --- Injection: fold correction into nominal state ---
        d_xi = dx[:6]
        d_twist = dx[6:]

        T_new = state.T * se3_exp(d_xi)  # Right multiplication
        twist_new = state.twist + d_twist

        return FilterState(
            T=T_new,
            twist=twist_new,
            P=P_new,
            timestamp=meas.timestamp,
        )
