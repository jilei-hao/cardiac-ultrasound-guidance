from __future__ import annotations

import numpy as np
from .se3 import SE3, se3_exp
from .types import FilterState


class ConstantVelocityModel:
    """
    Constant body-frame velocity process model on SE(3).

    State: (T ∈ SE(3), twist ∈ ℝ⁶)
    Model: T_{k+1} = T_k · Exp(twist · dt)
           twist_{k+1} = twist_k
    """

    def __init__(
        self,
        sigma_pose: float = 0.01,
        sigma_vel: float = 0.1,
        use_full_adjoint: bool = False,
    ):
        """
        Args:
            sigma_pose: Process noise std dev for pose (rad and meters).
            sigma_vel: Process noise std dev for velocity (rad/s and m/s).
            use_full_adjoint: If True, compute full Ad(Exp(-Ω)) for F matrix.
                              If False, use I₆ approximation (valid when dt·twist is small).
        """
        self.sigma_pose = sigma_pose
        self.sigma_vel = sigma_vel
        self.use_full_adjoint = use_full_adjoint

    def predict(self, state: FilterState, dt: float) -> FilterState:
        """
        Propagate nominal state and covariance forward by dt.

        Returns a new FilterState (does not mutate input).
        """
        omega = state.twist * dt  # Motion increment in body frame

        # --- Nominal state propagation ---
        T_new = state.T * se3_exp(omega)
        twist_new = state.twist.copy()

        # --- Error-state transition matrix F (12x12) ---
        F = self._compute_F(omega, dt, state.T)

        # --- Process noise Q (12x12) ---
        Q = self._compute_Q(dt)

        # --- Covariance propagation ---
        P_new = F @ state.P @ F.T + Q

        return FilterState(
            T=T_new,
            twist=twist_new,
            P=P_new,
            timestamp=state.timestamp + dt,
        )

    def _compute_F(self, omega: np.ndarray, dt: float, T: SE3) -> np.ndarray:
        """Compute the 12x12 error-state transition matrix."""
        F = np.eye(12)

        if self.use_full_adjoint:
            Ad = se3_exp(-omega).adj()  # 6x6 adjoint of Exp(-omega)
            F[:6, :6] = Ad
        # else: F[:6, :6] stays as I₆

        # F[:6, 6:] = Jr(omega) * dt ≈ I₆ * dt under small-angle
        F[:6, 6:] = dt * np.eye(6)

        return F

    def _compute_Q(self, dt: float) -> np.ndarray:
        """Compute 12x12 process noise, scaled by dt."""
        Q = np.zeros((12, 12))
        Q[:6, :6] = (self.sigma_pose ** 2) * dt * np.eye(6)
        Q[6:, 6:] = (self.sigma_vel ** 2) * dt * np.eye(6)
        return Q
