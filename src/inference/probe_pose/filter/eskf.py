from __future__ import annotations

import numpy as np
from .se3 import SE3, se3_exp, se3_log
from .types import FilterState, PoseMeasurement
from .process_model import ConstantVelocityModel
from .measurement_model import PoseMeasurementModel


class SE3ErrorStateKF:
    """
    Error-State Extended Kalman Filter on SE(3).

    Right-error convention: T_true = T_nom · Exp(δξ)
    State: (T ∈ SE(3), body_twist ∈ ℝ⁶)
    Error state: δx ∈ ℝ¹², covariance P ∈ ℝ¹²ˣ¹²
    """

    def __init__(
        self,
        process_model: ConstantVelocityModel | None = None,
        measurement_model: PoseMeasurementModel | None = None,
        min_dt: float = 1e-6,
    ):
        self.process = process_model or ConstantVelocityModel()
        self.measurement = measurement_model or PoseMeasurementModel()
        self.min_dt = min_dt
        self.state: FilterState | None = None
        self._initialized = False
        self._prev_meas: PoseMeasurement | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, meas: PoseMeasurement,
                   P_pose: float = 0.1,
                   P_vel: float = 1.0) -> None:
        """Initialize filter from a first measurement."""
        self.state = FilterState(
            T=meas.T,
            twist=np.zeros(6),
            P=np.diag([P_pose] * 6 + [P_vel] * 6),
            timestamp=meas.timestamp,
        )
        self._prev_meas = meas
        self._initialized = True

    def process_measurement(self, meas: PoseMeasurement) -> FilterState:
        """
        Full predict-update cycle for a new pose measurement.

        1st call: initializes the filter.
        2nd call: estimates initial velocity via finite difference, then updates.
        Subsequent calls: predict → update.
        """
        if not self._initialized:
            self.initialize(meas)
            return self.state

        # Predict to measurement time
        dt = meas.timestamp - self.state.timestamp
        if dt > self.min_dt:
            self.state = self.process.predict(self.state, dt)

        # Bootstrap velocity from first two measurements
        if self._prev_meas is not None and np.allclose(self.state.twist, 0.0):
            self.state.twist = self._estimate_initial_velocity(
                self._prev_meas, meas
            )
            self._prev_meas = None

        # Update with measurement
        self.state = self.measurement.update(self.state, meas)
        return self.state

    def query(self, t: float) -> SE3:
        """
        Extrapolate the current pose estimate to time t.

        Uses constant-velocity forward prediction from last update.
        Does NOT modify filter state.
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized.")

        dt = t - self.state.timestamp
        if abs(dt) < self.min_dt:
            return self.state.T

        omega = self.state.twist * dt
        return self.state.T * se3_exp(omega)

    def query_with_covariance(self, t: float) -> tuple[SE3, np.ndarray]:
        """Extrapolate pose and pose-covariance to time t."""
        if not self._initialized:
            raise RuntimeError("Filter not initialized.")

        dt = t - self.state.timestamp
        if abs(dt) < self.min_dt:
            return self.state.T, self.state.P[:6, :6].copy()

        temp_state = self.process.predict(self.state, dt)
        return temp_state.T, temp_state.P[:6, :6].copy()

    def _estimate_initial_velocity(
        self, prev: PoseMeasurement, curr: PoseMeasurement
    ) -> np.ndarray:
        """
        Estimate body-frame twist from two consecutive poses via finite difference.

        velocity = Log(T_prev⁻¹ · T_curr) / dt
        """
        dt = curr.timestamp - prev.timestamp
        if dt < self.min_dt:
            return np.zeros(6)

        T_rel = prev.T.inverse() * curr.T
        return se3_log(T_rel) / dt
