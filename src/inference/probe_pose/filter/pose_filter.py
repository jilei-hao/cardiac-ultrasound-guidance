"""
Standalone SE(3) pose filter for real-time control loops.

Vendored from USGR ``usgr/recording/filter/pose_filter.py``; imports rewritten
to this vendored package (the only change).

Usage:
    from .pose_filter import PoseFilter

    filt = PoseFilter()

    # In your control loop:
    filt(timestamp_ns, pose_vector_12)  # new measurement -> filtered 12-vec
    filt(timestamp_ns)                  # extrapolate only -> 12-vec
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import (
    SE3ErrorStateKF,
    ConstantVelocityModel,
    PoseMeasurementModel,
    PoseMeasurement,
)
from .se3 import SE3
from .utils import make_diagonal_R


class PoseFilter:
    """
    Thin wrapper around the SE(3) Error-State EKF for real-time use.

    Call with a measurement to update + predict, or without to extrapolate.
    Timestamps are in nanoseconds (converted internally).

    Args:
        sigma_pose:      Process noise on pose (rad, m). Higher = trust measurements more.
        sigma_vel:       Process noise on velocity (rad/s, m/s). Higher = allow faster changes.
        sigma_meas_rot:  Measurement noise on rotation (rad).
        sigma_meas_trans: Measurement noise on translation (m).
        timestamp_unit:  Scale factor to convert input timestamps to seconds.
                         Default 1e-9 for nanoseconds. Use 1.0 if already in seconds.
    """

    def __init__(
        self,
        sigma_pose: float = 0.05,
        sigma_vel: float = 0.25,
        sigma_meas_rot: float = 0.5,
        sigma_meas_trans: float = 0.5,
        timestamp_unit: float = 1e-9,
    ):
        process = ConstantVelocityModel(
            sigma_pose=sigma_pose,
            sigma_vel=sigma_vel,
            use_full_adjoint=True,
        )
        measurement = PoseMeasurementModel(
            default_R=make_diagonal_R(sigma_meas_rot, sigma_meas_trans),
        )
        self._ekf = SE3ErrorStateKF(
            process_model=process,
            measurement_model=measurement,
        )
        self._ts_scale = timestamp_unit

    def __call__(
        self,
        timestamp: float | int,
        pose_vector: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Filter a pose measurement or extrapolate to the given time.

        Args:
            timestamp: Current time (in units matching timestamp_unit, default ns).
            pose_vector: 12-vector pose in body-centric coordinates. (R_flat, t)

        Returns:
            12-vector pose in body-centric coordinates. (R_flat, t)
        """
        ts = timestamp * self._ts_scale

        has_measurement = pose_vector is not None

        if has_measurement:
            R = pose_vector[:9].reshape(3, 3)
            t = pose_vector[9:]
            T = SE3(R, t)
            meas = PoseMeasurement(T=T, timestamp=ts)
            state = self._ekf.process_measurement(meas)
            R_filtered = state.T.rotation()
            t_filtered = state.T.translation()
            return np.concatenate([R_filtered.flatten(), t_filtered])
        else:
            if not self._ekf.initialized:
                raise RuntimeError(
                    "No measurement received yet — cannot extrapolate."
                )
            T = self._ekf.query(ts)
            return np.concatenate([T.rotation().flatten(), T.translation()])
