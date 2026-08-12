"""Vendored USGR SE(3) Error-State Kalman Filter for pose smoothing.

Upstream: https://github.com/trevor-chan/usgr (usgr/usgr/recording/filter)
Vendored verbatim except ``pose_filter.py`` imports rewritten to this package.
Pure numpy + scipy. Smooths the body-centric [R|T] pose on the SE(3) manifold.
"""

from .types import FilterState, PoseMeasurement
from .se3 import SE3, se3_exp, se3_log
from .process_model import ConstantVelocityModel
from .measurement_model import PoseMeasurementModel
from .eskf import SE3ErrorStateKF
from .utils import make_diagonal_R
from .pose_filter import PoseFilter

__all__ = [
    "FilterState", "PoseMeasurement", "SE3", "se3_exp", "se3_log",
    "ConstantVelocityModel", "PoseMeasurementModel", "SE3ErrorStateKF",
    "make_diagonal_R", "PoseFilter",
]
