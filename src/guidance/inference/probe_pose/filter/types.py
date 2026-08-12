from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .se3 import SE3


@dataclass
class FilterState:
    """Full filter state: nominal + covariance."""
    T: SE3                               # Nominal pose on SE(3)
    twist: np.ndarray                    # Body-frame twist velocity, shape (6,)
    P: np.ndarray                        # Error-state covariance, shape (12, 12)
    timestamp: float                     # Time of last update

    @classmethod
    def from_identity(cls, timestamp: float = 0.0,
                      P_pose: float = 0.1,
                      P_vel: float = 1.0) -> FilterState:
        """Initialize at identity with diagonal covariance."""
        P = np.diag([P_pose] * 6 + [P_vel] * 6)
        return cls(
            T=SE3.identity(),
            twist=np.zeros(6),
            P=P,
            timestamp=timestamp,
        )


@dataclass
class PoseMeasurement:
    """A single pose observation in body-centric coordinates."""
    T: SE3                               # Measured pose
    R: np.ndarray | None = None          # Measurement noise covariance (6x6), or None for default
    timestamp: float = 0.0
