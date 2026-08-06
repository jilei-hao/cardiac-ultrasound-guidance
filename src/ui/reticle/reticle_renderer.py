"""CUSG adapter around the vendored USGR 2D reticle.

Keeps ``reticle.py`` a verbatim upstream copy: everything CUSG-specific lives
here -- turning a :class:`ProbePosePrediction` into the 12-vector upstream
wants, RGB->BGR (the rest of the UI pipeline is BGR until the final display
conversion), output sizing, and a placeholder frame for the frames where no
usable pose exists.

Convention note (this is the easy thing to get wrong): the reticle consumes the
USGR model's **native forward** transform -- ``[R_rel.flatten(), T_rel]``
exactly as ``ProbePosePredictor`` returns it, matching upstream
``realtime_inference.py``. Do **not** transpose. The 3D guidance widget looks
transposed at its call site only because that renderer internally composes the
legacy *inverse* convention; the reticle does not.
"""

from typing import Optional

import cv2
import numpy as np

from .reticle import render_body_centric_transform


# Upstream renders at 768 and returns a 512x512 RGB image.
_UPSTREAM_SIZE = 512


class ReticleRenderer:
    """Render the probe-pose correction as USGR's flat 2D reticle (BGR)."""

    def __init__(self, output_size: int = _UPSTREAM_SIZE):
        self.output_size = int(output_size)
        self._placeholder: Optional[np.ndarray] = None

    def render(self, probe_pose) -> np.ndarray:
        """Return an ``output_size`` square BGR frame for ``probe_pose``.

        Args:
            probe_pose: a ``ProbePosePrediction`` (or ``None``). Anything
                missing/invalid yields the "no prediction" placeholder rather
                than a stale or meaningless reticle -- an invalid pose still
                carries a well-formed rotation matrix, so drawing it would show
                a confident-looking but arbitrary correction.
        """
        if probe_pose is None or not getattr(probe_pose, "valid", False):
            return self._get_placeholder()

        transform = np.concatenate(
            [np.asarray(probe_pose.R_rel, dtype=float).flatten(),
             np.asarray(probe_pose.T_rel, dtype=float).flatten()]
        )
        image = render_body_centric_transform(transform, None)
        if image is None:
            # Upstream returns None on a transform it can't decompose.
            return self._get_placeholder()

        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self._fit(frame_bgr)

    def _fit(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[0] == self.output_size and frame_bgr.shape[1] == self.output_size:
            return frame_bgr
        return cv2.resize(
            frame_bgr, (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA
        )

    def _get_placeholder(self) -> np.ndarray:
        """Black square with a centred notice; built once and reused."""
        if self._placeholder is None:
            size = self.output_size
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            text = "no probe-pose prediction"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(frame, text, ((size - tw) // 2, (size + th) // 2),
                        font, scale, (110, 110, 110), thickness, cv2.LINE_AA)
            self._placeholder = frame
        # Copy: callers draw HUD text in place.
        return self._placeholder.copy()
