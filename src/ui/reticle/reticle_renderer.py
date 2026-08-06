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
import PIL.ImageColor

from ..legend_text import load_legend_font, blit_pil_text
from .reticle import render_body_centric_transform


# Upstream renders at 768 and returns a 512x512 RGB image.
_UPSTREAM_SIZE = 512


def _pil_color_bgr(name: str) -> tuple:
    """Resolve a PIL colour name to a BGR tuple.

    Read from PIL rather than hard-coded so a legend swatch can never drift
    from the line it labels: these are the exact same colour names ``reticle.py``
    passes to ``ImageDraw``.
    """
    r, g, b = PIL.ImageColor.getrgb(name)
    return (b, g, r)


# The two layers upstream's ``render_body_centric_transform`` draws, in legend
# order. Colour names must match the literals in ``reticle.py``.
_LEGEND_ROWS = (
    ("Current", "cornflowerblue"),
    ("Target", "darkgrey"),
)


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
        frame_bgr = self._fit(frame_bgr)
        self._draw_legend(frame_bgr)
        return frame_bgr

    @staticmethod
    def _draw_legend(frame_bgr: np.ndarray) -> None:
        """Draw the current/target colour key top-left (in-place).

        Deliberately mirrors ``GuidanceWidgetRenderer3D._draw_guidance_legend``
        — same font stack, same proportional geometry, same "Current" / "Target"
        wording, transparent background — so the two guidance panels read as one
        display rather than two applications. Swatches are solid (not dashed
        like the 3D widget's target row) because both reticle layers are drawn
        as solid lines; a dashed swatch would key a line style that isn't there.
        """
        H, W = frame_bgr.shape[:2]

        # Geometry, all proportional to frame size (matches the 3D widget).
        pad = max(6, int(round(H * 0.022)))
        margin = max(8, int(round(H * 0.03)))
        row_h = max(12, int(round(H * 0.06)))
        swatch_w = max(24, int(round(W * 0.06)))
        swatch_h = max(4, int(round(H * 0.017)))
        label_gap = max(6, int(round(W * 0.018)))

        pil_font = load_legend_font(max(11, int(round(H * 0.045))))
        cv_scale = max(0.34, H * 0.0011)
        cv_thick = max(1, int(round(H * 0.0028)))
        text_col = (238, 238, 238)   # BGR

        def _text_size(text: str) -> tuple:
            if pil_font is not None:
                left, top, right, bottom = pil_font.getbbox(text)
                return right - left, bottom - top
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, cv_scale, cv_thick)
            return tw, th

        def _draw_label(text: str, x: int, cy: int) -> None:
            _tw, th = _text_size(text)
            if pil_font is not None:
                blit_pil_text(frame_bgr, text, x, cy - th // 2,
                              pil_font, text_col)
            else:
                cv2.putText(frame_bgr, text, (x, cy + th // 2),
                            cv2.FONT_HERSHEY_DUPLEX, cv_scale,
                            text_col, cv_thick, cv2.LINE_AA)

        label_w = max(_text_size(label)[0] for label, _ in _LEGEND_ROWS)
        swatch_x = margin + pad + label_w + label_gap
        row_top = margin + pad

        for r, (label, color_name) in enumerate(_LEGEND_ROWS):
            cy = row_top + r * row_h + row_h // 2
            _draw_label(label, margin + pad, cy)
            sy = cy - swatch_h // 2
            cv2.rectangle(frame_bgr, (swatch_x, sy),
                          (swatch_x + swatch_w, sy + swatch_h),
                          _pil_color_bgr(color_name), -1, cv2.LINE_AA)

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
