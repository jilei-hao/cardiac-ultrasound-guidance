"""Shared on-frame legend text rendering.

Extracted from ``guidance_widget/guidance_widget_renderer_3d.py`` so every
guidance panel that draws an on-frame key (the 3D widget, the 2D reticle) uses
the *same* font stack and blitting — a legend that changes typeface between two
panels of the same display reads as two different applications.

Labels render as real anti-aliased TrueType via PIL rather than the
technical-looking cv2 Hershey stroke font: the target users are clinicians, not
engineers.
"""

import os

import numpy as np


# Legend label font, tried in order. The primary is Quicksand, a rounded
# geometric sans chosen for a friendly, non-technical look (Bold reads best
# against the dark panel at small sizes). The fallbacks mirror the vigil-system
# guidance *display* font stack (Inter, "Segoe UI", Arial, sans-serif — see
# display/vigil_display_utils/Style.py & GuidanceWidget.py), so where Quicksand
# is unavailable the legend still matches the operator UI: Inter (the house
# font), else the Arial substitute (Liberation Sans), else the generic sans
# (DejaVu). If nothing resolves, callers fall back to the cv2 Hershey font so
# rendering never breaks. Override with the GUIDANCE_LEGEND_FONT env var
# (absolute font path).
LEGEND_FONT_CANDIDATES = (
    os.environ.get("GUIDANCE_LEGEND_FONT", ""),
    # Quicksand — friendly rounded primary.
    "/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf",
    "/usr/share/fonts/truetype/quicksand/Quicksand-Medium.ttf",
    "Quicksand-Bold.ttf",
    # Inter (vigil house font). Per-user install dirs first (no sudo needed),
    # then the system paths a `fonts-inter` apt install would use.
    os.path.expanduser("~/.local/share/fonts/inter/Inter-Regular.otf"),
    os.path.expanduser("~/.fonts/inter/Inter-Regular.otf"),
    "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "/usr/share/fonts/truetype/inter/InterVariable.ttf",
    "Inter-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # ~Arial
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "LiberationSans-Regular.ttf",
    "Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",               # macOS
    "C:\\Windows\\Fonts\\arial.ttf",                      # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # generic sans fallback
    "DejaVuSans.ttf",
)
# (font_path, size) -> PIL ImageFont | None. Module-level so it is shared and
# survives across renderer instances; loading a TrueType face is not free.
_FONT_CACHE: dict = {}


def load_legend_font(size_px: int):
    """Return a PIL ImageFont near ``size_px``, or None if none resolve."""
    size = max(8, int(size_px))
    for cand in LEGEND_FONT_CANDIDATES:
        if not cand:
            continue
        key = (cand, size)
        if key not in _FONT_CACHE:
            try:
                from PIL import ImageFont
                _FONT_CACHE[key] = ImageFont.truetype(cand, size)
            except Exception:
                _FONT_CACHE[key] = None
        if _FONT_CACHE[key] is not None:
            return _FONT_CACHE[key]
    return None


def alpha_blit_rgba(frame_bgra: np.ndarray, sprite_rgba: np.ndarray,
                    x: int, y: int) -> None:
    """Alpha-composite an RGBA sprite onto a BGRA (or BGR) frame at (x, y)."""
    fh, fw = frame_bgra.shape[:2]
    sh, sw = sprite_rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + sw), min(fh, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    sprite = sprite_rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sprite[..., 3:4].astype(np.float32) / 255.0
    src_bgr = sprite[..., 2::-1].astype(np.float32)   # RGB -> BGR
    roi = frame_bgra[y0:y1, x0:x1]
    roi[..., :3] = (
        src_bgr * a + roi[..., :3].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)
    if frame_bgra.shape[2] == 4:
        roi[..., 3] = np.maximum(roi[..., 3], sprite[..., 3])


def blit_pil_text(frame_bgra: np.ndarray, text: str, x: int, y: int,
                  font, color_bgr: tuple) -> None:
    """Draw ``text`` (TrueType, anti-aliased) with its visual top-left at
    (x, y) onto the BGRA (or BGR) frame."""
    from PIL import Image, ImageDraw
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    if tw <= 0 or th <= 0:
        return
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (-left, -top), text, font=font,
        fill=(color_bgr[2], color_bgr[1], color_bgr[0], 255),
    )
    alpha_blit_rgba(frame_bgra, np.asarray(img), int(x), int(y))
