"""Vendored USGR 2D reticle renderer.

Upstream: https://github.com/trevor-chan/usgr, branch ``guidance``
          (``usgr/usgr/recording/reticle.py``, file last touched in f4250057).
Vendored verbatim -- keep it that way so upstream fixes drop in cleanly. Any
CUSG-specific behaviour (colours, sizing, "no prediction" placeholder, BGR
conversion) belongs in ``reticle_renderer.py``, not here.

The renderer consumes the same 12-vector the USGR ``PoseTransformer`` emits --
``[R_rel (9, row-major), T_rel (3, metres)]``, body-centric, current -> target
standard view -- and draws the operator's correction as a flat instrument-style
reticle: guideline arcs rolled by the roll error, a dot for the two tilt axes,
the whole group translated by the in-plane (y, z) offset, plus a right-hand bar
for the out-of-plane (x) offset.
"""

import PIL.Image, PIL.ImageDraw
import numpy as np
from scipy.spatial.transform import Rotation


def parse_RT(x):
    R = x[0:9].reshape(3, 3)
    T = x[9:12]

    translation = - R.T @ T

    # Tilt: project body x-axis to avoid Euler singularities
    x_body = R @ np.array([1, 0, 0])
    tilt_y = x_body[1]  # left-right on reticle
    tilt_z = x_body[2]  # up-down on reticle

    # Roll from Euler decomposition
    roll = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)[2]

    return translation, tilt_y, tilt_z, roll


def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x


def _draw_reticle(transform, color, mode='prediction'):
    """Draw a single reticle layer for a body-centric transform.

    Args:
        transform: [12] body-centric transform [R_flat (9), T (3)]
        color: fill color for the reticle elements
        mode: 'target', 'ground_truth', or 'prediction'

    Returns:
        (reticle_image, y_offset, z_offset, x_translation)
    """
    translation, tilt_y, tilt_z, roll = parse_RT(transform)
    x, y, z = translation.tolist()

    width = 320
    height = 320
    reticle = PIL.Image.new(mode='RGBA', size=(width, height), color=(0, 0, 0, 0))
    drawer = PIL.ImageDraw.Draw(reticle)

    # guidelines
    bar_pad = 32
    arc_r = 64
    arc_span = 45
    # right arc: center shifted so rightmost point is flush with bar end
    rcx = width - bar_pad - arc_r
    rcy = height // 2
    drawer.arc(((rcx - arc_r, rcy - arc_r), (rcx + arc_r, rcy + arc_r)), start=-arc_span, end=arc_span, fill=color, width=5)
    # left arc: center shifted so leftmost point is flush with bar end
    lcx = bar_pad + arc_r
    lcy = height // 2
    drawer.arc(((lcx - arc_r, lcy - arc_r), (lcx + arc_r, lcy + arc_r)), start=180 - arc_span, end=180 + arc_span, fill=color, width=5)
    drawer.line((bar_pad, height // 2, width - bar_pad, height // 2), fill=color, width=3)
    drawer.polygon([(width // 2 - 8, height // 2 - 16), (width // 2 + 8, height // 2 - 16), (width // 2, height // 2 - 32)], outline=color, width=3)
    drawer.line((width // 2, height // 2, width // 2, height // 2 -18), fill=color, width=3)

    # roll (rotates guidelines but not the tilt dot)
    reticle = reticle.rotate(roll)

    # tilt dot (drawn after roll so it's decoupled)
    drawer = PIL.ImageDraw.Draw(reticle)
    if mode == 'target':
        tilt_bbox_size = 20.0
        dot_margin = 0
    elif mode == 'ground_truth':
        tilt_bbox_size = 1.0
        dot_margin = 0
    else:
        tilt_bbox_size = 1.0
        dot_margin = 48
    clamp_hi = width - max(1, dot_margin)
    dot_x = clamp((-tilt_y / tilt_bbox_size * .5 + .5) * width, dot_margin, clamp_hi)
    dot_y = clamp(( tilt_z / tilt_bbox_size * .5 + .5) * height, dot_margin, clamp_hi)
    if mode == 'target':
        drawer.circle((dot_x, dot_y), fill=color, radius=6, width=3)
    elif mode == 'ground_truth':
        drawer.circle((dot_x, dot_y), outline=color, radius=10, width=3)
    else:
        drawer.circle((dot_x, dot_y), outline=color, radius=10, width=3)

    # yz position
    yz_bbox_size = .2
    y_off = round(clamp(( y / yz_bbox_size) * width,  -(3 * width)  // 4, (3 * width)  // 4))
    z_off = round(clamp(( z / yz_bbox_size) * height, -(3 * height) // 4, (3 * height) // 4))

    return reticle, y_off, z_off, x


def _paste_reticle(image, reticle, y_off, z_off):
    image.paste(reticle, (image.height // 2 - reticle.width // 2 + y_off,
                          image.height // 2 - reticle.height // 2 + z_off), reticle)


# x-bar layout constants (for 768x768 image)
_BAR_X = 720        # horizontal center of the bar strip
_BAR_TOP = 96       # top of bar region
_BAR_BOTTOM = 672   # bottom of bar region
_BAR_MID = 384      # target reference level (image center)
_BAR_HALF = _BAR_MID - _BAR_TOP  # 288px half-range
_BAR_HW = 16        # half-width of the target marker line
_X_BBOX_SIZE = .1   # x displacement that maps to full bar range


def _draw_x_bar(drawer, x_val, color, mode):
    """Draw x-displacement bar elements on the base image.

    Target mode: short horizontal marker at the reference level.
    Ground truth / prediction: vertical bar from bottom to current x level.
    """
    if mode == 'target':
        # target reference marker
        drawer.line((_BAR_X - _BAR_HW, _BAR_MID, _BAR_X + _BAR_HW, _BAR_MID), fill=color, width=5)
    else:
        # bar from bottom up to current x level
        bar_level = round(clamp(_BAR_MID - (x_val / _X_BBOX_SIZE) * _BAR_HALF, _BAR_TOP, _BAR_BOTTOM))
        drawer.line((_BAR_X, _BAR_BOTTOM, _BAR_X, bar_level), fill=color, width=6)


def render_body_centric_transform(predicted, ground_truth):
    """Render a body-centric transform visualization with optional ground truth overlay.

    Args:
        predicted: [12] body-centric transform [R_flat (9), T (3)]
        ground_truth: Optional [12] ground truth transform for overlay

    Returns:
        PIL.Image with reticle visualization, or None if transforms are invalid.
    """
    image = PIL.Image.new(mode='RGB', size=(768, 768), color='black')
    drawer = PIL.ImageDraw.Draw(image)
    # drawer.line((0, 384, 768, 384), fill='darkgrey', width=1)
    # drawer.line((384, 0, 384, 768), fill='darkgrey', width=1)

    # target guidelines (identity transform)
    identity = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
    try:
        reticle, y_off, z_off, x_val = _draw_reticle(identity, 'darkgrey', mode='target')
        _paste_reticle(image, reticle, y_off, z_off)
        _draw_x_bar(drawer, x_val, 'darkgrey', mode='target')
    except ValueError:
        return None

    # ground truth overlay
    if ground_truth is not None:
        try:
            reticle, y_off, z_off, x_val = _draw_reticle(ground_truth, 'lightblue', mode='ground_truth')
            _paste_reticle(image, reticle, y_off, z_off)
            _draw_x_bar(drawer, x_val, 'lightblue', mode='ground_truth')
        except ValueError:
            return None

    # predicted
    try:
        reticle, y_off, z_off, x_val = _draw_reticle(predicted, 'cornflowerblue', mode='prediction')
        _paste_reticle(image, reticle, y_off, z_off)
        _draw_x_bar(drawer, x_val, 'cornflowerblue', mode='prediction')
    except ValueError:
        return None

    # border
    # PIL.ImageDraw.Draw(image).rectangle((0, 0, 767, 767), fill=None, outline='grey', width=16)
    
    # resize to 512x512 before display
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    return image


if __name__ == '__main__':
    a = np.concatenate([np.eye(3).flatten(), np.zeros(3)])
    a[9:12] = np.array([.10, .0, .0])
    a[0:9] = Rotation.from_euler('ZYX', [0, 0, 0], degrees=True).as_matrix().flatten()

    img = render_body_centric_transform(a, None)
    img.save('out.png')
