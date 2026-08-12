"""Small OpenCV helpers shared by the live cardiac renderers."""

import cv2
import numpy as np


def add_rgb_border_to_result(input_frame_bgr, rgb):
    """Draw the traffic-light border, converting its RGB value to OpenCV BGR."""
    bgr = (rgb[2], rgb[1], rgb[0])
    cv2.rectangle(
        input_frame_bgr,
        (0, 0),
        (input_frame_bgr.shape[1] - 1, input_frame_bgr.shape[0] - 1),
        bgr,
        40,
    )
    return input_frame_bgr


def overlay_color_legend(
    input_frame, legend_bgra, top=10, left=10, pct_width=0.2, margin=20
):
    """Overlay a BGRA color legend onto an OpenCV frame."""
    height, width = input_frame.shape[:2]
    legend_height, legend_width = legend_bgra.shape[:2]
    new_width = int(width * pct_width)
    new_height = int(new_width / (legend_width / legend_height))
    legend_bgra = cv2.resize(
        legend_bgra, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    y1 = top + margin
    x1 = left + margin
    y2 = min(height, y1 + new_height)
    x2 = min(width, x1 + new_width)
    legend_crop = legend_bgra[: y2 - y1, : x2 - x1]
    legend_bgr = legend_crop[:, :, :3]
    alpha = legend_crop[:, :, 3] / 255.0

    for channel in range(3):
        input_frame[y1:y2, x1:x2, channel] = (
            alpha * legend_bgr[:, :, channel]
            + (1 - alpha) * input_frame[y1:y2, x1:x2, channel]
        ).astype(np.uint8)
    return input_frame
