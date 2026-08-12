"""Contour sampling and uncertainty helpers used by live 4-chamber inference."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def biggest_contour(
    binary01: np.ndarray, min_area: int = 50
) -> Optional[np.ndarray]:
    """Return the largest external contour whose area meets ``min_area``."""
    bin255 = (binary01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= float(min_area)
    ]
    return max(contours, key=cv2.contourArea) if contours else None


def sample_equally_spaced_points(
    contour: np.ndarray, num_points: int
) -> List[Tuple[int, int]]:
    """Sample integer points at equal distances around a closed contour."""
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 2:
        return []

    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(closed[1:] - closed[:-1], axis=1)
    total_length = float(segment_lengths.sum())
    if total_length < 1e-6:
        return []

    cumulative = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    targets = np.linspace(0.0, total_length, num_points, endpoint=False)
    sampled = []
    for target in targets:
        index = int(np.searchsorted(cumulative, target, side="right") - 1)
        index = min(max(index, 0), len(segment_lengths) - 1)
        segment_length = segment_lengths[index]
        if segment_length < 1e-6:
            point = closed[index]
        else:
            ratio = (target - cumulative[index]) / segment_length
            point = closed[index] + ratio * (closed[index + 1] - closed[index])
        sampled.append((int(round(point[0])), int(round(point[1]))))
    return sampled


def _auto_canny_thresholds(
    blurred: np.ndarray, sigma: float = 0.33
) -> Tuple[int, int]:
    median = float(np.median(blurred))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if lower == upper:
        lower = max(0, upper - 50)
    return lower, upper


def compute_uncertainty_maps(
    gray: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge, edge-distance, and normalized gradient maps."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lower, upper = _auto_canny_thresholds(blurred)
    edges = cv2.Canny(blurred, lower, upper)
    distance = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)

    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    reference = np.percentile(magnitude, 90) if magnitude.size > 0 else 1.0
    if reference < 1e-6:
        reference = 1.0
    normalized = np.clip(magnitude / reference, 0.0, 1.0)
    return edges, distance, normalized


def landmark_uncertainty_batch(
    xs: np.ndarray,
    ys: np.ndarray,
    dist_map: np.ndarray,
    gradmag_norm: np.ndarray,
    grad_radius: int = 4,
    d0: float = 5.0,
    w_dist: float = 0.5,
    w_grad: float = 0.5,
) -> np.ndarray:
    """Compute the live uncertainty score for all sampled landmarks."""
    height, width = dist_map.shape[:2]
    xs_i = np.clip(np.asarray(xs, dtype=np.int64), 0, width - 1)
    ys_i = np.clip(np.asarray(ys, dtype=np.int64), 0, height - 1)

    distance = dist_map[ys_i, xs_i].astype(np.float32, copy=False)
    distance_confidence = np.exp(-distance / max(1e-6, float(d0)))

    radius = int(max(1, grad_radius))
    dy_grid, dx_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    dy_offset, dx_offset = np.where(
        (dx_grid * dx_grid + dy_grid * dy_grid) <= radius * radius
    )
    dy_offset = dy_offset.astype(np.int64) - radius
    dx_offset = dx_offset.astype(np.int64) - radius

    ys_index = ys_i[:, None] + dy_offset[None, :]
    xs_index = xs_i[:, None] + dx_offset[None, :]
    valid = (
        (ys_index >= 0)
        & (ys_index < height)
        & (xs_index >= 0)
        & (xs_index < width)
    )
    values = gradmag_norm[
        np.clip(ys_index, 0, height - 1),
        np.clip(xs_index, 0, width - 1),
    ]
    valid_count = valid.sum(axis=1).astype(np.float32)
    local_gradient = np.where(
        valid_count > 0,
        np.where(valid, values, 0.0).sum(axis=1, dtype=np.float32)
        / np.maximum(valid_count, 1.0),
        np.float32(0.0),
    )

    confidence = np.clip(
        np.float32(w_dist) * distance_confidence
        + np.float32(w_grad) * local_gradient,
        0.0,
        1.0,
    )
    return np.clip(1.0 - confidence, 0.0, 1.0).astype(
        np.float32, copy=False
    )
