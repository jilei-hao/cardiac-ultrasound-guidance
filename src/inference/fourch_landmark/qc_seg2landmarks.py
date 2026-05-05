#!/usr/bin/env python3
"""
Moving-window QC on segmentation masks -> landmark extraction with overlay output + uncertainty.

Enhancements in this version:
- Robust frame/mask pairing when masks have suffixes like "_pred" (configurable via --strip-suffixes).
- Per-landmark UNCERTAINTY estimation from the frame intensities:
    * Uses a combination of (a) distance to nearest edge (from Canny) and
      (b) local gradient magnitude contrast (Sobel) in a small disk around the landmark.
    * Produces a confidence in [0,1]; uncertainty = 1 - confidence.
- JSON stores per-landmark dictionaries: {"x","y","uncertainty","edge_dist","grad_local","confidence"}.
- Overlay draws each landmark using the label's hue but with saturation/brightness mapped by uncertainty:
    * low uncertainty => more saturated/brighter
    * high uncertainty => darker/desaturated
- NEW: Contour polyline also colored by local uncertainty, varying along the contour
       (disable with --no-contour-uncertainty).

Assumptions:
- Frame and mask PNGs share matching basenames after stripping mask suffixes (e.g., "_pred").
- Mask PNGs are single-channel or will be converted to grayscale; label values are e.g. 50,100,150,200.
- Label 0 is usually background; by default we skip it (pass --include-zero to include it).
"""

import argparse
import glob
import json
import os
import csv
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        # Convert RGB/BGR mask to grayscale (assumes labels encoded uniformly across channels)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def ensure_same_size(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    hm, wm = mask.shape[:2]
    if (h, w) != (hm, wm):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def strip_suffix_once(name_root: str, suffixes: Sequence[str]) -> str:
    """If name_root endswith any suffix in suffixes, strip exactly one occurrence (first match)."""
    for sfx in suffixes:
        if sfx and name_root.endswith(sfx):
            return name_root[: -len(sfx)]
    return name_root


def map_mask_to_frame_path(mask_path: str,
                           frame_dir: str,
                           strip_suffixes: Sequence[str],
                           frame_ext: str = ".png") -> Optional[str]:
    """
    Given a mask file path, return the expected frame path by removing a known mask suffix
    (e.g., '_pred') from the basename and appending frame_ext. If not found, try a glob fallback.
    """
    base = os.path.basename(mask_path)
    root, _ = os.path.splitext(base)
    frame_root = strip_suffix_once(root, strip_suffixes)
    # Preferred exact path
    fpath = os.path.join(frame_dir, frame_root + frame_ext)
    if os.path.exists(fpath):
        return fpath
    # Fallback: any extension
    candidates = sorted(glob.glob(os.path.join(frame_dir, frame_root + ".*")))
    if candidates:
        return candidates[0]
    return None


def intersect_labels(prev_mask: Optional[np.ndarray],
                     curr_mask: np.ndarray,
                     next_mask: Optional[np.ndarray],
                     label: int,
                     strict: bool = True) -> Optional[np.ndarray]:
    """
    Return a binary mask (0/1) for the intersection of the given label across frames.
    If `strict` is True, both neighbors must exist; otherwise we intersect with whichever is available.
    """
    if strict and (prev_mask is None or next_mask is None):
        return None
    inter = (curr_mask == label)
    if prev_mask is not None:
        inter = inter & (prev_mask == label)
    if next_mask is not None:
        inter = inter & (next_mask == label)
    return inter.astype(np.uint8)


def biggest_contour(binary01: np.ndarray, min_area: int = 50) -> Optional[np.ndarray]:
    """
    Given a binary (0/1) mask, return the largest external contour with area >= min_area.
    """
    bin255 = (binary01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # Filter very small speckles
    contours = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    if not contours:
        return None
    # Choose by area (robust for irregular shapes)
    return max(contours, key=cv2.contourArea)


def sample_equally_spaced_points(contour: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
    """
    Sample `num_points` equally spaced points along a closed contour.
    Returns list of (x, y) integer pixel coordinates.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return []
    # Close the loop
    pts_closed = np.vstack([pts, pts[0]])
    seg_vecs = pts_closed[1:] - pts_closed[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len < 1e-6:
        return []
    cum = np.insert(np.cumsum(seg_lens), 0, 0.0)
    # Equally spaced, start at t=0
    targets = np.linspace(0.0, total_len, num_points, endpoint=False)
    sampled: List[Tuple[int, int]] = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side='right') - 1)
        idx = min(max(idx, 0), len(seg_lens) - 1)
        seg_len = seg_lens[idx]
        if seg_len < 1e-6:
            p = pts_closed[idx]
        else:
            ratio = (t - cum[idx]) / seg_len
            p = pts_closed[idx] + ratio * (pts_closed[idx + 1] - pts_closed[idx])
        sampled.append((int(round(p[0])), int(round(p[1]))))
    return sampled


def color_for_label(label: int) -> Tuple[int, int, int]:
    # BGR base colors
    cmap = {
        0: (180, 180, 180),   # grey
        50: (0, 0, 255),      # red
        100: (0, 255, 0),     # green
        150: (255, 0, 0),     # blue
        200: (0, 255, 255),   # yellow
    }
    return cmap.get(label, (255, 255, 255))


def bgr_to_hsv_tuple(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    color = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def hsv_to_bgr_tuple(hsv: Tuple[int, int, int]) -> Tuple[int, int, int]:
    color = np.uint8([[list(hsv)]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def adjust_color_by_uncertainty(base_bgr: Tuple[int, int, int],
                                uncertainty: float,
                                s_min: int = 40, s_max: int = 255,
                                v_min: int = 60, v_max: int = 255) -> Tuple[int, int, int]:
    """
    Map uncertainty [0,1] to saturation/value. Lower uncertainty -> more saturated/brighter.
    """
    u = float(np.clip(uncertainty, 0.0, 1.0))
    h, s0, v0 = bgr_to_hsv_tuple(base_bgr)
    s = int(round(s_min + (1.0 - u) * (s_max - s_min)))
    v = int(round(v_min + (1.0 - u) * (v_max - v_min)))
    return hsv_to_bgr_tuple((h, s, v))


def auto_canny_thresholds(gray_blur: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """Auto Canny thresholds using the median heuristic."""
    v = float(np.median(gray_blur))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower == upper:  # degenerate case
        lower = max(0, upper - 50)
    return lower, upper


def disk_kernel(radius: int) -> np.ndarray:
    """Create a circular (disk) boolean kernel of given radius centered at (r,r)."""
    r = int(max(1, radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    return mask.astype(np.uint8)


def compute_uncertainty_maps(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute edge map (Canny), distance transform to nearest edge, and gradient magnitude (Sobel).
    Returns: edges (uint8 0/255), dist_map (float32), gradmag (float32 normalized 0..1 by robust max).
    """
    # Slight blur to reduce noise before edge/grad
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = auto_canny_thresholds(blur)
    edges = cv2.Canny(blur, low, high)

    # Distance to nearest edge
    inv = cv2.bitwise_not(edges)  # edges=255 -> 0; non-edge=0 -> 255
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    # Gradient magnitude
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradmag = cv2.magnitude(gx, gy)

    # Robust normalization for gradmag
    g_ref = np.percentile(gradmag, 90) if gradmag.size > 0 else 1.0
    if g_ref < 1e-6:
        g_ref = 1.0
    gradmag_norm = np.clip(gradmag / g_ref, 0.0, 1.0)

    return edges, dist_map, gradmag_norm


def landmark_uncertainty(x: int, y: int,
                         dist_map: np.ndarray,
                         gradmag_norm: np.ndarray,
                         grad_radius: int = 4,
                         d0: float = 5.0,
                         w_dist: float = 0.5,
                         w_grad: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Compute per-landmark uncertainty based on:
      - Distance to nearest edge (smaller is better -> higher confidence via exp(-d/d0))
      - Local mean gradient magnitude within a disk of radius grad_radius (0..1)
    Returns tuple: (uncertainty, edge_dist, grad_local, confidence)
    """
    h, w = dist_map.shape[:2]
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))

    d = float(dist_map[yi, xi])
    # Convert distance to a [0,1] closeness score: 1 near edges, 0 far
    c_dist = float(np.exp(-d / max(1e-6, d0)))

    # Local gradient mean
    r = int(max(1, grad_radius))
    y0 = max(0, yi - r); y1 = min(h, yi + r + 1)
    x0 = max(0, xi - r); x1 = min(w, xi + r + 1)
    patch = gradmag_norm[y0:y1, x0:x1]
    mask = disk_kernel(r)[(y0 - (yi - r)):(y1 - (yi - r)), (x0 - (xi - r)):(x1 - (xi - r))]
    if patch.size == 0 or mask.size == 0:
        g_local = 0.0
    else:
        vals = patch[mask.astype(bool)]
        g_local = float(vals.mean()) if vals.size > 0 else 0.0

    # Combine into confidence and uncertainty
    confidence = float(np.clip(w_dist * c_dist + w_grad * g_local, 0.0, 1.0))
    uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))

    return uncertainty, d, g_local, confidence


def landmark_uncertainty_batch(xs: np.ndarray,
                               ys: np.ndarray,
                               dist_map: np.ndarray,
                               gradmag_norm: np.ndarray,
                               grad_radius: int = 4,
                               d0: float = 5.0,
                               w_dist: float = 0.5,
                               w_grad: float = 0.5) -> np.ndarray:
    """Vectorized version of ``landmark_uncertainty`` for N points at once.

    Equivalent semantics to the scalar version: per-point distance-based
    confidence ``exp(-d/d0)`` plus the mean of ``gradmag_norm`` over the
    in-bounds intersection of a disk of radius ``grad_radius`` centered at
    each point. ~60× fewer Python/numpy dispatches than calling the scalar
    function in a loop.

    Returns: uncertainty array, shape (N,), dtype float32, in [0, 1].
    """
    h, w = dist_map.shape[:2]
    xs_i = np.clip(np.asarray(xs, dtype=np.int64), 0, w - 1)
    ys_i = np.clip(np.asarray(ys, dtype=np.int64), 0, h - 1)

    # Distance-based closeness, one gather over the distance map.
    d = dist_map[ys_i, xs_i].astype(np.float32, copy=False)
    c_dist = np.exp(-d / max(1e-6, float(d0)))

    # Precompute the disk's pixel offsets once. ``disk`` is the same boolean
    # kernel as ``disk_kernel(r)``; we only keep the in-disk offsets to skip
    # the square corners entirely.
    r = int(max(1, grad_radius))
    dy_grid, dx_grid = np.ogrid[-r:r + 1, -r:r + 1]
    in_disk = (dx_grid * dx_grid + dy_grid * dy_grid) <= r * r
    dy_off, dx_off = np.where(in_disk)
    dy_off = dy_off.astype(np.int64) - r
    dx_off = dx_off.astype(np.int64) - r

    # (N, K) absolute pixel coords for each landmark's disk.
    ys_idx = ys_i[:, None] + dy_off[None, :]
    xs_idx = xs_i[:, None] + dx_off[None, :]
    valid = (ys_idx >= 0) & (ys_idx < h) & (xs_idx >= 0) & (xs_idx < w)

    # Clip out-of-bounds indices so the gather is safe; the validity mask
    # zeroes them out before the mean.
    ys_clip = np.clip(ys_idx, 0, h - 1)
    xs_clip = np.clip(xs_idx, 0, w - 1)
    vals = gradmag_norm[ys_clip, xs_clip]
    vals_masked = np.where(valid, vals, 0.0)
    valid_count = valid.sum(axis=1).astype(np.float32)
    g_local = np.where(
        valid_count > 0,
        vals_masked.sum(axis=1, dtype=np.float32) / np.maximum(valid_count, 1.0),
        np.float32(0.0),
    )

    confidence = np.clip(
        np.float32(w_dist) * c_dist + np.float32(w_grad) * g_local,
        0.0, 1.0,
    )
    return np.clip(1.0 - confidence, 0.0, 1.0).astype(np.float32, copy=False)


def draw_contour_with_uncertainty(overlay: np.ndarray,
                                  contour: np.ndarray,
                                  base_bgr: Tuple[int, int, int],
                                  dist_map: np.ndarray,
                                  gradmag_norm: np.ndarray,
                                  grad_radius: int,
                                  d0: float,
                                  w_dist: float,
                                  w_grad: float,
                                  thickness: int = 2,
                                  step: int = 1) -> None:
    """
    Draw a closed polyline where each short segment's color is adjusted by local uncertainty.
    step >= 1 can be increased to speed drawing at the cost of smoothness.
    """
    pts = contour.reshape(-1, 2).astype(int)
    n = len(pts)
    if n < 2:
        return

    step = max(1, int(step))

    # Precompute uncertainty per vertex
    u = np.zeros(n, dtype=np.float32)
    for i in range(0, n, step):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        u_i, _, _, _ = landmark_uncertainty(x, y, dist_map, gradmag_norm,
                                            grad_radius=grad_radius, d0=d0, w_dist=w_dist, w_grad=w_grad)
        u[i] = u_i
    # Fill missing u entries (if step>1) by simple forward fill + wrap
    if step > 1:
        last_val = u[0]
        for i in range(1, n):
            if u[i] == 0 and u[i-1] != 0:
                last_val = u[i-1]
            if u[i] == 0:
                u[i] = last_val
        if u[0] == 0:
            u[0] = u[-1]

    # Draw segment-wise with color based on mean uncertainty of the endpoints
    for i in range(n):
        j = (i + 1) % n
        u_seg = float(0.5 * (u[i] + u[j]))
        col = adjust_color_by_uncertainty(base_bgr, u_seg)
        cv2.line(overlay, (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])),
                 col, thickness, lineType=cv2.LINE_AA)


def process_sequence(frame_dir: str,
                     mask_dir: str,
                     dest_dir: str,
                     labels: Sequence[int],
                     n_landmarks: int = 12,
                     strict: bool = True,
                     draw_contours: bool = True,
                     alpha: float = 0.30,
                     point_radius: int = 3,
                     thickness: int = 2,
                     min_area: int = 50,
                     save_coords: bool = False,
                     suffix: str = "_landmarks",
                     strip_suffixes: Sequence[str] = ("_pred", "_mask", "_seg", "_label"),
                     frame_ext: str = ".png",
                     grad_radius: int = 4,
                     d0: float = 5.0,
                     w_dist: float = 0.5,
                     w_grad: float = 0.5,
                     contour_uncertainty: bool = True,
                     contour_step: int = 1) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        raise RuntimeError(f"No mask PNGs found in {mask_dir}")
    
    # Initialize CSV data collection if saving coordinates
    csv_data = []
    
    for i, mpath in enumerate(mask_paths):
        base = os.path.basename(mpath)

        fpath = map_mask_to_frame_path(mpath, frame_dir, strip_suffixes, frame_ext=frame_ext)
        if fpath is None or not os.path.exists(fpath):
            print(f"[WARN] No corresponding frame for mask {base}; expected in {frame_dir} (suffixes={list(strip_suffixes)}). Skipping.")
            continue

        prev_mask = read_mask(mask_paths[i - 1]) if i - 1 >= 0 else None
        curr_mask = read_mask(mpath)
        next_mask = read_mask(mask_paths[i + 1]) if i + 1 < len(mask_paths) else None

        frame = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Could not read frame {fpath}; skipping.")
            continue

        # Ensure masks match frame size
        if prev_mask is not None:
            prev_mask = ensure_same_size(frame, prev_mask)
        curr_mask = ensure_same_size(frame, curr_mask)
        if next_mask is not None:
            next_mask = ensure_same_size(frame, next_mask)

        if strict and (prev_mask is None or next_mask is None):
            print(f"[INFO] Skipping boundary frame {base} due to strict QC (needs -1 and +1).")
            continue

        # Precompute uncertainty maps from the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges, dist_map, gradmag_norm = compute_uncertainty_maps(gray)

        # Transparent overlay
        overlay = np.zeros_like(frame)
        coords_out: Dict[str, List[Dict[str, float]]] = {}
        
        # Get frame basename for CSV
        frame_basename = os.path.basename(fpath)

        for label in labels:
            inter = intersect_labels(prev_mask, curr_mask, next_mask, label, strict=strict)
            if inter is None or inter.sum() == 0:
                continue

            contour = biggest_contour(inter, min_area=min_area)
            if contour is None:
                continue

            # Draw contour (either with variable uncertainty color or constant label color)
            if draw_contours:
                base_col = color_for_label(label)
                if contour_uncertainty:
                    draw_contour_with_uncertainty(overlay, contour, base_col,
                                                  dist_map, gradmag_norm,
                                                  grad_radius, d0, w_dist, w_grad,
                                                  thickness=thickness, step=contour_step)
                else:
                    cv2.drawContours(overlay, [contour], -1, base_col, thickness=thickness)

            pts = sample_equally_spaced_points(contour, n_landmarks)
            if not pts:
                continue

            label_points: List[Dict[str, float]] = []
            for (x, y) in pts:
                # Compute uncertainty metrics
                u, d, g_local, conf = landmark_uncertainty(
                    x, y, dist_map, gradmag_norm,
                    grad_radius=grad_radius, d0=d0, w_dist=w_dist, w_grad=w_grad
                )
                # Draw point with uncertainty-adjusted color
                base = color_for_label(label)
                adj = adjust_color_by_uncertainty(base, u)
                cv2.circle(overlay, (x, y), point_radius, adj, -1, lineType=cv2.LINE_AA)

                label_points.append({
                    "x": int(x),
                    "y": int(y),
                    "uncertainty": float(u),
                    # "edge_dist": float(d),
                    # "grad_local": float(g_local),
                    # "confidence": float(conf),
                })
                
                # Add to CSV data if saving coordinates
                if save_coords:
                    csv_data.append({
                        "image_frame": frame_basename,
                        "label": label,
                        "x": int(x),
                        "y": int(y),
                        "uncertainty": float(u),
                        # "edge_dist": float(d),
                        # "grad_local": float(g_local),
                        # "confidence": float(conf),

                    })

            if label_points:
                coords_out[str(label)] = label_points

        # Blend overlay with original frame
        blended = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)

        # Generate output base name from the *frame* basename (normalized without suffixes)
        out_base = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(dest_dir, out_base + suffix + ".png")
        cv2.imwrite(out_path, blended)

        print(f"[OK] Wrote {out_path}")
    
    # Write CSV file if saving coordinates
    if save_coords and csv_data:
        csv_path = os.path.join(dest_dir, "landmarks.csv")
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["image_frame", "label", "x", "y", "uncertainty"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"[OK] Wrote CSV with {len(csv_data)} landmarks to {csv_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Moving-window QC of segmentation and conversion to landmarks with overlay output + uncertainty."
    )
    ap.add_argument("--frame-dir", required=True, help="Directory of frame PNGs.")
    ap.add_argument("--mask-dir", required=True, help="Directory of mask PNGs (label values like 50,100,150,200).")
    ap.add_argument("--dest-dir", required=True, help="Output directory for overlays (and optional JSON).")
    ap.add_argument("--labels", nargs="+", type=int, default=[50, 100, 150, 200],
                    help="Label values to process. Default: 50 100 150 200")
    ap.add_argument("--include-zero", action="store_true",
                    help="Include label 0 (often background) in addition to provided labels.")
    ap.add_argument("--landmarks", type=int, default=12, help="Number of landmarks per label. Default: 12")
    ap.add_argument("--strict", action="store_true",
                    help="Require both neighbors (-1 and +1) for the intersection QC. "
                         "If not set, intersects with whichever neighbor(s) exist(s).")
    ap.add_argument("--no-contours", dest="draw_contours", action="store_false",
                    help="Do not draw contour lines, only the points.")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="Overlay alpha (0..1) for blending landmarks with the original frame. Default: 0.30")
    ap.add_argument("--point-radius", type=int, default=3, help="Radius of landmark points in pixels. Default: 3")
    ap.add_argument("--thickness", type=int, default=2, help="Contour line thickness. Default: 2")
    ap.add_argument("--min-area", type=int, default=50, help="Minimum contour area to keep (pixels). Default: 50")
    ap.add_argument("--save-coords", action="store_true", help="Also write CSV file with landmark coordinates + uncertainty.")
    ap.add_argument("--suffix", default="_landmarks", help="Suffix appended to output filename. Default: _landmarks")
    ap.add_argument("--strip-suffixes", nargs="+", default=["_pred", "_mask", "_seg", "_label"],
                    help="Suffixes to strip from mask basenames before mapping to frames. Default: _pred _mask _seg _label")
    ap.add_argument("--frame-ext", default=".png",
                    help="Expected frame file extension when mapping from mask basenames. Default: .png")

    # Uncertainty params
    ap.add_argument("--grad-radius", type=int, default=4, help="Disk radius (px) for local gradient averaging. Default: 4")
    ap.add_argument("--d0", type=float, default=5.0, help="Characteristic distance (px) for edge proximity confidence. Default: 5.0")
    ap.add_argument("--w-dist", type=float, default=0.5, help="Weight for edge proximity in confidence. Default: 0.5")
    ap.add_argument("--w-grad", type=float, default=0.5, help="Weight for local gradient magnitude in confidence. Default: 0.5")

    # Contour coloring
    ap.set_defaults(contour_uncertainty=True)
    ap.add_argument("--no-contour-uncertainty", dest="contour_uncertainty", action="store_false",
                    help="Use constant label color for contours (disable uncertainty coloring).")
    ap.add_argument("--contour-step", type=int, default=1,
                    help="Stride for uncertainty-evaluated contour segments (>=1). Larger = faster but blockier. Default: 1")

    return ap.parse_args()


def main():
    args = parse_args()
    labels = list(args.labels)
    if args.include_zero and 0 not in labels:
        labels = [0] + labels

    process_sequence(
        frame_dir=args.frame_dir,
        mask_dir=args.mask_dir,
        dest_dir=args.dest_dir,
        labels=labels,
        n_landmarks=args.landmarks,
        strict=args.strict,
        draw_contours=args.draw_contours,
        alpha=args.alpha,
        point_radius=args.point_radius,
        thickness=args.thickness,
        min_area=args.min_area,
        save_coords=args.save_coords,
        suffix=args.suffix,
        strip_suffixes=args.strip_suffixes,
        frame_ext=args.frame_ext,
        grad_radius=args.grad_radius,
        d0=args.d0,
        w_dist=args.w_dist,
        w_grad=args.w_grad,
        contour_uncertainty=args.contour_uncertainty,
        contour_step=args.contour_step,
    )


if __name__ == "__main__":
    main()
