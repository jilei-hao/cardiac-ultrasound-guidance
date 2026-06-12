#!/usr/bin/env python3
"""Per-frame USGR SE(3) inference on two datasets, for chirality comparison.

Runs the (vendored) USGR PoseTransformer forward exactly as
``usgr/recording/inference_simulation.py`` does -- ``denormalize(model(img, onehot))``
-> raw 12-vector ``[R_flat(9), T(3)]`` -- and records one row per frame.

Two datasets, fixed conditioning view (default A4C = invivo index 2):
  * usgr-invivo : Trevor's in-vivo recording (LV-on-left), mirror OFF.
  * cusg-green  : the 2025-nov green Clarius set (LV-on-right), mirror ON.

The mirror is applied as a horizontal flip of the grayscale frame before
preprocessing (identical to ``ProbePosePredictor(mirror_input=True)``).

Outputs result-comparison.csv (raw SE(3) per frame) incrementally.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import cv2
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from inference.probe_pose import ProbePosePredictor          # noqa: E402
from inference.probe_pose.probe_pose_predictor import (        # noqa: E402
    preprocess_image_for_model,
    _nearest_rotation,
)

WEIGHTS = "/Users/jileihao/data/vigil/__demo_weights"
_INVIVO_ROOT = "/Users/jileihao/data/vigil/usgr_invivo_2026-02-04--13-02-35"

# 3-way comparison. (name, frames_dir, mirror). Both usgr-* are invivo (LV-on-left,
# mirror OFF); cusg-green is the Nov Clarius set (LV-on-right, mirror ON).
DATASETS = [
    # pose-head A4C convergence segment (frames 3450-3750; pose head's own min |R_rel|/|T_rel|)
    ("usgr-a4c",   f"{_INVIVO_ROOT}_a4c_poseseg/images", False),
    # traffic-light-green segment (frames 1480-1780; CUSG view-quality classifier)
    ("usgr-green", f"{_INVIVO_ROOT}_greenseg/images",     False),
    # Nov demo green Clarius set (out-of-distribution, LV-on-right)
    ("cusg-green", "/Users/jileihao/data/vigil/2025-nov-demo_2246f_green", True),
]


def geodesic_deg(R: np.ndarray) -> float:
    Rp = _nearest_rotation(R)
    return float(np.degrees(np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1.0, 1.0))))


@torch.no_grad()
def raw_pose(model, normalizer, device, img_gray, view_idx, mirror):
    """Raw denormalized 12-vector, matching usgr inference_simulation.py."""
    if mirror:
        img_gray = cv2.flip(img_gray, 1)
    t = preprocess_image_for_model(img_gray).to(device)
    onehot = torch.zeros(12, dtype=torch.float32)
    onehot[view_idx] = 1.0
    onehot = onehot.unsqueeze(0).to(device)
    pose = normalizer.denormalize(model(t, onehot))[0]
    return pose.float().cpu().numpy()  # [r00..r22, tx, ty, tz]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view-idx", type=int, default=2, help="A4C=2 (invivo mapping)")
    ap.add_argument("--view-name", default="A4C")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_REPO), "result-comparison.csv"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pred = ProbePosePredictor(
        checkpoint_path=f"{WEIGHTS}/ema_11000.pth",
        norm_stats_path=f"{WEIGHTS}/normalization_statistics.pth",
        dino_weights_path=f"{WEIGHTS}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        device=device, dino_model="dinov3_vitl16", image_size=512, use_filter=False,
    )
    model, normalizer = pred.model, pred.normalizer
    assert model is not None and normalizer is not None, "predictor failed to load"

    datasets = [(name, sorted(glob.glob(f"{d}/*.png")), mirror)
                for name, d, mirror in DATASETS]

    cols = (["dataset", "frame_index", "frame_file", "mirror", "view_name", "view_idx"]
            + [f"r{r}{c}" for r in range(3) for c in range(3)]
            + ["tx", "ty", "tz", "det_R_raw", "rot_angle_deg", "trans_norm_cm"])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, files, mirror in datasets:
            print(f"== {name}: {len(files)} frames, mirror={'ON' if mirror else 'OFF'}", flush=True)
            for i, fp in enumerate(files):
                g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                pose = raw_pose(model, normalizer, device, g, args.view_idx, mirror)
                R = pose[:9].reshape(3, 3); T = pose[9:12]
                row = ([name, i, os.path.basename(fp), int(mirror), args.view_name, args.view_idx]
                       + [f"{v:.6f}" for v in pose[:9]]
                       + [f"{T[0]:.6f}", f"{T[1]:.6f}", f"{T[2]:.6f}",
                          f"{np.linalg.det(R):.4f}", f"{geodesic_deg(R):.2f}",
                          f"{np.linalg.norm(T) * 100:.3f}"])
                w.writerow(row)
                if i % 500 == 0:
                    f.flush()
                    print(f"   {name} {i}/{len(files)}", flush=True)
    print(f"-- wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
