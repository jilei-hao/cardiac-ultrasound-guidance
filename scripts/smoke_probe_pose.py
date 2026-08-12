#!/usr/bin/env python3
"""Smoke test for the USGR probe-pose predictor integration (Verification step 1).

Loads the vendored PoseTransformer + checkpoint + normalizer from the paths in
``.env`` (or CLI args) and runs one forward pass per standard view, asserting the
output is a well-formed [R|T] pose. Run this on a machine with torch + the three
assets before launching the full UI.

Usage:
    python scripts/smoke_probe_pose.py                 # paths from .env
    python scripts/smoke_probe_pose.py --frame path.png
    python scripts/smoke_probe_pose.py \
        --checkpoint .../ema_9000.pth \
        --norm-stats .../normalization_statistics.pth \
        --dino-weights .../dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
"""

import argparse
import os
import sys

import numpy as np

# Make ``src`` importable when run from the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.getenv("PROBE_POSE_WEIGHTS"))
    ap.add_argument("--norm-stats", default=os.getenv("PROBE_POSE_NORM_STATS"))
    ap.add_argument("--dino-weights", default=os.getenv("PROBE_POSE_DINO_WEIGHTS"))
    ap.add_argument("--frame", default=None, help="optional grayscale image to use")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    ap.add_argument("--dino-model", default="dinov3_convnext_base",
                    help="encoder variant (e.g. dinov3_vitl16 for the 0610 run)")
    ap.add_argument("--image-size", type=int, default=512, help="encoder input size")
    args = ap.parse_args()

    missing = [n for n, v in [
        ("PROBE_POSE_WEIGHTS", args.checkpoint),
        ("PROBE_POSE_NORM_STATS", args.norm_stats),
        ("PROBE_POSE_DINO_WEIGHTS", args.dino_weights),
    ] if not v]
    if missing:
        print(f"❌ Missing asset paths: {missing}. Set them in .env or pass via CLI.")
        return 2

    import torch
    from guidance.inference.probe_pose import ProbePosePredictor, CANONICAL_VIEW_LIST

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"-- device: {device}")

    # Build a frame: load if given, else a synthetic 512x512 gradient.
    if args.frame:
        import cv2
        frame = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"❌ Could not read frame: {args.frame}")
            return 2
    else:
        yy, xx = np.mgrid[0:512, 0:512]
        frame = ((xx + yy) % 256).astype(np.uint8)
    print(f"-- frame: {frame.shape} {frame.dtype}")

    predictor = ProbePosePredictor(
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        dino_weights_path=args.dino_weights,
        device=device,
        dino_model=args.dino_model,
        image_size=args.image_size,
    )
    if predictor.model is None:
        print("❌ Predictor failed to load (see errors above).")
        return 1

    ok = True
    for idx, name in enumerate(CANONICAL_VIEW_LIST):
        pred = predictor.predict(frame, idx)
        if pred is None:
            print(f"❌ view {idx} ({name}): predict() returned None")
            ok = False
            continue
        det = float(np.linalg.det(pred.R_rel))
        finite = bool(np.all(np.isfinite(pred.R_rel)) and np.all(np.isfinite(pred.T_rel)))
        shape_ok = pred.R_rel.shape == (3, 3) and pred.T_rel.shape == (3,)
        status = "✅" if (shape_ok and finite and pred.valid) else "⚠️"
        print(f"{status} view {idx} ({name:9s}): det(R)={det:+.3f} valid={pred.valid} "
              f"finite={finite} T(m)={np.round(pred.T_rel, 4).tolist()}")
        ok = ok and shape_ok and finite

    print("\n✅ Smoke test passed" if ok else "\n❌ Smoke test had failures")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
