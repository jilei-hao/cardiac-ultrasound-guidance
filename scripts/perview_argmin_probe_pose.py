#!/usr/bin/env python3
"""Experiment A: per-view-minimum test for the probe-pose model.

For each green A4C frame, run the model against ALL 6 target views and record
which view yields the smallest |T_rel| (and smallest R angle). If the model is
merely undertrained but conventionally correct, green A4C frames should pick
A4C as the argmin (or A4C's rank should improve with training). If they
systematically pick another view, suspect a view-index / canonical-pose
convention mismatch.

Usage:
    python scripts/perview_argmin_probe_pose.py \
        --checkpoint .../ema_XXXX.pth --good-dir /path/green_set --n 80
"""
import argparse, glob, os, random, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def geodesic_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    from dotenv import load_dotenv
    load_dotenv()
    W = "/Users/jileihao/data/vigil/__demo_weights"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.getenv("PROBE_POSE_WEIGHTS", f"{W}/ema_21000.pth"))
    ap.add_argument("--norm-stats", default=os.getenv("PROBE_POSE_NORM_STATS", f"{W}/normalization_statistics.pth"))
    ap.add_argument("--dino-weights", default=os.getenv("PROBE_POSE_DINO_WEIGHTS", f"{W}/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"))
    ap.add_argument("--good-dir", default="/Users/jileihao/data/vigil/2025-nov-demo_2246f_green")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--flip", action="store_true",
                    help="horizontally mirror frames before inference (L-R chirality test)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from guidance.inference.probe_pose import ProbePosePredictor, CANONICAL_VIEW_LIST

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    pred = ProbePosePredictor(args.checkpoint, args.norm_stats, args.dino_weights, device,
                              image_size=args.image_size)
    nviews = len(CANONICAL_VIEW_LIST)

    random.seed(0)
    files = random.sample(sorted(glob.glob(f"{args.good_dir}/*.png")), args.n)

    # per-frame matrices: |T_rel| (cm) and R angle (deg) for every target view
    Tn = np.zeros((len(files), nviews))
    Ra = np.zeros((len(files), nviews))
    for i, fp in enumerate(files):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if args.flip:
            img = cv2.flip(img, 1)
        for v in range(nviews):
            p = pred.predict(img, v)
            Tn[i, v] = np.linalg.norm(p.T_rel) * 100
            Ra[i, v] = geodesic_deg(p.R_rel)

    print(f"checkpoint: {os.path.basename(args.checkpoint)}  frames: {len(files)} (green A4C)\n")
    hdr = "  ".join(f"{v:>9s}" for v in CANONICAL_VIEW_LIST)
    print(f"{'':14s}{hdr}")
    print(f"{'|T| mean cm':14s}" + "  ".join(f"{Tn[:, v].mean():9.2f}" for v in range(nviews)))
    print(f"{'Rang mean deg':14s}" + "  ".join(f"{Ra[:, v].mean():9.1f}" for v in range(nviews)))

    for name, M in [("|T_rel|", Tn), ("R angle", Ra)]:
        argmin = M.argmin(axis=1)
        counts = np.bincount(argmin, minlength=nviews)
        # mean rank of A4C (0 = best)
        ranks = M.argsort(axis=1).argsort(axis=1)[:, 0]
        print(f"\nargmin by {name}:")
        for v in range(nviews):
            bar = "#" * int(round(40 * counts[v] / len(files)))
            print(f"  {CANONICAL_VIEW_LIST[v]:>9s}: {counts[v]:4d} ({100*counts[v]/len(files):5.1f}%) {bar}")
        print(f"  A4C mean rank (0=best, {nviews-1}=worst): {ranks.mean():.2f}")


if __name__ == "__main__":
    main()
