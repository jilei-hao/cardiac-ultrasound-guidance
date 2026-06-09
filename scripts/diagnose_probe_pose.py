#!/usr/bin/env python3
"""Diagnose probe-pose model quality on a labelled frame set.

For frames that are ALREADY a good target view, the predicted relative transform
to that view should be ~identity (|T_rel| ~ 0, small R angle). This script
measures that on a "good" frame folder (e.g. the green-filtered A4C set) vs a
baseline folder, and checks whether the head actually responds to the image
(per-frame variance) or just emits a per-view constant (undertrained).

Usage (paths default to the demo assets / green set):
    python scripts/diagnose_probe_pose.py \
        --checkpoint .../ema_XXXX.pth \
        --good-dir   /path/2025-nov-demo_2246f_green \
        --base-dir   /path/2025-nov-demo_2246f \
        --view A4C
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
    ap.add_argument("--checkpoint", default=os.getenv("PROBE_POSE_WEIGHTS", f"{W}/ema_9000.pth"))
    ap.add_argument("--norm-stats", default=os.getenv("PROBE_POSE_NORM_STATS", f"{W}/normalization_statistics.pth"))
    ap.add_argument("--dino-weights", default=os.getenv("PROBE_POSE_DINO_WEIGHTS", f"{W}/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"))
    ap.add_argument("--good-dir", default="/Users/jileihao/data/vigil/2025-nov-demo_2246f_green")
    ap.add_argument("--base-dir", default="/Users/jileihao/data/vigil/2025-nov-demo_2246f")
    ap.add_argument("--view", default="A4C")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from inference.probe_pose import ProbePosePredictor, CANONICAL_VIEW_LIST
    from inference.probe_pose.usgr_model import Normalizer

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    view_idx = CANONICAL_VIEW_LIST.index(args.view)

    nz = Normalizer.load(args.norm_stats)
    print(f"normalizer T mean(cm)={np.round(nz.mean.numpy()[9:12]*100,2)} "
          f"std(cm)={np.round(nz.std.numpy()[9:12]*100,2)}\n")

    pred = ProbePosePredictor(args.checkpoint, args.norm_stats, args.dino_weights, device)

    def run(files):
        T, ang = [], []
        for fp in files:
            p = pred.predict(cv2.imread(fp, cv2.IMREAD_GRAYSCALE), view_idx)
            T.append(p.T_rel); ang.append(geodesic_deg(p.R_rel))
        return np.array(T), np.array(ang)

    random.seed(0)
    good = random.sample(sorted(glob.glob(f"{args.good_dir}/*.png")), args.n)
    good_names = {os.path.basename(x) for x in glob.glob(f"{args.good_dir}/*.png")}
    base = random.sample([f for f in glob.glob(f"{args.base_dir}/*.png")
                          if os.path.basename(f) not in good_names], args.n)

    print(f"== target = {args.view} ==  (good frames should give |T_rel|~0, small R angle)")
    for label, fs in [(f"GOOD({args.view})", good), ("baseline", base)]:
        T, a = run(fs)
        nrm = np.linalg.norm(T, axis=1) * 100
        print(f"  {label:14s} |T_rel|cm mean={nrm.mean():5.2f} median={np.median(nrm):5.2f} "
              f"min={nrm.min():5.2f} max={nrm.max():5.2f} | Rangle mean={a.mean():5.1f}deg")
    Tg, _ = run(good)
    print(f"\n  image responsiveness (good frames) per-axis T std(cm)={np.round(Tg.std(0)*100,2)} "
          f"-> ~0 means the head ignores the image (undertrained)")


if __name__ == "__main__":
    main()
