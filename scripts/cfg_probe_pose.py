#!/usr/bin/env python3
"""Classifier-free-guidance test for the probe-pose model.

The USGR guidance model is trained with conditioning dropout
(proportion_unconditioned=0.2), so it supports CFG inference:

    R_g = R_u * (R_u^-1 R_c)^w        (SLERP extrapolation)
    t_g = t_u + w * (t_c - t_u)       (linear extrapolation)

w=1 is plain conditional inference (what ProbePosePredictor does). w>1
amplifies the view-conditioning signal, which may sharpen view discrimination
on an undertrained head. This script measures, on the green A4C set, the
target-A4C |T_rel| / R angle and the per-view argmin at several scales.

Usage:
    python scripts/cfg_probe_pose.py --checkpoint .../ema_XXXX.pth \
        --scales 1.0 1.5 2.0 3.0 --n 60 --image-size 384
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
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from scipy.spatial.transform import Rotation, Slerp
    from inference.probe_pose import ProbePosePredictor, CANONICAL_VIEW_LIST
    from inference.probe_pose.probe_pose_predictor import (
        _nearest_rotation, preprocess_image_for_model, CLASS_LENGTH)

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    pred = ProbePosePredictor(args.checkpoint, args.norm_stats, args.dino_weights, device,
                              image_size=args.image_size, use_filter=False)
    model, nz = pred.model, pred.normalizer
    nviews = len(CANONICAL_VIEW_LIST)

    @torch.no_grad()
    def forward(img_t, onehot):
        pose = nz.denormalize(model(img_t, onehot))[0].float().cpu().numpy()
        return _nearest_rotation(pose[:9].reshape(3, 3)), pose[9:12]

    def cfg(Rc, tc, Ru, tu, w):
        t = tu + w * (tc - tu)
        r_u = Rotation.from_matrix(Ru)
        r_rel = r_u.inv() * Rotation.from_matrix(Rc)
        rv = r_rel.as_rotvec()  # axis * angle; power w = scale angle
        R = (r_u * Rotation.from_rotvec(w * rv)).as_matrix()
        return R, t

    random.seed(0)
    files = random.sample(sorted(glob.glob(f"{args.good_dir}/*.png")), args.n)

    # collect cond predictions for all views + the uncond prediction, per frame
    conds = []   # list of [(R, t) x nviews]
    unconds = [] # list of (R, t)
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        img_t = preprocess_image_for_model(img).to(device)
        row = []
        for v in range(nviews):
            onehot = torch.zeros(1, CLASS_LENGTH, device=device); onehot[0, v] = 1.0
            row.append(forward(img_t, onehot))
        conds.append(row)
        unconds.append(forward(img_t, torch.zeros(1, CLASS_LENGTH, device=device)))

    print(f"checkpoint: {os.path.basename(args.checkpoint)}  image_size={args.image_size}  "
          f"frames: {len(files)} (green A4C)\n")
    for w in args.scales:
        Tn = np.zeros((len(files), nviews)); Ra = np.zeros((len(files), nviews))
        for i in range(len(files)):
            Ru, tu = unconds[i]
            for v in range(nviews):
                Rc, tc = conds[i][v]
                R, t = cfg(Rc, tc, Ru, tu, w)
                Tn[i, v] = np.linalg.norm(t) * 100
                Ra[i, v] = geodesic_deg(R)
        argT = np.bincount(Tn.argmin(1), minlength=nviews)
        argR = np.bincount(Ra.argmin(1), minlength=nviews)
        rankT = Tn.argsort(1).argsort(1)[:, 0].mean()
        rankR = Ra.argsort(1).argsort(1)[:, 0].mean()
        tstd = np.round(np.array([t for _, t in [cfg(conds[i][0][0], conds[i][0][1],
                        unconds[i][0], unconds[i][1], w) for i in range(len(files))]]).std(0) * 100, 2)
        print(f"w={w:4.1f}  A4C |T|={Tn[:,0].mean():5.2f}cm Rang={Ra[:,0].mean():5.1f}deg | "
              f"A4C rank: T={rankT:.2f} R={rankR:.2f} | "
              f"argmin%% T:A4C={100*argT[0]/len(files):4.1f} R:A4C={100*argR[0]/len(files):4.1f} | "
              f"perframe Tstd(cm)={tstd}")
        line = "        per-view |T|: " + "  ".join(
            f"{CANONICAL_VIEW_LIST[v]}={Tn[:, v].mean():.2f}" for v in range(nviews))
        line += "\n        per-view Rang: " + "  ".join(
            f"{CANONICAL_VIEW_LIST[v]}={Ra[:, v].mean():.1f}" for v in range(nviews))
        print(line)


if __name__ == "__main__":
    main()
