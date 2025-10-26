import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch, pandas as pd
from torch.utils.data import DataLoader
from rot6d_pose.config import load_config
from rot6d_pose.utils.io import load_all_csvs
from rot6d_pose.utils.splits import subject_split
from rot6d_pose.data.dataset import Rot6DDataset
from rot6d_pose.utils.rotmat import rot6d_to_matrix, geodesic_angle_deg
from rot6d_pose.models.dinov3_convnext import load_backbone_from_local_hub, infer_feat_dim, DINOConvNeXt6DDual

def write_predictions_csv(path, rows):
    out_csv = Path(path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows → {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="Path to best.pt (contains 'model' state_dict)")
    # Only solo inference mode supported
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_all_csvs(cfg["dataset"]["root_preproc"], cfg["dataset"]["csv_basename"])
    _, _, te, _, _, _ = subject_split(df, cfg["dataset"]["train_subjects"], cfg["dataset"]["test_subjects"],
                                      cfg["dataset"]["val_fraction"], cfg["seed"])

    # build model
    bb = load_backbone_from_local_hub(cfg["model"]["dinov3_repo"], cfg["model"]["hub_entry"])
    feat_dim = infer_feat_dim(bb, cfg["dataset"]["img_size"])
    model = DINOConvNeXt6DDual(bb, feat_dim, head_dropout=cfg["model"].get("head_dropout", 0.0)).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    out_root = Path(cfg["output_dir"]) / "preds"; out_root.mkdir(parents=True, exist_ok=True)

    def run_and_save():
        by_group = te.groupby("group_id")
        for group_id, part in by_group:
            subject_id, collection_id = group_id.split("/", 1)
            ds = Rot6DDataset(part, cfg["dataset"]["img_size"], with_anchor=False)
            dl = torch.utils.data.DataLoader(ds, batch_size=cfg["dataset"]["batch_size"],
                                             shuffle=False, num_workers=2, pin_memory=True)
            rows = []
            with torch.no_grad():
                for batch in dl:
                    imgs, R6, Rgt, meta = batch
                    imgs = imgs.to(device, non_blocking=True)
                    out = model(imgs, x0=None, return_both=True)
                    y6 = out["solo"]
                    Rpred = rot6d_to_matrix(y6)
                    Rgt = Rgt.to(device)
                    ang = geodesic_angle_deg(Rpred, Rgt).cpu().numpy()
                    Rpred = Rpred.cpu().numpy()
                    for k in range(Rpred.shape[0]):
                        r = Rpred[k]
                        rows.append({
                            "frame_path": meta["frame_path"][k],
                            "pred_r00": r[0,0], "pred_r01": r[0,1], "pred_r02": r[0,2],
                            "pred_r10": r[1,0], "pred_r11": r[1,1], "pred_r12": r[1,2],
                            "pred_r20": r[2,0], "pred_r21": r[2,1], "pred_r22": r[2,2],
                            "err_deg": float(ang[k]),
                        })
            out_dir = out_root / subject_id / collection_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "predictions.csv"
            write_predictions_csv(out_csv, rows)

    run_and_save()

if __name__ == "__main__":
    main()
