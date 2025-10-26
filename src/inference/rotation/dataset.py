from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Rot6DDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size=224, with_anchor: bool = False):
        self.df = df.reset_index(drop=True)
        self.with_anchor = with_anchor
        self.t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # Build per-group anchor path by reading the FIRST row of the group's original CSV.
        # If the path in the CSV is relative or missing, resolve relative to the CSV's directory;
        # if still missing, fall back to the first available frame in this group.
        self.anchor_path = {}
        for gid, g in self.df.groupby("group_id"):
            csv_path = g["csv_path"].iloc[0] if "csv_path" in g.columns else None
            anchor_fp = None
            if csv_path and Path(csv_path).is_file():
                dfc = pd.read_csv(csv_path)
                if len(dfc) > 0:
                    raw = str(dfc.iloc[0]["frame_path"])
                    cand = Path(raw)
                    if not cand.is_absolute():
                        cand = Path(csv_path).parent / raw
                    if cand.is_file():
                        anchor_fp = str(cand)

            if anchor_fp is None:
                # fallback: first available frame listed in this df group
                anchor_fp = g["frame_path"].iloc[0]

            self.anchor_path[gid] = anchor_fp

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["frame_path"]).convert("RGB")
        x = self.t(img)

        R = torch.tensor([[row["r00"], row["r01"], row["r02"]],
                          [row["r10"], row["r11"], row["r12"]],
                          [row["r20"], row["r21"], row["r22"]]], dtype=torch.float32)
        t6 = torch.tensor([row[f"t{i}"] for i in range(6)], dtype=torch.float32)

        meta = {
            "frame_path": row["frame_path"],
            "subject_id": row["subject_id"],
            "collection_id": row["collection_id"],
            "group_id": row["group_id"],
        }

        if self.with_anchor:
            a_path = self.anchor_path[row["group_id"]]
            a = self.t(Image.open(a_path).convert("RGB"))
            return x, a, t6, R, meta
        else:
            return x, t6, R, meta

