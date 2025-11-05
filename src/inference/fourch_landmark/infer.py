import os, glob, argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .echo_dataset import EchoSegDataset, LABEL_VALUES
from .simple_convnext_fpn import SimpleConvNeXtFPN

# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # ----- dataset / I/O -----
    ap.add_argument("--data_root", type=str, required=False,
                    help="Root with frame/ and (optional) label/. Only needed for list/split mode.")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--list", type=str, default=None)
    ap.add_argument("--images_dir", type=str, default=None,
                    help="If set, run inference on ALL images in this folder (no labels needed).")
    ap.add_argument("--recursive", type=int, default=1,
                    help="Folder mode: recurse into subfolders (1=yes, 0=no).")
    ap.add_argument("--outdir", type=str, default="preds")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=512)

    # ----- model checkpoint -----
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to .pt saved by training (contains 'model' key).")



    # ----- intensity preproc (folder mode only; applied to model input) -----
    ap.add_argument("--preproc", type=str, default="none",
                    choices=["none", "clahe", "equalize", "clip", "match"],
                    help="Intensity preproc applied to model input only.")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--clip_low", type=float, default=2.0)
    ap.add_argument("--clip_high", type=float, default=98.0)
    ap.add_argument("--hist_ref", type=str, default=None,
                    help="Reference image path for 'match' mode.")
    return ap.parse_args()


# -----------------------
# Visualization helpers
# -----------------------
def colorize(mask_id):
    palette = np.array([
        [0, 0, 0],       # bg
        [0, 128, 255],   # c1
        [0, 255, 0],     # c2
        [255, 0, 0],     # c3
        [255, 255, 0],   # c4
    ], dtype=np.uint8)
    return palette[mask_id.clip(0, len(palette)-1)]


def _percentile_clip_to_uint8(gray, p_low=2.0, p_high=98.0):
    lo = np.percentile(gray, p_low)
    hi = np.percentile(gray, p_high)
    if hi <= lo:
        lo, hi = 0.0, 255.0
    g = np.clip((gray.astype(np.float32) - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


def _center_crop_box(h, w, frac=2.0/3.0):
    ch = max(1, int(round(h * frac)))
    cw = max(1, int(round(w * frac)))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    y1 = min(h, y0 + ch)
    x1 = min(w, x0 + cw)
    return y0, y1, x0, x1


# -----------------------
# Folder dataset (no labels)
# -----------------------
class FolderImageDataset(Dataset):
    """
    Loads images from a directory (no labels).
    Optionally applies intensity pre-processing ONLY to the model input.
    If clarius=True, center-crops the middle 2/3 before preproc+transform and
    returns the crop box + original size so we can paste predictions back.
    """
    def __init__(self, images_dir, transform, recursive=True,
                 exts=("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
                 preproc="none", clahe_clip=2.0, clahe_grid=8,
                 clip_low=2.0, clip_high=98.0, hist_ref=None,
                 clarius=False):
        self.images_dir = images_dir
        self.transform = transform
        self.preproc = preproc
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = int(clahe_grid)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.hist_ref = hist_ref
        self.clarius = bool(clarius)

        pattern = "**/*" if recursive else "*"
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(images_dir, f"{pattern}.{ext}"), recursive=recursive))
            paths.extend(glob.glob(os.path.join(images_dir, f"{pattern}.{ext.upper()}"), recursive=recursive))
        self.paths = sorted(set(paths))
        if not self.paths:
            raise RuntimeError(f"No images found in {images_dir} (recursive={recursive}).")

        # resources
        self._clahe = None
        if self.preproc == "clahe":
            self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))

        self._hist_ref_img = None
        if self.preproc == "match":
            if self.hist_ref is None:
                print("[infer] 'match' mode requested but --hist_ref not provided; falling back to CLAHE.")
                self.preproc = "clahe"
                self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
            else:
                ref = cv2.imread(self.hist_ref, cv2.IMREAD_GRAYSCALE)
                if ref is None:
                    print(f"[infer] Could not read hist_ref '{self.hist_ref}'; falling back to CLAHE.")
                    self.preproc = "clahe"
                    self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
                else:
                    self._hist_ref_img = ref

    def __len__(self):
        return len(self.paths)

    def _apply_preproc(self, gray):
        g = gray
        if self.preproc == "none":
            return g
        elif self.preproc == "equalize":
            return cv2.equalizeHist(g)
        elif self.preproc == "clahe":
            return self._clahe.apply(g)
        elif self.preproc == "clip":
            return _percentile_clip_to_uint8(g, self.clip_low, self.clip_high)
        elif self.preproc == "match":
            try:
                from skimage.exposure import match_histograms
                matched = match_histograms(g.astype(np.float32), self._hist_ref_img.astype(np.float32))
                matched = np.clip(matched, 0, 255).astype(np.uint8)
                return matched
            except Exception as e:
                print(f"[infer] Histogram matching failed ({e}); falling back to CLAHE.")
                if self._clahe is None:
                    self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
                return self._clahe.apply(g)
        else:
            return g

    def __getitem__(self, idx):
        f = self.paths[idx]
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f)

        H0, W0 = gray.shape[:2]
        if self.clarius:
            y0, y1, x0, x1 = _center_crop_box(H0, W0, frac=2.0/3.0)
            roi = gray[y0:y1, x0:x1]
            roi_proc = self._apply_preproc(roi)
            img_rgb = np.stack([roi_proc, roi_proc, roi_proc], axis=-1)
            x = self.transform(image=img_rgb)["image"]
            return x, f, np.array([y0, y1, x0, x1], dtype=np.int32), np.array([H0, W0], dtype=np.int32)
        else:
            gray_proc = self._apply_preproc(gray)
            img_rgb = np.stack([gray_proc, gray_proc, gray_proc], axis=-1)
            x = self.transform(image=img_rgb)["image"]
            return x, f


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    # Simple transforms for validation
    from torchvision import transforms
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Choose dataset mode
    if args.images_dir:
        ds = FolderImageDataset(
            args.images_dir, transform=tfms, recursive=bool(args.recursive),
            preproc=args.preproc, clahe_clip=args.clahe_clip, clahe_grid=args.clahe_grid,
            clip_low=args.clip_low, clip_high=args.clip_high, hist_ref=args.hist_ref,
            clarius=False
        )
        folder_mode = True
    else:
        if not args.data_root:
            raise SystemExit("--data_root is required for list/split mode. Or pass --images_dir for folder mode.")
        list_file = args.list or (args.split + ".txt")
        ds = EchoSegDataset(args.data_root, split_list=list_file, img_size=args.img_size, augment=tfms)
        folder_mode = False

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Model: Simple ConvNeXt + FPN ----------------
    model = SimpleConvNeXtFPN(
        num_classes=len(LABEL_VALUES),
        weights=None  # checkpoint will provide weights
    )

    # Load training checkpoint (robust)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
        print("[infer] Loaded checkpoint strictly.")
    except Exception as e:
        print(f"[infer] Strict load failed ({e}). Retrying non-strict...")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[infer] Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device).eval()

    # ---------------- Output dirs ----------------
    os.makedirs(args.outdir, exist_ok=True)
    subdir_mask = os.path.join(args.outdir, "mask")
    subdir_frame = os.path.join(args.outdir, "frame")
    os.makedirs(subdir_frame, exist_ok=True)
    os.makedirs(subdir_mask, exist_ok=True)

    # ---------------- Inference loops ----------------
    with torch.no_grad():
        if folder_mode:
            # Standard folder mode
            for imgs, fpaths in dl:
                imgs = imgs.to(device)
                logits = model(imgs)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for p, f in zip(preds, fpaths):
                    gray_orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    H0, W0 = gray_orig.shape[:2]
                    p_up = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_NEAREST)

                    color = colorize(p_up)
                    overlay = (0.6 * cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR) + 0.4 * color).clip(0, 255).astype(np.uint8)

                    base_name = os.path.splitext(os.path.basename(f))[0]
                    original_vals = np.array(LABEL_VALUES, dtype=np.uint8)
                    out_mask = original_vals[p_up]
                    cv2.imwrite(os.path.join(subdir_frame, base_name + ".png"), gray_orig)
                    cv2.imwrite(os.path.join(subdir_mask, base_name + "_pred.png"), out_mask)
                    cv2.imwrite(os.path.join(args.outdir, base_name + "_overlay.png"), overlay)
        else:
            # list/split mode (labels exist but unused here)
            for imgs, _, fpaths in dl:
                imgs = imgs.to(device)
                logits = model(imgs)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for p, f in zip(preds, fpaths):
                    gray_orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    H0, W0 = gray_orig.shape[:2]
                    p_up = cv2.resize(p, (W0, H0), interpolation=cv2.INTER_NEAREST)

                    color = colorize(p_up)
                    overlay = (0.6 * cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR) + 0.4 * color).clip(0, 255).astype(np.uint8)

                    base_name = os.path.splitext(os.path.basename(f))[0]
                    original_vals = np.array(LABEL_VALUES, dtype=np.uint8)
                    out_mask = original_vals[p_up]
                    cv2.imwrite(os.path.join(subdir_frame, base_name + ".png"), gray_orig)
                    cv2.imwrite(os.path.join(subdir_mask, base_name + "_pred.png"), out_mask)
                    cv2.imwrite(os.path.join(args.outdir, base_name + "_overlay.png"), overlay)


if __name__ == "__main__":
    main()
