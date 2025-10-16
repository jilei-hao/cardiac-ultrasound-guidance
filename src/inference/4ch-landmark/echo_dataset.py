import os, glob
import numpy as np
import cv2
from torch.utils.data import Dataset

LABEL_VALUES = [0, 50, 100, 150, 200]  # map -> {0,1,2,3,4}
LABEL2ID = {v: i for i, v in enumerate(LABEL_VALUES)}
IGNORE_INDEX = 255

def map_label_values(mask_uint8):
    mapped = np.full_like(mask_uint8, IGNORE_INDEX, dtype=np.uint8)
    for v, i in LABEL2ID.items():
        mapped[mask_uint8 == v] = i
    return mapped

def default_pairer(frame_path, frame_dir_name='frame', label_dir_name='label'):
    # e.g., image_..._frame_0003.png -> image_..._frame_0003_label.png
    d, bn = os.path.split(frame_path)
    root = d.replace(os.sep + frame_dir_name, os.sep + label_dir_name)
    base, _ = os.path.splitext(bn)
    cand = os.path.join(root, base + "_label.png")
    return cand

class EchoSegDataset(Dataset):
    def __init__(self, data_root, split_list=None, img_size=224,
                 augment=None, pair_from_frame=True,
                 frame_subdir='frame', label_subdir='label',
                 allow_missing=False, replicate_gray_to_rgb=True):
        self.data_root = data_root
        self.frame_dir = os.path.join(data_root, frame_subdir)
        self.label_dir = os.path.join(data_root, label_subdir)
        self.img_size = img_size
        self.augment = augment
        self.replicate_gray_to_rgb = replicate_gray_to_rgb

        if split_list is None:
            # Take all frames under frame_dir
            frame_paths = sorted(glob.glob(os.path.join(self.frame_dir, "*.png")))
        else:
            with open(split_list, "r") as f:
                names = [l.strip() for l in f if l.strip()]
            frame_paths = [os.path.join(self.frame_dir, n) if not os.path.isabs(n) else n
                           for n in names]

        self.samples = []
        for fp in frame_paths:
            lp = default_pairer(fp) if pair_from_frame else None
            if lp is None or not os.path.exists(lp):
                if not allow_missing:
                    continue
            self.samples.append((fp, lp))

        if len(self.samples) == 0:
            raise RuntimeError("No (frame, label) pairs found. Check paths and naming.")

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # ultrasound is grayscale
        if img is None:
            raise FileNotFoundError(path)
        # resize -> square IMG_SIZE (keep aspect if needed; we assume frames are square 112)
        if self.img_size is not None:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.replicate_gray_to_rgb:
            img = np.stack([img, img, img], axis=-1)
        else:
            img = img[..., None]
        return img

    def _read_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        if self.img_size is not None:
            m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        m = map_label_values(m)
        return m

    def __getitem__(self, idx):
        fpath, lpath = self.samples[idx]
        img = self._read_rgb(fpath)
        mask = self._read_mask(lpath)

        if self.augment is not None:
            out = self.augment(image=img, mask=mask)
            img, mask = out["image"], out["mask"]  # torch tensors (ToTensorV2)
        return img, mask, fpath
