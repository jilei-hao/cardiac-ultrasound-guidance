from __future__ import annotations
import os, torch, torch.nn as nn

def strip_prefixes(sd: dict, prefixes=("module.","backbone.","model.","encoder.","student.","teacher.")):
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def count_matching_keys(model: nn.Module, sd: dict) -> int:
    msd = model.state_dict()
    return sum((k in msd) and (v.shape == msd[k].shape) for k, v in sd.items())

def load_backbone_from_local_hub(repo: str, entry: str):
    if not os.path.isdir(repo):
        raise RuntimeError(f"Local hub repo not found: {repo}")
    try:
        # Try to load without pretrained weights by setting pretrained=False
        model = torch.hub.load(repo, entry, source="local", trust_repo=True, pretrained=False)
    except TypeError:
        try:
            model = torch.hub.load(repo, entry, source="local", pretrained=False)
        except:
            # If pretrained=False doesn't work, try without any pretrained parameter
            model = torch.hub.load(repo, entry, source="local")
    return getattr(model, "backbone", model)

def load_local_weights(backbone: nn.Module, pth: str, strict_key_match_min: int = 200):
    ckpt = torch.load(pth, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        sd = ckpt
    sd = strip_prefixes(sd)
    try:
        matched = count_matching_keys(backbone, sd)
        msg = backbone.load_state_dict(sd, strict=False)
    except RuntimeError:
        sd2 = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
        matched = count_matching_keys(backbone, sd2)
        msg = backbone.load_state_dict(sd2, strict=False)
    print("[dinov3] load_state_dict:", msg)
    print(f"[dinov3] matched keys: {matched}")
    if matched < strict_key_match_min:
        raise RuntimeError(f"Only {matched} keys matched (<{strict_key_match_min}). Check repo/entry/weights.")
    return backbone

def infer_feat_dim(backbone: nn.Module, img_size: int = 224) -> int:
    backbone.eval()
    with torch.no_grad():
        x = torch.zeros(1,3,img_size,img_size)
        if hasattr(backbone, "forward_features"):
            f = backbone.forward_features(x)
            # Handle case where forward_features returns a dict
            if isinstance(f, dict):
                # Try common keys for feature tensors
                for key in ["x_norm_clstoken", "x_norm_patchtokens", "x_prenorm", "last_hidden_state"]:
                    if key in f:
                        f = f[key]
                        break
                else:
                    # If no common key found, take the first tensor value
                    f = next(v for v in f.values() if isinstance(v, torch.Tensor))
            if f.ndim == 4: f = f.mean((-2,-1))
        else:
            if hasattr(backbone, "features"):
                f = backbone.features(x)
                if hasattr(backbone, "avgpool"): f = backbone.avgpool(f)
                if f.ndim == 4: f = f.mean((-2,-1))
                if hasattr(backbone, "norm"):
                    try:    f = backbone.norm(f)
                    except: f = backbone.norm(f[:, :, None, None]).squeeze(-1).squeeze(-1)
            else:
                raise RuntimeError("Backbone lacks forward_features/features")
    return int(f.shape[1])

class DINOConvNeXt6DDual(nn.Module):
    """Shared ConvNeXt backbone with two heads:
         - pair_head: takes [feat_current, feat_anchor] (2*D -> 6)
         - solo_head: takes [feat_current] (D -> 6)
       Forward can accept x (current) and optional x0 (anchor)."""
    def __init__(self, backbone: nn.Module, feat_dim: int, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        # strip classifier-like attributes if present
        for name in ["head","classifier","fc"]:
            if hasattr(self.backbone, name):
                setattr(self.backbone, name, nn.Identity())
        if head_dropout and head_dropout > 0.0:
            self.pair_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(2*feat_dim, 6))
            self.solo_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(feat_dim, 6))
        else:
            self.pair_head = nn.Linear(2*feat_dim, 6)
            self.solo_head = nn.Linear(feat_dim, 6)

    def _features(self, x):
        if hasattr(self.backbone, "forward_features"):
            f = self.backbone.forward_features(x)
            # Handle case where forward_features returns a dict
            if isinstance(f, dict):
                # Try common keys for feature tensors
                for key in ["x_norm_clstoken", "x_norm_patchtokens", "x_prenorm", "last_hidden_state"]:
                    if key in f:
                        f = f[key]
                        break
                else:
                    # If no common key found, take the first tensor value
                    f = next(v for v in f.values() if isinstance(v, torch.Tensor))
            if f.ndim == 4: f = f.mean((-2,-1))
        else:
            x = self.backbone.features(x)
            if hasattr(self.backbone, "avgpool"): x = self.backbone.avgpool(x)
            f = x.mean((-2,-1)) if x.ndim == 4 else x
            if hasattr(self.backbone, "norm"):
                try:    f = self.backbone.norm(f)
                except: f = self.backbone.norm(f[:, :, None, None]).squeeze(-1).squeeze(-1)
        return f

    def forward(self, x, x0=None, return_both=True):
        f = self._features(x)
        y_solo = self.solo_head(f)
        y_pair = None
        if x0 is not None:
            f0 = self._features(x0)
            y_pair = self.pair_head(torch.cat([f, f0], dim=1))
        return {"solo": y_solo, "pair": y_pair} if return_both else y_solo
