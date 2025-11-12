# models/convnext_fpn.py
import os
import torch
import torch.nn as nn

# -------------------------
# Small conv-bn-relu block
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


# -------------------------
# FPN Decoder
# -------------------------
class FPNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_classes=5):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(out_channels, out_channels) for _ in in_channels])
        self.head = nn.Sequential(
            ConvBNAct(out_channels, out_channels),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )

    def forward(self, feats):
        # feats = [C2, C3, C4, C5], low->high resolution
        c2, c3, c4, c5 = feats
        p5 = self.lats[3](c5)
        p4 = self.lats[2](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lats[1](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lats[0](c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth[0](p2)
        return self.head(p2)


# -------------------------
# Hub-loaded ConvNeXt → C2..C5 feature adapter
# -------------------------
class HubConvNeXtFeatures(nn.Module):
    """
    Wrap a ConvNeXt (loaded via torch.hub from local DINOv3 repo) to return
    [C2, C3, C4, C5] feature maps like timm(features_only=True).
    """
    def __init__(self, weights: str | None = None, dinov3_repo_dir: str | None = None):
        super().__init__()
        # Local DINOv3 repo path
        if dinov3_repo_dir is None:
            print("Using default DINOv3 repo path.")
            repo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dinov3")
        entry = "dinov3_convnext_base"
        
        assert os.path.isdir(repo_dir), f"repo_dir does not exist: {repo_dir}"
        # Load backbone from local repo (no internet, no pretrained weights)
        try:
            self.model = torch.hub.load(repo_dir, entry, source='local', trust_repo=True, pretrained=False)
        except TypeError:
            # older torch without 'trust_repo' kwarg
            self.model = torch.hub.load(repo_dir, entry, source='local', pretrained=False)

        # The DINOv3 ConvNeXt model is already the backbone
        # No need to unwrap

        # RGB images (3 channels)
        self.in_chans = 3

        # Estimate output channels (C2..C5) with a tiny dry run
        self._out_channels = self._infer_out_channels()

    # --- utilities ---
    def _first_stem_conv(self):
        """
        Find the first Conv2d in downsample_layers[0] (ConvNeXt stem).
        """
        m = self.model
        if not hasattr(m, "downsample_layers"):
            return None, None, None
        stem = m.downsample_layers[0]
        if isinstance(stem, nn.Sequential):
            for idx, sub in enumerate(stem):
                if isinstance(sub, nn.Conv2d):
                    return stem, idx, sub
        # fallback: search globally (rarely needed)
        for name, mod in m.named_modules():
            if isinstance(mod, nn.Conv2d):
                parent = m
                parent_idx = None
                # not strictly necessary to return parent/idx in fallback
                return parent, parent_idx, mod
        return None, None, None


    def _forward_extract(self, x: torch.Tensor):
        """
        Reproduce ConvNeXt forward but expose stage outputs.
        Assumes attributes: downsample_layers (4) and stages (4).
        """
        m = self.model
        assert hasattr(m, "downsample_layers") and hasattr(m, "stages"), \
            "ConvNeXt with downsample_layers & stages is required for feature extraction."

        feats = []
        y = m.downsample_layers[0](x)
        y = m.stages[0](y); feats.append(y)  # C2 (stride 4)

        y = m.downsample_layers[1](y)
        y = m.stages[1](y); feats.append(y)  # C3 (stride 8)

        y = m.downsample_layers[2](y)
        y = m.stages[2](y); feats.append(y)  # C4 (stride 16)

        y = m.downsample_layers[3](y)
        y = m.stages[3](y); feats.append(y)  # C5 (stride 32)
        return feats

    def _infer_out_channels(self):
        """
        Run a tiny dry pass on CPU to get channel dims for C2..C5.
        """
        dev_backup = next(self.model.parameters()).device
        self.model.to("cpu").eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)  # RGB images
            feats = self._forward_extract(x)
        chs = [f.shape[1] for f in feats]
        # move back
        self.model.to(dev_backup)
        return chs

    # mimic timm's interface that your FPN code expects
    class _FeatInfo:
        def __init__(self, channels):
            self._channels = list(channels)
        def channels(self):
            return list(self._channels)

    @property
    def feature_info(self):
        return self._FeatInfo(self._out_channels)

    def forward(self, x):
        return self._forward_extract(x)


# -------------------------
# Main model (hub backbone + FPN)
# -------------------------
class ConvNeXtFPN(nn.Module):
    def __init__(self, num_classes=5, hub_weights=None, dinov3_repo_dir: str | None = None):
        """
        Load ConvNeXt from local DINOv3 repo via torch.hub.
        """
        super().__init__()

        self.backbone = HubConvNeXtFeatures(weights=hub_weights, dinov3_repo_dir=dinov3_repo_dir)
        print("hub loaded")

        in_chs = self.backbone.feature_info.channels()
        self.decoder = FPNDecoder(in_chs, out_channels=256, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)             # list of 4 tensors (C2..C5)
        logits = self.decoder(feats)         # predict at P2
        logits = nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    # Optional legacy hook — if you still want to try an extra encoder ckpt
    def load_encoder_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # best-effort partial load into hub backbone
        target = getattr(self.backbone, "model", self.backbone)
        tstate = target.state_dict()
        new_sd, matched = {}, 0
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor): continue
            if k in tstate and tstate[k].shape == v.shape:
                new_sd[k] = v; matched += 1
        missing, unexpected = target.load_state_dict(new_sd, strict=False)
        print(f"[load_encoder_ckpt] matched={matched} missing={len(missing)} unexpected={len(unexpected)}")
