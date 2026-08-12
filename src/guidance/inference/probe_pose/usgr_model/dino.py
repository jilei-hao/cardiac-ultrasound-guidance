"""Frozen DINOv3 feature encoder.

Vendored from USGR ``usgr/encoder/dino.py``.

The cardiac wrapper supplies both the inference-only DINOv3 hub path and the
local ViT-L weights path. Training-only model variants and machine-specific
fallback paths are intentionally omitted.
"""

import contextlib
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import warnings


class DinoEncoder(nn.Module):
    """
    A conditioning module that extracts DINO v3 features from images.

    Accepts batches of images and returns feature embeddings suitable
    for conditioning downstream models during training.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Only the backbone used by the deployed probe-pose and quality checkpoints.
    MODEL_CONFIGS = {
        'dinov3_vitl16': {
            'depth': 24,
            'embed_dim': 1024,
        },
    }

    @staticmethod
    def _resolve_model_patch_size(model_patch_size, fallback_patch_size: int) -> int:
        """Normalize a backbone patch size to a single integer."""
        if model_patch_size is None:
            return int(fallback_patch_size)
        if isinstance(model_patch_size, (tuple, list)):
            if len(model_patch_size) != 2 or model_patch_size[0] != model_patch_size[1]:
                raise ValueError(f"Unsupported non-square patch size: {model_patch_size}")
            return int(model_patch_size[0])
        return int(model_patch_size)

    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        weights_path: "str | Path | None" = None,
        repo_path: "str | Path" = '../dinov3',
        patch_size: int = 16,
        image_size: int = 1024,
        output_layers: "list[int] | None" = None,
        device: "str | torch.device" = 'cuda',
        compile_model: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the DINO conditioner.

        Args:
            model_name: Name of the DINO model variant
            weights_path: Path to pretrained weights (.pth file)
            repo_path: Path to local dinov3 repository
            patch_size: Patch size for ViT (typically 14 or 16)
            image_size: Target image size (will be adjusted to be divisible by patch_size)
            output_layers: Which transformer layers to extract features from.
                           If None, uses the last layer based on model depth.
            device: Device to run inference on
            compile_model: Whether to use torch.compile for optimization
            dtype: Data type for inference
        """
        super().__init__()

        # Look up model config
        if model_name not in self.MODEL_CONFIGS:
            available = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        config = self.MODEL_CONFIGS[model_name]

        if weights_path is None:
            raise ValueError("DINOv3 weights_path is required")

        # Use provided output_layers or default to last layer
        if output_layers is None:
            output_layers = [config['depth'] - 1]

        self.model_name = model_name
        self.requested_patch_size = patch_size
        self.image_size = image_size
        self.output_layers = output_layers
        self.embed_dim = config['embed_dim']
        self.device = torch.device(device)
        self.dtype = dtype

        # Load model from local repo
        self.model = torch.hub.load(
            repo_or_dir=str(repo_path),
            model=model_name,
            source='local',
            weights=str(weights_path),
        )
        self.patch_size = self._resolve_model_patch_size(
            getattr(self.model, "patch_size", None),
            fallback_patch_size=patch_size,
        )
        if self.patch_size != int(self.requested_patch_size):
            warnings.warn(
                f"DinoEncoder requested patch_size={self.requested_patch_size}, "
                f"but backbone {self.model_name} uses patch_size={self.patch_size}. "
                "Using the backbone patch size for preprocessing and grid computation.",
                stacklevel=2,
            )

        self.model.eval().to(self.device)

        if compile_model:
            self.model = torch.compile(self.model)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def resize_for_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Resize images to dimensions divisible by patch size.

        Args:
            images: Tensor of shape [B, C, H, W] or [C, H, W]

        Returns:
            Resized tensor with H and W divisible by patch_size
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        B, C, H, W = images.shape

        new_h, new_w = self.get_preprocessed_image_size(H, W)

        return TF.resize(images, [new_h, new_w], antialias=True)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        return TF.normalize(images, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize and normalize images for DINO."""
        images = self.resize_for_patches(images)
        images = self.normalize(images)
        return images

    def forward(
        self,
        images: torch.Tensor,
        reshape: bool = True,
        return_cls: bool = False,
    ) -> torch.Tensor:
        """
        Extract DINO features from a batch of images.

        Args:
            images: Input images [B, C, H, W], values in [0, 1]
            reshape: Whether to reshape features to spatial dimensions
            return_cls: Whether to return CLS token instead of patch features

        Returns:
            Feature tensor. If reshape=True: [B, D, H', W'] where H', W' are patch grid dims
                           If reshape=False: [B, N, D] where N is number of patches
        """
        with torch.no_grad():
            # Preprocess (resize + normalize)
            images = self.preprocess(images)
            images = images.to(device=self.device, dtype=self.dtype)

            # Extract features. autocast only matters for low-precision dtypes;
            # for float32 it is a numerical no-op (and on MPS an unsupported
            # autocast dtype warns + self-disables every call). Vendoring tweak:
            # skip the autocast context unless using fp16/bf16, preserving CUDA
            # numerics while avoiding per-frame MPS warnings.
            if self.dtype in (torch.float16, torch.bfloat16):
                autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self.dtype)
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                features = self.model.get_intermediate_layers(
                    images,
                    n=self.output_layers,
                    reshape=reshape,
                    norm=True,
                )

            # Return the last layer's features (or concatenate if multiple layers)
            if len(features) == 1:
                output = features[0]
            else:
                # Concatenate features from multiple layers along channel dim
                output = torch.cat(features, dim=1)

        return output.detach()

    def get_feature_dim(self) -> int:
        """Return the feature dimension of the model."""
        return self.embed_dim * len(self.output_layers)

    def get_preprocessed_image_size(self, image_h: int, image_w: int) -> "tuple[int, int]":
        """Calculate the resized image dimensions used before feature extraction."""
        h_patches, w_patches = self.get_patch_grid_size(image_h, image_w)
        return h_patches * self.patch_size, w_patches * self.patch_size

    def get_patch_grid_size(self, image_h: int, image_w: int) -> "tuple[int, int]":
        """Calculate the output patch grid dimensions for given input size."""
        h_patches = self.image_size // self.patch_size
        w_patches = max(1, (image_w * self.image_size) // (image_h * self.patch_size))
        return h_patches, w_patches
