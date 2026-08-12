"""Guidance model classes.

Vendored from USGR ``usgr/guidance/model.py``.

Vendoring change
----------------
``PoseTransformer.__init__`` gains two optional args -- ``dino_repo_path`` and
``dino_weights_path`` -- which are forwarded to the frozen ``DinoEncoder`` so the
host app can point the backbone at the dinov3 hub bundled under
``src/inference/dinov3/`` and a locally stored ConvNeXt-Base weights file.
When both are ``None`` the behaviour is identical to upstream (DinoEncoder uses
its own ``repo_path``/``MODEL_CONFIGS`` defaults).
"""

import torch.nn as nn

from .transformer import TransformerHead
from .dino import DinoEncoder


class PoseTransformer(nn.Module):
    """
    A transformer model for pose estimation built on top of a DINOv3 pretrained encoder.

    The encoder is stored as a non-registered attribute to exclude it from:
    - state_dict (checkpoints) - encoder weights are static/pretrained
    - parameters() iteration - encoder is frozen, not trained
    - EMA tracking - no need to track frozen weights
    """
    def __init__(
        self,
        dino_model='dinov3_vit7b16',
        embed_dim=512,
        depth=4,
        num_heads=8,
        use_conditioning_token=True,
        conditioning_dim=12,
        image_size=384,
        compile_encoder=True,
        device='cuda',
        dino_repo_path=None,
        dino_weights_path=None,
    ):
        super().__init__()
        # Look up embedding dimension from encoder config
        in_channels = DinoEncoder.MODEL_CONFIGS[dino_model]['embed_dim']

        # Build kwargs for the frozen encoder, forwarding the backbone repo/weights
        # paths only when the host app supplies them (otherwise upstream defaults).
        encoder_kwargs = dict(
            model_name=dino_model,
            image_size=image_size,
            compile_model=compile_encoder,
            device=device,
        )
        if dino_repo_path is not None:
            encoder_kwargs['repo_path'] = dino_repo_path
        if dino_weights_path is not None:
            encoder_kwargs['weights_path'] = dino_weights_path

        # Store encoder without registering as submodule (frozen, pretrained)
        object.__setattr__(self, '_encoder', DinoEncoder(**encoder_kwargs))
        self.transformer_head = TransformerHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_pose_token=True,
            use_conditioning_token=use_conditioning_token,
            conditioning_dim=conditioning_dim,
        )

    @property
    def encoder(self) -> DinoEncoder:
        """Access the frozen encoder."""
        return self._encoder

    def forward(self, x, y=None):
        x = self.encoder(x)
        x, pose = self.transformer_head(x, y)
        return pose

    def to(self, *args, **kwargs):
        """Override to also move the encoder."""
        super().to(*args, **kwargs)
        self._encoder.to(*args, **kwargs)
        return self
