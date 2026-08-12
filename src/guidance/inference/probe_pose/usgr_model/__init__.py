"""Vendored USGR guidance model (PoseTransformer + frozen DINOv3 encoder).

This package is a minimal, self-contained copy of the model code needed to run
the USGR ``PoseTransformer`` forward pass inside cardiac-ultrasound-guidance.

Upstream: https://github.com/trevor-chan/usgr  (usgr/usgr/{guidance,encoder,layers})
Vendored files:
    model.py          <- usgr/guidance/model.py        (PoseTransformer)
    dino.py           <- usgr/encoder/dino.py           (DinoEncoder, MODEL_CONFIGS)
    transformer.py    <- usgr/layers/transformer.py     (TransformerHead + blocks)
    normalization.py  <- usgr/layers/normalization.py   (Normalizer)

Local modifications (see file headers):
    * dino.py / model.py: the DINOv3 backbone repo + weights are now passed in
      explicitly so we reuse the inference-only DINOv3 hub bundled with cardiac.
    * relative imports rewritten to this vendored package.
"""

from .model import PoseTransformer
from .dino import DinoEncoder
from .normalization import Normalizer
from .transformer import TransformerHead

__all__ = ["PoseTransformer", "DinoEncoder", "Normalizer", "TransformerHead"]
