"""Probe-pose prediction for standard-view guidance.

Wraps the vendored USGR ``PoseTransformer`` (frozen DINOv3 ConvNeXt-Base encoder +
trainable transformer head). Given a single ultrasound frame and a target standard
view (0-5), it predicts the body-centric relative SE(3) transform from the current
probe pose to that canonical view:

    pose = [R_rel (9, row-major 3x3), T_rel (3, metres)]

This is a strict superset of the existing rotation-only ``RotationPredictor``:
``R_rel`` plays the same role as that model's rotation correction, and ``T_rel``
adds translation guidance, both conditioned on the selected view.

The model code is vendored under ``usgr_model/`` (see its package docstring).
The DINOv3 backbone is loaded from the dinov3 hub already bundled at
``src/inference/dinov3/`` -- the same repo the rotation predictor uses.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import torch
import einops

from .usgr_model import PoseTransformer, Normalizer


# View classes (from usgr/usgr/guidance/datasets.py). Index == conditioning one-hot slot.
#   0=A4C, 1=A2C, 2=PSAX_PAP, 3=PSAX_MV, 4=PLAX, 5=SC4C
CANONICAL_VIEW_LIST = ["A4C", "A2C", "PSAX_PAP", "PSAX_MV", "PLAX", "SC4C"]
NUM_VIEWS = len(CANONICAL_VIEW_LIST)

# One-hot conditioning vector length the head was trained with (6 used + padding).
CLASS_LENGTH = 12


@dataclass
class ProbePosePrediction:
    """Predicted body-centric relative transform to a target standard view.

    Attributes:
        R_rel:    (3, 3) proper rotation (SO(3), det=+1) from current pose -> target
                  view, SVD-projected from the head's raw (unconstrained) output.
        T_rel:    (3,) translation in metres, in the probe's local frame.
        view_idx: index into ``CANONICAL_VIEW_LIST`` of the requested target view.
        valid:    False if the RAW predicted rotation was degenerate (det <= 0);
                  a coarse confidence flag (R_rel is always a valid rotation).
    """
    R_rel: np.ndarray
    T_rel: np.ndarray
    view_idx: int
    valid: bool

    @property
    def view_name(self) -> str:
        return CANONICAL_VIEW_LIST[self.view_idx]


def _nearest_rotation(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest proper rotation (SO(3)) via SVD.

    Returns ``U @ diag(1, 1, det(U Vt)) @ Vt`` which is the closest orthonormal
    matrix with det = +1 (right-handed). Robust to the non-orthonormal, possibly
    near-degenerate matrices the regression head emits.
    """
    U, _s, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    return (U @ np.diag([1.0, 1.0, d]) @ Vt).astype(R.dtype)


def preprocess_image_for_model(image: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Preprocess a grayscale frame for the PoseTransformer.

    Mirrors USGR ``recording/inference_simulation.py::preprocess_image_for_model``.

    Args:
        image: grayscale numpy array [H, W] with values 0-255.

    Returns:
        [1, 3, H, W] float tensor with values in [0, 1].
    """
    image_tensor = torch.from_numpy(image).to(dtype) / 255.0
    image_tensor = einops.repeat(image_tensor, "h w -> c h w", c=3)
    return image_tensor.unsqueeze(0)


class ProbePosePredictor:
    """Real-time probe-pose-to-standard-view prediction for ultrasound frames."""

    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str,
        dino_weights_path: str,
        device,
        dinov3_repo_dir: Optional[str] = None,
        dino_model: str = "dinov3_convnext_base",
        embed_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 16,
        image_size: int = 384,
        conditioning_dim: int = CLASS_LENGTH,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.norm_stats_path = norm_stats_path
        self.dino_weights_path = dino_weights_path
        self.dino_model = dino_model
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.image_size = image_size
        self.conditioning_dim = conditioning_dim
        self.model = None
        self.normalizer = None

        # Reuse the dinov3 hub bundled at src/inference/dinov3 (same as RotationPredictor).
        if dinov3_repo_dir is None:
            self.dinov3_repo_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "dinov3"
            )
        else:
            self.dinov3_repo_dir = dinov3_repo_dir

        self._load_model()

    def _load_model(self):
        """Instantiate the PoseTransformer, load the EMA head weights + normalizer."""
        try:
            print(f"🔧 [ProbePose] Loading DINOv3 backbone '{self.dino_model}' from: {self.dinov3_repo_dir}")
            print(f"🔧 [ProbePose] Backbone weights: {self.dino_weights_path}")
            model = PoseTransformer(
                dino_model=self.dino_model,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                use_conditioning_token=True,
                conditioning_dim=self.conditioning_dim,
                image_size=self.image_size,
                compile_encoder=False,
                device=self.device,
                dino_repo_path=self.dinov3_repo_dir,
                dino_weights_path=self.dino_weights_path,
            )

            print(f"🔧 [ProbePose] Loading head checkpoint: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            # Unwrap common training-checkpoint containers. The EMA demo file is
            # already a flat state_dict (keys "transformer_head.*"), so this is a
            # no-op for it; it only fires for wrapped checkpoints.
            if isinstance(state, dict):
                for _wrap in ("model", "state_dict", "ema", "model_ema"):
                    if _wrap in state and isinstance(state[_wrap], dict):
                        state = state[_wrap]
                        break
            # The EMA checkpoint is saved from a torch.compile'd model -> strip the
            # "_orig_mod." prefix so keys match the (uncompiled) module here.
            state = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in state.items()
            }
            # The frozen encoder is a non-registered attribute, so it is absent from
            # both this module's state_dict and the checkpoint -> strict=False, then
            # assert nothing unexpected slipped in.
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys in probe-pose checkpoint: {unexpected[:8]}")
            head_missing = [k for k in missing if k.startswith("transformer_head.")]
            if head_missing:
                raise RuntimeError(f"Missing head keys in probe-pose checkpoint: {head_missing[:8]}")
            print(f"✅ [ProbePose] Loaded head weights (missing(non-head)={len(missing)}, unexpected=0)")

            model.eval()
            model.to(self.device)
            self.model = model

            self.normalizer = Normalizer.load(self.norm_stats_path)
            print("✅ [ProbePose] Probe-pose predictor loaded successfully")

        except Exception as e:
            print(f"❌ [ProbePose] Failed to load probe-pose predictor: {e}")
            self.model = None
            self.normalizer = None

    @torch.no_grad()
    def predict(self, frame: np.ndarray, view_idx: int = 0) -> Optional[ProbePosePrediction]:
        """Predict the relative transform from the current frame to ``view_idx``.

        Args:
            frame: grayscale [H, W] (uint8) or RGB/BGR [H, W, 3] ultrasound frame.
            view_idx: target standard view, 0-5 (see ``CANONICAL_VIEW_LIST``).

        Returns:
            ``ProbePosePrediction`` or ``None`` if the model is unavailable.
        """
        if self.model is None or self.normalizer is None:
            return None
        if not (0 <= view_idx < NUM_VIEWS):
            raise ValueError(f"view_idx must be in [0, {NUM_VIEWS - 1}], got {view_idx}")

        try:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            image_tensor = preprocess_image_for_model(frame).to(self.device)

            onehot = torch.zeros(self.conditioning_dim, dtype=torch.float32)
            onehot[view_idx] = 1.0
            onehot = onehot.unsqueeze(0).to(self.device)  # [1, conditioning_dim]

            pose = self.model(image_tensor, onehot)               # [1, 12] normalized
            pose = self.normalizer.denormalize(pose)[0]            # [12] metric scale
            pose = pose.float().cpu().numpy()

            R_raw = pose[:9].reshape(3, 3)
            T_rel = pose[9:12]
            # The head regresses a flattened 3x3 with no orthogonality constraint,
            # so R_raw is generally not a proper rotation (it collapses toward a
            # degenerate mean when the model is unconfident). Project to the nearest
            # SO(3) matrix (det=+1) so the renderer's euler/quaternion math gets a
            # valid rotation. ``valid`` still reflects whether the RAW prediction was
            # a sane right-handed matrix (det>0) -- a coarse confidence signal.
            valid = bool(np.linalg.det(R_raw) > 0)
            R_rel = _nearest_rotation(R_raw)

            return ProbePosePrediction(R_rel=R_rel, T_rel=T_rel, view_idx=view_idx, valid=valid)

        except Exception as e:
            print(f"❌ [ProbePose] Prediction failed: {e}")
            return None


def create_probe_pose_predictor(
    checkpoint_path: str,
    norm_stats_path: str,
    dino_weights_path: str,
    device,
    **kwargs,
) -> ProbePosePredictor:
    """Factory mirroring ``create_rotation_predictor``."""
    return ProbePosePredictor(
        checkpoint_path=checkpoint_path,
        norm_stats_path=norm_stats_path,
        dino_weights_path=dino_weights_path,
        device=device,
        **kwargs,
    )
