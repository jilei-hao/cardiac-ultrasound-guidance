"""USGR image-quality scoring ("traffic light v2") for cardiac ultrasound.

Wraps the USGR ``ScoreTransformer`` -- a frozen DINOv3 encoder followed by the
SAME vendored ``TransformerHead`` used by ``probe_pose``, but with a single
scalar regression output instead of the 12-d pose vector. Given one ultrasound
frame it predicts a continuous image-quality score; the score is mapped to a
red/yellow/green traffic-light colour for the annotated-ultrasound border.

Relationship to the legacy traffic light
-----------------------------------------
This is an ALTERNATIVE to the ResNet18 classifier in
``inference/pose_classification`` (the original "traffic light"). Both stay
available at runtime; the guidance UI toggles between them with 'm'. The legacy
classifier is the default; this one is opt-in (it needs the DINOv3 backbone, so
it is only built when ``QUALITY_SCORE_WEIGHTS`` is configured) and only runs
when toggled on.

Model-code reuse
----------------
``ScoreTransformer`` is structurally ``probe_pose``'s ``PoseTransformer`` minus
the conditioning token:

    DinoEncoder (frozen) + TransformerHead(use_pose_token=True,
                                           use_conditioning_token=False,
                                           pose_out_channels=1)

so we import the already-vendored ``DinoEncoder``/``TransformerHead`` from
``inference.probe_pose.usgr_model`` rather than re-vendoring them. The reference
training/inference code is USGR ``usgr/quality/model.py::ScoreTransformer`` and
``usgr/quality/visualize_quality_scores.py`` (kept under
``projects/usgr_integration/`` in vigil-developer for reference).

Checkpoint
----------
``new_traffic_light_ema5000.pth`` (USGR, EMA @ 5000 steps). Keys are
``_orig_mod.transformer_head.*`` (saved from a ``torch.compile``'d model); the
frozen encoder is absent from the checkpoint and is loaded separately from the
DINOv3 backbone weights file (the same ``dinov3_vitl16`` file probe_pose uses).
"""

from collections import deque
from dataclasses import dataclass
import os
import contextlib
from typing import Optional

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from ..probe_pose.usgr_model.dino import DinoEncoder
from ..probe_pose.usgr_model.transformer import TransformerHead


# Traffic-light thresholds on the raw score. The USGR reference visualizer
# (visualize_quality_scores.py) renders a continuous red->yellow->green gradient
# over score in [-3, 3] with yellow centred at 0. We discretize to the same
# three-colour semantics the existing border uses; both bounds are configurable.
DEFAULT_RED_BELOW = -1.0
DEFAULT_GREEN_ABOVE = 1.0

# Score range over which the CONTINUOUS gradient spans red->yellow->green. The
# USGR reference visualizer (visualize_quality_scores.py) defaults to [-3, 3]
# with yellow centred at 0; we mirror that so the live border matches it.
DEFAULT_SCORE_MIN = -3.0
DEFAULT_SCORE_MAX = 3.0

# Continuous-gradient endpoints (RGB). Same three anchor colours the USGR
# reference visualizer uses (it expresses them in BGR); kept here in RGB to
# match the rest of this module. red @ score_min, yellow @ midpoint,
# green @ score_max.
_GRADIENT_LOW_RGB = (220, 55, 45)    # red
_GRADIENT_MID_RGB = (230, 210, 45)   # yellow
_GRADIENT_HIGH_RGB = (55, 210, 70)   # green


def score_to_traffic_rgb(
    score: float,
    red_below: float = DEFAULT_RED_BELOW,
    green_above: float = DEFAULT_GREEN_ABOVE,
) -> tuple:
    """Map a continuous quality score to a discrete traffic-light (R, G, B).

    score <  red_below            -> red    (poor view)
    red_below <= score < green_above -> yellow (marginal)
    score >= green_above          -> green  (good view)
    """
    if score < red_below:
        return (255, 0, 0)
    if score >= green_above:
        return (0, 255, 0)
    return (255, 255, 0)


def score_to_continuous_rgb(
    score: float,
    score_min: float = DEFAULT_SCORE_MIN,
    score_max: float = DEFAULT_SCORE_MAX,
) -> tuple:
    """Map a continuous quality score to a CONTINUOUS red->yellow->green (R, G, B).

    Unlike ``score_to_traffic_rgb`` (which quantizes to three colours), this
    linearly interpolates the border colour across the score range so it
    changes smoothly as view quality drifts. The score is clipped to
    ``[score_min, score_max]``; the lower half fades red->yellow and the upper
    half yellow->green, with yellow at the midpoint. Mirrors the USGR reference
    visualizer's ``score_color_bgr``.
    """
    span = score_max - score_min
    if span <= 0:
        # Degenerate range: fall back to the midpoint (yellow) colour.
        return _GRADIENT_MID_RGB
    t = (min(max(score, score_min), score_max) - score_min) / span

    if t < 0.5:
        u = t / 0.5
        lo, hi = _GRADIENT_LOW_RGB, _GRADIENT_MID_RGB
    else:
        u = (t - 0.5) / 0.5
        lo, hi = _GRADIENT_MID_RGB, _GRADIENT_HIGH_RGB
    return tuple(int(round(lo[i] * (1.0 - u) + hi[i] * u)) for i in range(3))


@dataclass(frozen=True)
class QualityResult:
    """One quality-score prediction.

    Attributes:
        score: continuous regression output from the ScoreTransformer.
        rgb:   discretized traffic-light colour as an (R, G, B) tuple.
    """
    score: float
    rgb: tuple


class ScoreTransformer(nn.Module):
    """DINOv3 encoder + transformer head that regresses a scalar quality score.

    Mirrors ``probe_pose.usgr_model.PoseTransformer`` (the encoder is held as a
    non-registered attribute so it is excluded from ``state_dict`` /
    ``parameters()`` -- the head is the only trained part) but configures the
    head with a single ``pose_out_channels`` and no conditioning token.
    """

    def __init__(
        self,
        dino_model: str = "dinov3_vitl16",
        embed_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 16,
        image_size: int = 512,
        compile_encoder: bool = False,
        device: str = "cuda",
        dino_repo_path: Optional[str] = None,
        dino_weights_path: Optional[str] = None,
    ):
        super().__init__()
        in_channels = DinoEncoder.MODEL_CONFIGS[dino_model]["embed_dim"]

        encoder_kwargs = dict(
            model_name=dino_model,
            image_size=image_size,
            compile_model=compile_encoder,
            device=device,
        )
        if dino_repo_path is not None:
            encoder_kwargs["repo_path"] = dino_repo_path
        if dino_weights_path is not None:
            encoder_kwargs["weights_path"] = dino_weights_path

        # Store the frozen encoder without registering it as a submodule.
        object.__setattr__(self, "_encoder", DinoEncoder(**encoder_kwargs))
        self.transformer_head = TransformerHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_pose_token=True,
            use_conditioning_token=False,
            pose_out_channels=1,
        )

    @property
    def encoder(self) -> DinoEncoder:
        """Access the frozen encoder."""
        return self._encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the [B, 1] quality score for a batch of frames."""
        x = self.encoder(x)
        # TransformerHead(use_pose_token=True) returns (spatial, pose_out); the
        # pose_out is the score-token projection -> [B, 1].
        _spatial, score = self.transformer_head(x)
        return score

    def to(self, *args, **kwargs):
        """Override to also move the (non-registered) encoder."""
        super().to(*args, **kwargs)
        self._encoder.to(*args, **kwargs)
        return self


def preprocess_frame_for_model(
    image: np.ndarray, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Preprocess a grayscale frame for the ScoreTransformer.

    The DINOv3 encoder resizes + ImageNet-normalizes internally, so we only need
    a [1, 3, H, W] float tensor in [0, 1]. A grayscale frame is broadcast across
    the 3 channels (mirrors probe_pose's ``preprocess_image_for_model``).

    Args:
        image: grayscale numpy array [H, W] with values 0-255.
    """
    image_tensor = torch.from_numpy(image).to(dtype) / 255.0
    image_tensor = einops.repeat(image_tensor, "h w -> c h w", c=3)
    return image_tensor.unsqueeze(0)


class QualityScorePooler:
    """Moving-average smoother for the raw quality score.

    The ResNet pose classifier pools discrete colours by majority vote
    (``PoseClassifierResultPooler``); the quality model emits a continuous
    score, so we average the score over a sliding window and map the smoothed
    value to a colour. This keeps the border stable without quantizing first.
    """

    def __init__(self, pool_size: int = 15):
        self.pool_size = max(1, pool_size)
        self._scores: deque = deque(maxlen=self.pool_size)

    def add_score(self, score: float) -> None:
        self._scores.append(float(score))

    def get_pooled_score(self) -> Optional[float]:
        if not self._scores:
            return None
        return sum(self._scores) / len(self._scores)

    def clear(self) -> None:
        self._scores.clear()


class QualityScorePredictor:
    """Real-time image-quality scoring for ultrasound frames."""

    def __init__(
        self,
        checkpoint_path: str,
        dino_weights_path: str,
        device,
        dinov3_repo_dir: Optional[str] = None,
        dino_model: str = "dinov3_vitl16",
        embed_dim: int = 1024,
        depth: int = 4,
        num_heads: int = 16,
        image_size: int = 512,
        red_below: float = DEFAULT_RED_BELOW,
        green_above: float = DEFAULT_GREEN_ABOVE,
        # USGR training data is LV-on-left; live Clarius/CUSG frames are
        # LV-on-right. The reference visualizer does no flipping, so this is OFF
        # by default -- enable only if the score is clearly inverted on live
        # data (see probe_pose's mirror_input note).
        mirror_input: bool = False,
        # Run the forward under torch.autocast(cuda, fp16): ~2.25x faster
        # (32.9 -> 14.6 ms per ViT-L/16 @ 512 forward on a 4090) with no
        # meaningful score change (fp16_parity_test.py, 2,090 recorded frames:
        # |dScore| p95 = 0.007 on a 2.0 threshold span, zero colour flips).
        # CUDA-only; ignored on CPU.
        autocast_fp16: bool = False,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.dino_weights_path = dino_weights_path
        self.dino_model = dino_model
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.image_size = image_size
        self.red_below = red_below
        self.green_above = green_above
        self.mirror_input = mirror_input
        self.autocast_fp16 = autocast_fp16
        self.model = None

        # Reuse the dinov3 hub bundled at src/inference/dinov3 (same as
        # ProbePosePredictor / RotationPredictor).
        if dinov3_repo_dir is None:
            self.dinov3_repo_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "dinov3"
            )
        else:
            self.dinov3_repo_dir = dinov3_repo_dir

        self._load_model()

    def _load_model(self):
        """Instantiate the ScoreTransformer and load the EMA head weights."""
        try:
            print(f"🔧 [Quality] Loading DINOv3 backbone '{self.dino_model}' from: {self.dinov3_repo_dir}")
            print(f"🔧 [Quality] Backbone weights: {self.dino_weights_path}")
            model = ScoreTransformer(
                dino_model=self.dino_model,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                image_size=self.image_size,
                compile_encoder=False,
                device=self.device,
                dino_repo_path=self.dinov3_repo_dir,
                dino_weights_path=self.dino_weights_path,
            )

            print(f"🔧 [Quality] Loading head checkpoint: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            # Unwrap common training-checkpoint containers (no-op for the flat
            # EMA file, which is already a bare state_dict).
            if isinstance(state, dict):
                for _wrap in ("model", "state_dict", "ema", "model_ema"):
                    if _wrap in state and isinstance(state[_wrap], dict):
                        state = state[_wrap]
                        break
            # The EMA checkpoint is saved from a torch.compile'd model -> strip
            # the "_orig_mod." prefix so keys match the uncompiled module here.
            state = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in state.items()
            }
            # The frozen encoder is a non-registered attribute, so it is absent
            # from both this module's state_dict and the checkpoint -> load
            # non-strict, then assert nothing unexpected slipped in and the head
            # is fully populated.
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys in quality checkpoint: {unexpected[:8]}")
            head_missing = [k for k in missing if k.startswith("transformer_head.")]
            if head_missing:
                raise RuntimeError(f"Missing head keys in quality checkpoint: {head_missing[:8]}")
            print(f"✅ [Quality] Loaded head weights (missing(non-head)={len(missing)}, unexpected=0)")

            model.eval()
            model.to(self.device)
            self.model = model
            print("✅ [Quality] Quality-score predictor loaded successfully")

        except Exception as e:
            print(f"❌ [Quality] Failed to load quality-score predictor: {e}")
            self.model = None

    def _autocast_ctx(self):
        """fp16 autocast context when enabled (CUDA device only), else a no-op."""
        if self.autocast_fp16 and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Optional[QualityResult]:
        """Predict the quality score for a single frame.

        Args:
            frame: grayscale [H, W] (uint8) or RGB/BGR [H, W, 3] ultrasound frame.

        Returns:
            ``QualityResult`` (score + traffic-light rgb), or ``None`` if the
            model is unavailable.
        """
        if self.model is None:
            return None
        try:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.mirror_input:
                frame = cv2.flip(frame, 1)

            image_tensor = preprocess_frame_for_model(frame).to(self.device)
            with self._autocast_ctx():
                out = self.model(image_tensor)
            score = float(out.detach().float().cpu().reshape(-1)[0])
            rgb = score_to_traffic_rgb(score, self.red_below, self.green_above)
            return QualityResult(score=score, rgb=rgb)

        except Exception as e:
            print(f"❌ [Quality] Prediction failed: {e}")
            return None


def create_quality_score_predictor(
    checkpoint_path: str,
    dino_weights_path: str,
    device,
    **kwargs,
) -> QualityScorePredictor:
    """Factory mirroring ``create_probe_pose_predictor``."""
    return QualityScorePredictor(
        checkpoint_path=checkpoint_path,
        dino_weights_path=dino_weights_path,
        device=device,
        **kwargs,
    )
