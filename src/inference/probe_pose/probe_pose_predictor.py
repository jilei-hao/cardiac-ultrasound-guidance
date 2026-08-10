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

import contextlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import torch
import einops

from .usgr_model import PoseTransformer, Normalizer
from .filter import PoseFilter


# Default SE(3) ESKF noise params, tuned for smoothing the cm/few-degree jitter
# of the relative-transform prediction (the USGR defaults are far looser and
# drift on this scale). Lower sigma_vel + cm-scale measurement noise => ~3-7x
# jitter reduction with sub-mm drift on a roughly-stationary view.
DEFAULT_FILTER_KWARGS = dict(
    sigma_pose=0.01,
    sigma_vel=0.02,
    sigma_meas_rot=0.1,
    sigma_meas_trans=0.05,
    # No two-point velocity bootstrap: the filter is reset on every view
    # switch, exactly when raw predictions jitter most, and differencing two
    # noisy poses ~62 ms apart amplifies that noise (~2/dt^2) into a huge
    # twist that the low sigma_vel then clings to for seconds -- the "target
    # probe swings wildly after switching views" bug (worse when degenerate
    # frames coast on it, measurement-free). Velocity is instead learned
    # through updates (init P_vel=1.0 allows it); measured cost: <=1 deg
    # transient lag on a 30 deg/s moving target.
    bootstrap_velocity=False,
)


# View classes. Index == conditioning one-hot slot. The correct list depends on
# which dataset the checkpoint was trained on:
#   - SIMULATED (CanonicalViewDataset, e.g. demo convnext ema_9000/21000):
#     order from usgr/usgr/guidance/datasets.py.
#   - INVIVO (InVivoReferenceViewDataset, e.g. ema_11000): reference labels "1"-"6"
#     sort to indices 0-5; anatomy from usgr/publication/evaluations/
#     invivo_recording_analysis/server.py::VIEW_NAMES (confirmed with the USGR dev).
CANONICAL_VIEW_LIST = ["A4C", "A2C", "PSAX_PAP", "PSAX_MV", "PLAX", "SC4C"]
INVIVO_VIEW_LIST = ["PLAX", "PSAX", "A4C", "A2C", "SC4C", "SCIVC"]
# Both lists are length 6, so one-hot indexing / NUM_VIEWS is shared.
NUM_VIEWS = len(CANONICAL_VIEW_LIST)


def view_list_for(view_set: str) -> list:
    """Return the view-name list for ``view_set`` ('simulated' or 'invivo')."""
    vs = (view_set or "simulated").strip().lower()
    if vs == "invivo":
        return INVIVO_VIEW_LIST
    if vs == "simulated":
        return CANONICAL_VIEW_LIST
    raise ValueError(f"view_set must be 'simulated' or 'invivo', got {view_set!r}")

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
        valid:    True when the pose is usable: either the RAW predicted rotation
                  was sane (det > 0), or it was degenerate but the ESKF had state
                  to extrapolate from (matching USGR realtime_inference.py). False
                  only when the raw rotation was degenerate AND no filter state
                  exists -- consumers should skip the pose then (R_rel is still a
                  valid rotation matrix, but its value is meaningless).
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
        # USGR training default (DinoEncoderConfig.image_size = 512). Evaluating at
        # 384 degrades the head: green-A4C rotation error 47.5deg -> 27.8deg when
        # moving 384 -> 512 on ema_21000 (see scripts/diagnose_probe_pose.py).
        image_size: int = 512,
        conditioning_dim: int = CLASS_LENGTH,
        # L-R chirality: USGR training data is LV-on-left; CUSG frames are
        # LV-on-right (the mirror). The training h-flip augmentation flips the
        # image AND right-multiplies the source rotation by F=diag(1,-1,-1)
        # (180deg about probe x), so a *converged* model is flip-equivariant and
        # feeding the wrong chirality injects ~180deg-about-x error. ema_21000 is
        # still flip-INVARIANT (output changes <5deg under flip) because it is
        # undertrained, so this toggle is a no-op today -- leave OFF until a
        # converged checkpoint, then confirm with the USGR dev whether the
        # canonical poses want mirror-input-only or mirror-input + F-on-output.
        mirror_input: bool = False,
        use_filter: bool = True,
        filter_kwargs: Optional[dict] = None,
        # Run the forward under torch.autocast(cuda, fp16): ~2.25x faster
        # (32.9 -> 14.6 ms per ViT-L/16 @ 512 forward on a 4090) with no
        # meaningful pose change (fp16_parity_test.py, 2,090 recorded frames:
        # raw rotation delta p95 = 0.6 deg vs the model's ~5 deg converged
        # error; |dT| p95 = 0.3 mm; the ESKF smooths the residual jitter).
        # CUDA-only; ignored on CPU.
        autocast_fp16: bool = False,
        # How many CONSECUTIVE degenerate (det<=0) frames the filter may coast
        # through before the prediction is surfaced as valid=False.
        #
        # Without a limit the coast path below forces valid=True indefinitely, so
        # a model that has stopped producing sane rotations -- which is exactly
        # what happens off-view, since the head "collapses toward a degenerate
        # mean when unconfident" -- keeps the ghost probe on screen, extrapolating
        # on filter velocity with no measurement pulling it back. That renders a
        # confident-looking cue built from nothing. Coasting a few frames is still
        # right (it bridges isolated dropouts); coasting forever is not.
        #
        # 0 disables coasting entirely; None restores the old unbounded behaviour.
        max_coast_frames: Optional[int] = 5,
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
        self.mirror_input = mirror_input
        self.autocast_fp16 = autocast_fp16
        self.model = None
        self.normalizer = None

        # SE(3) ESKF smoothing of the predicted relative transform (reduces the
        # frame-to-frame jitter of the raw model output). The filter is reset
        # whenever the target view changes or time runs backwards.
        self.use_filter = use_filter
        self._filter_kwargs = filter_kwargs or dict(DEFAULT_FILTER_KWARGS)
        self._filter: Optional[PoseFilter] = None
        self._filter_view_idx: Optional[int] = None
        self._last_ts: Optional[float] = None
        self.max_coast_frames = max_coast_frames
        self._coast_frames = 0
        if self.use_filter:
            self._filter = PoseFilter(**self._filter_kwargs)

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

    def reset_filter(self) -> None:
        """Reset the SE(3) smoother (call when the pose sequence is discontinuous)."""
        self._filter = PoseFilter(**self._filter_kwargs) if self.use_filter else None
        self._filter_view_idx = None
        self._last_ts = None
        self._coast_frames = 0

    @property
    def coast_frames(self) -> int:
        """Consecutive degenerate frames the filter is currently coasting through.

        A direct read-out of how unconfident the head has been lately: the raw
        3x3 collapses toward a degenerate mean when the model has nothing to go
        on, so a nonzero run here means the cue is being extrapolated rather
        than measured. 0 whenever the last frame produced a usable rotation.
        """
        return self._coast_frames

    def set_filter_enabled(self, enabled: bool) -> None:
        """Turn ESKF smoothing on/off at runtime; resets filter state."""
        self.use_filter = bool(enabled)
        self.reset_filter()

    def _autocast_ctx(self):
        """fp16 autocast context when enabled (CUDA device only), else a no-op."""
        if self.autocast_fp16 and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def predict(self, frame: np.ndarray, view_idx: int = 0,
                timestamp: Optional[float] = None) -> Optional[ProbePosePrediction]:
        """Predict the relative transform from the current frame to ``view_idx``.

        Args:
            frame: grayscale [H, W] (uint8) or RGB/BGR [H, W, 3] ultrasound frame.
            view_idx: target standard view, 0-5 (see ``CANONICAL_VIEW_LIST``).
            timestamp: frame time in nanoseconds. Required for ESKF smoothing;
                when omitted (or filtering disabled) the raw per-frame prediction
                is returned.

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

            # Match the training chirality (LV-on-left) when enabled. See
            # mirror_input in __init__ for why this is OFF by default.
            if self.mirror_input:
                frame = cv2.flip(frame, 1)

            image_tensor = preprocess_image_for_model(frame).to(self.device)

            onehot = torch.zeros(self.conditioning_dim, dtype=torch.float32)
            onehot[view_idx] = 1.0
            onehot = onehot.unsqueeze(0).to(self.device)  # [1, conditioning_dim]

            with self._autocast_ctx():
                pose = self.model(image_tensor, onehot)           # [1, 12] normalized
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

            # SE(3) ESKF smoothing of the (proper) relative transform. Degenerate
            # raw rotations are never fed to the filter as measurements; instead
            # (matching USGR realtime_inference.py) we extrapolate from the filter
            # state, which gives a sane, continuous pose rather than the SVD
            # projection of garbage. Only when the filter has no state yet (or is
            # disabled) does an invalid frame surface as valid=False.
            if self.use_filter and self._filter is not None and timestamp is not None:
                view_changed = (self._filter_view_idx is not None
                                and view_idx != self._filter_view_idx)
                time_back = (self._last_ts is not None and timestamp < self._last_ts)
                if view_changed or time_back:
                    self._filter = PoseFilter(**self._filter_kwargs)
                    self._coast_frames = 0
                self._filter_view_idx = view_idx
                self._last_ts = timestamp
                if valid:
                    self._coast_frames = 0
                    filtered = self._filter(timestamp, np.concatenate([R_rel.flatten(), T_rel]))
                elif self._filter._ekf.initialized:
                    filtered = self._filter(timestamp)  # predict-only, no measurement
                    self._coast_frames += 1
                    # Bridge isolated degenerate frames, but stop vouching for the
                    # extrapolation once the model has been unconfident for a
                    # sustained run -- past that point the pose is pure filter
                    # momentum, not evidence. Callers hide the cue on valid=False.
                    valid = (self.max_coast_frames is None
                             or self._coast_frames <= self.max_coast_frames)
                else:
                    filtered = None
                if filtered is not None:
                    R_rel = _nearest_rotation(filtered[:9].reshape(3, 3))
                    T_rel = filtered[9:12]

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
