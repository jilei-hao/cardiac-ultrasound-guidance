"""
Guidance pipeline for cardiac ultrasound analysis.
Provides unified interface to run:
- 4-chamber landmark detection 
- probe rotation prediction
- pose classification (traffic light)
- lvef estimation
"""


from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import os
import time

import cv2
import numpy as np
import torch
from .inference_helper import get_device
from .fourch_landmark.fourch_landmark_pipeline import (
    configure_fourch_landmark_pipeline,
    run_fourch_core_prediction
)
from .rotation.rotation_predictor import create_rotation_predictor
from .probe_pose import create_probe_pose_predictor, ProbePosePrediction
from .pose_classification import pose_classification
from .quality_score import (
    create_quality_score_predictor,
    QualityScorePooler,
)
from .traffic_light import score_to_border_rgb
from .lvef.lvef_prediction import LVEFPredictor

# The EchoNet LVEF model (r2plus1d_18) is trained on 112x112 frames. The LVEF
# buffer input is resized to this regardless of the caller's display resolution.
LVEF_INPUT_SIZE = 112

@dataclass
class GuidancePipelineConfig:
    fourch_landmark_weights: str
    rotation_prediction_config: str
    rotation_prediction_weights: str
    pose_classification_weights: str
    lvef_weights: str
    landmarks_per_chamber: dict
    # Local dinov3 repo dir for the 4-chamber landmark backbone. None lets
    # configure_fourch_landmark_pipeline fall back to its own default (e.g.
    # torch.hub); deployments with an offline dinov3 checkout pass it here.
    dinov3_repo_dir: Optional[str] = None
    # Optional probe-pose (USGR PoseTransformer) standard-view guidance.
    # When any of these is missing, the probe-pose stage is disabled and the
    # pipeline falls back to rotation-only guidance.
    probe_pose_weights: Optional[str] = None
    probe_pose_norm_stats: Optional[str] = None
    probe_pose_dino_weights: Optional[str] = None
    # Encoder variant + input size the checkpoint was trained with. The demo
    # checkpoints (ema_9000/21000) use convnext_base; the 0610 invivo run
    # (ema_11000) uses dinov3_vitl16. Both emit 1024-dim features, so a wrong
    # pairing fails loudly at load (mismatched ConvNeXt vs ViT keys).
    probe_pose_dino_model: str = "dinov3_convnext_base"
    probe_pose_image_size: int = 512
    # H-flip the input frame before inference to match the training chirality
    # (USGR trains LV-on-left). Clarius/CUSG frames are LV-on-right, so the
    # flip-equivariant ema_11000 needs this ON for live use (verified: recovers
    # a converged A4C prediction from 178deg to 4.6deg). Keep OFF when the input
    # is already LV-on-left (e.g. the USGR invivo reproduction data).
    probe_pose_mirror_input: bool = False

    # Optional USGR quality-score ("traffic light v2") classifier. This is an
    # ALTERNATIVE to the ResNet18 pose classifier above (pose_classification);
    # both stay loaded and the consumer toggles between them at runtime. The
    # stage is disabled (predictor not built) when quality_score_weights is
    # missing. Needs a DINOv3 backbone, so quality_score_dino_weights must also
    # be set (reuse the same dinov3_vitl16 file probe_pose uses).
    quality_score_weights: Optional[str] = None
    quality_score_dino_weights: Optional[str] = None
    # Encoder variant + input size the score checkpoint was trained with. The
    # delivered checkpoint reuses the probe_pose backbone (dinov3_vitl16 @ 512,
    # 1024-d, num_heads=16); override if the USGR config.yaml says otherwise.
    quality_score_dino_model: str = "dinov3_vitl16"
    quality_score_image_size: int = 512
    quality_score_num_heads: int = 16
    quality_score_mirror_input: bool = False
    # Traffic-light thresholds on the raw score (red < red_below <= yellow <
    # green_above <= green). Used only in DISCRETE border mode.
    quality_score_red_below: float = -1.0
    quality_score_green_above: float = 1.0
    quality_score_pool_size: int = 15
    quality_score_score_min: float = -3.0
    quality_score_score_max: float = 3.0

    # --- Shared traffic-light border scale (applies to BOTH sources) ---------
    # Each source maps its raw score to a 0..1 "quality fraction" via its own
    # [score_min, score_max] range; these shared PERCENTAGE cut-points then pick
    # the colour, and one continuous toggle switches discrete<->gradient for
    # whichever source is active. Defaults (1/3, 2/3) reproduce the previously
    # hard-coded per-model thresholds. See inference/traffic_light.py.
    traffic_light_continuous: bool = False
    traffic_light_red_below_frac: float = 1.0 / 3.0
    traffic_light_green_above_frac: float = 2.0 / 3.0
    # Legacy pose-classifier raw-score range used to normalize to the 0..1
    # scale. [-2, 1] places the default 1/3, 2/3 cut-points at the classifier's
    # historical -1 (red) / 0 (green) raw thresholds.
    pose_classification_score_min: float = -2.0
    pose_classification_score_max: float = 1.0
    pose_classification_pool_size: int = 15
    # Deprecated alias for traffic_light_continuous (kept so existing configs
    # that set the quality-only flag still work). Prefer traffic_light_continuous.
    quality_score_continuous: bool = False

@dataclass
class GuidancePipelineOutput:
    cla: np.ndarray
    landmarks_tensor: np.ndarray
    rotation_prediction: np.ndarray
    pose_classification_prediction_rgb: np.ndarray
    # Body-centric relative transform to the requested standard view (or None
    # when the probe-pose predictor is unavailable).
    probe_pose_prediction: Optional[ProbePosePrediction] = None
    # USGR quality-score ("traffic light v2") outputs. Populated only when the
    # quality classifier is loaded AND enabled; both None otherwise.
    quality_score: Optional[float] = None
    quality_rgb: Optional[tuple] = None


class GuidancePipeline:
    def __init__(self, config: GuidancePipelineConfig):
        self.__device = get_device()
        self.__landmark_pipeline = configure_fourch_landmark_pipeline(
            weight_path=config.fourch_landmark_weights,
            device=self.__device,
            landmarks_per_chamber=config.landmarks_per_chamber,
            dinov3_repo_dir=config.dinov3_repo_dir
        )
        self.__rotation_predictor = create_rotation_predictor(
            config_path=config.rotation_prediction_config,
            checkpoint_path=config.rotation_prediction_weights,
            device=self.__device
        )

        # Optional probe-pose (USGR PoseTransformer) predictor for standard-view
        # translation + rotation guidance. Enabled only when all three assets exist.
        self.__probe_pose_predictor = None
        _pp_paths = (
            config.probe_pose_weights,
            config.probe_pose_norm_stats,
            config.probe_pose_dino_weights,
        )
        if all(_pp_paths) and all(os.path.exists(p) for p in _pp_paths):
            self.__probe_pose_predictor = create_probe_pose_predictor(
                checkpoint_path=config.probe_pose_weights,
                norm_stats_path=config.probe_pose_norm_stats,
                dino_weights_path=config.probe_pose_dino_weights,
                device=self.__device,
                dino_model=config.probe_pose_dino_model,
                image_size=config.probe_pose_image_size,
                mirror_input=config.probe_pose_mirror_input,
            )
        else:
            print("[GuidancePipeline] Probe-pose assets not found; standard-view "
                  "guidance disabled (rotation-only fallback).")

        _model, _preprocess = pose_classification.load_pretrained_model(
            self.__device, weight_path=config.pose_classification_weights
        )
        self.__pose_classifier_model = _model
        self.__pose_classifier_preprocess = _preprocess
        # Continuous-score pooler for the legacy classifier, mirroring the
        # quality pooler: average the raw score over a window, then map the
        # smoothed value through the shared traffic_light scale (discrete or
        # continuous). This replaces the old majority-vote-on-colours pooling so
        # the legacy border honours the same percentage cut-points + continuous
        # toggle as the quality border.
        self.__pose_score_pooler = QualityScorePooler(
            pool_size=config.pose_classification_pool_size)
        self.__pose_score_min = config.pose_classification_score_min
        self.__pose_score_max = config.pose_classification_score_max

        # Optional USGR quality-score ("traffic light v2") classifier. Built only
        # when both the head checkpoint and a DINOv3 backbone weights file are
        # configured and exist (it loads a vitl16 backbone, so we don't pay that
        # cost unless the user opts in). Loaded but inert until enabled via
        # set_quality_score_enabled(True) -- the consumer toggles it at runtime.
        self.__quality_score_predictor = None
        self.__quality_score_enabled = False
        self.__quality_score_pooler = QualityScorePooler(
            pool_size=config.quality_score_pool_size
        )
        self.__quality_score_red_below = config.quality_score_red_below
        self.__quality_score_green_above = config.quality_score_green_above
        self.__quality_score_score_min = config.quality_score_score_min
        self.__quality_score_score_max = config.quality_score_score_max
        # Shared continuous-gradient toggle + percentage cut-points, applied to
        # BOTH the legacy and quality borders through inference/traffic_light.
        # Display-only; toggled at runtime via set_traffic_light_continuous().
        self.__traffic_light_continuous = bool(
            config.traffic_light_continuous or config.quality_score_continuous)
        self.__tl_red_below_frac = config.traffic_light_red_below_frac
        self.__tl_green_above_frac = config.traffic_light_green_above_frac
        _qs_paths = (config.quality_score_weights, config.quality_score_dino_weights)
        if all(_qs_paths) and all(os.path.exists(p) for p in _qs_paths):
            self.__quality_score_predictor = create_quality_score_predictor(
                checkpoint_path=config.quality_score_weights,
                dino_weights_path=config.quality_score_dino_weights,
                device=self.__device,
                dino_model=config.quality_score_dino_model,
                image_size=config.quality_score_image_size,
                num_heads=config.quality_score_num_heads,
                red_below=config.quality_score_red_below,
                green_above=config.quality_score_green_above,
                mirror_input=config.quality_score_mirror_input,
            )
        else:
            print("[GuidancePipeline] Quality-score assets not found; the "
                  "'traffic light v2' classifier is unavailable (legacy "
                  "pose-classifier only).")

        self.__lvef_predictor = LVEFPredictor(device=self.__device, weight_path=config.lvef_weights)
        self.__executor = ThreadPoolExecutor(max_workers=2)
        self.__lvef_future = None
        # Per-stage wall-clock for the last run_inference call, in ms.
        # Populated only when timing_enabled is True.
        self.timing_enabled = False
        self.last_timings: dict = {}

        # Wall-clock meter for the actual rate (Hz) at which frames are consumed
        # into the LVEF buffer, updated every classify_and_buffer call and
        # averaged over a ~1s window. This is the TRUE buffer fill cadence — if
        # it sits below the publish rate, frames are being dropped upstream
        # before reaching the buffer (which temporally subsamples LVEF input).
        self.__lvef_buffer_fps = 0.0
        self.__lvef_buffer_frame_count = 0
        self.__lvef_buffer_fps_t0 = None
        self.__lvef_buffer_fps_window_s = 1.0


    def _sync_for_timing(self):
        if self.timing_enabled and self.__device.type == "cuda":
            torch.cuda.synchronize()

    @property
    def probe_pose_available(self) -> bool:
        """True when the optional standard-view probe-pose predictor is loaded."""
        return self.__probe_pose_predictor is not None and self.__probe_pose_predictor.model is not None

    @property
    def probe_pose_filter_enabled(self) -> bool:
        """True when ESKF smoothing of the probe-pose prediction is on."""
        return self.__probe_pose_predictor is not None and self.__probe_pose_predictor.use_filter

    def set_probe_pose_filter(self, enabled: bool) -> bool:
        """Enable/disable ESKF smoothing of the probe-pose prediction. Returns new state."""
        if self.__probe_pose_predictor is not None:
            self.__probe_pose_predictor.set_filter_enabled(enabled)
        return self.probe_pose_filter_enabled

    @property
    def quality_score_available(self) -> bool:
        """True when the optional USGR quality-score classifier is loaded."""
        return (self.__quality_score_predictor is not None
                and self.__quality_score_predictor.model is not None)

    @property
    def quality_score_enabled(self) -> bool:
        """True when the quality-score classifier is loaded AND switched on."""
        return self.quality_score_available and self.__quality_score_enabled

    def set_quality_score_enabled(self, enabled: bool) -> bool:
        """Turn the quality-score classifier on/off. Returns the effective state.

        Returns False (and stays off) if the predictor is unavailable, so the
        caller can fall back to the legacy classifier. Clears the smoothing
        window on every transition so a stale colour doesn't linger.
        """
        if not self.quality_score_available:
            self.__quality_score_enabled = False
            return False
        self.__quality_score_enabled = bool(enabled)
        self.__quality_score_pooler.clear()
        return self.__quality_score_enabled

    @property
    def traffic_light_continuous(self) -> bool:
        """True when the border (either source) renders as a continuous gradient."""
        return self.__traffic_light_continuous

    def set_traffic_light_continuous(self, enabled: bool) -> bool:
        """Switch BOTH borders between a continuous red->yellow->green gradient
        and the discrete three-colour traffic light.

        Source-agnostic and display-only (does not touch any predictor or
        smoothing window), so it can be set regardless of which traffic-light
        source is active or whether the quality model is loaded. Returns the
        effective state.
        """
        self.__traffic_light_continuous = bool(enabled)
        return self.__traffic_light_continuous

    # Backwards-compatible aliases -- the continuous toggle is now source-agnostic
    # (it used to be quality-only). Prefer the traffic_light_* names above.
    @property
    def quality_score_continuous(self) -> bool:
        """Deprecated alias for ``traffic_light_continuous``."""
        return self.__traffic_light_continuous

    def set_quality_score_continuous(self, enabled: bool) -> bool:
        """Deprecated alias for ``set_traffic_light_continuous``."""
        return self.set_traffic_light_continuous(enabled)

    def classify_quality(self, frame):
        """Run the USGR quality-score classifier on a single grayscale frame.

        Returns the raw ``QualityResult`` (continuous score + traffic-light
        rgb), or None when the classifier is unavailable. Updates the smoothing
        window as a side effect; the smoothed colour is available via
        ``get_pooled_quality_rgb()``.
        """
        if not self.quality_score_available:
            return None
        result = self.__quality_score_predictor.predict(frame)
        if result is not None:
            self.__quality_score_pooler.add_score(result.score)
        return result

    def get_pooled_quality_rgb(self):
        """Smoothed (windowed-mean-score) quality border colour, or None.

        Maps the pooled score to either a continuous red->yellow->green
        gradient (when continuous mode is on) or the discrete three-colour
        traffic light (default).
        """
        pooled_score = self.__quality_score_pooler.get_pooled_score()
        if pooled_score is None:
            return None
        return score_to_border_rgb(
            pooled_score,
            self.__quality_score_score_min,
            self.__quality_score_score_max,
            red_below_frac=self.__tl_red_below_frac,
            green_above_frac=self.__tl_green_above_frac,
            continuous=self.__traffic_light_continuous,
        )

    def classify(self, frame):
        """Run the traffic-light pose classifier on a single grayscale frame.

        The deployed classifier is image-only, so this does NOT depend on
        landmark detection and can run before the rest of the pipeline.

        Returns:
            PoseResult: continuous ``score`` + discretized ``rgb`` colour.
        """
        return pose_classification.run_pose_classifier(
            self.__pose_classifier_model,
            self.__device,
            frame,
            None,
            self.__pose_classifier_preprocess
        )

    def classify_and_buffer(self, frame):
        """Fast path: classify the frame and push it to the LVEF buffer.

        Intentionally minimal so callers can run it on the critical per-frame
        path without waiting for landmark/rotation/probe-pose. The RAW
        per-frame PoseResult (score + rgb) is stored in the buffer; the
        smoothed (pooled) colour for display is updated as a side effect and
        is available via ``get_pooled_pose_rgb()``.

        Returns:
            PoseResult: the raw per-frame classification.
        """
        pose_result = self.classify(frame)
        # Smooth the raw score for display/border stability; the colour is
        # mapped from the pooled score in get_pooled_pose_rgb (shared scale).
        self.__pose_score_pooler.add_score(pose_result.score)
        # The EchoNet LVEF model (r2plus1d_18) was trained on 112x112. Callers
        # pass the display/landmark-resolution frame (512x512 in the vigil
        # cardiac node); feeding 512 to the model collapses LVEF to a near-
        # constant (~48), destroying per-clip discrimination. Resize to 112 for
        # the LVEF buffer only (classification/display keep their own sizes).
        lvef_frame = cv2.resize(frame, (LVEF_INPUT_SIZE, LVEF_INPUT_SIZE),
                                interpolation=cv2.INTER_AREA)
        # Buffer keeps the RAW per-frame result so the LVEF model and any
        # diagnostics see exactly what was classified each frame.
        frame_rgb = cv2.cvtColor(lvef_frame, cv2.COLOR_GRAY2RGB)
        self.__lvef_predictor.add_frame_to_buffer(frame_rgb, pose_result)
        self.__update_lvef_buffer_fps()
        return pose_result

    def __update_lvef_buffer_fps(self):
        """Update the windowed wall-clock LVEF buffer-consume rate."""
        now = time.perf_counter()
        if self.__lvef_buffer_fps_t0 is None:
            self.__lvef_buffer_fps_t0 = now
        self.__lvef_buffer_frame_count += 1
        elapsed = now - self.__lvef_buffer_fps_t0
        if elapsed >= self.__lvef_buffer_fps_window_s:
            self.__lvef_buffer_fps = self.__lvef_buffer_frame_count / elapsed
            self.__lvef_buffer_frame_count = 0
            self.__lvef_buffer_fps_t0 = now

    def get_lvef_buffer_fps(self) -> float:
        """Actual rate (Hz) frames are being consumed into the LVEF buffer,
        averaged over ~1s. Below the publish rate => frames dropped upstream."""
        return self.__lvef_buffer_fps

    def get_pooled_pose_rgb(self):
        """Smoothed legacy traffic-light colour, or None if no frames yet.

        Pools the classifier's continuous score and maps it through the shared
        traffic_light scale, so the legacy border honours the same percentage
        cut-points and discrete<->continuous toggle as the quality border.
        """
        pooled_score = self.__pose_score_pooler.get_pooled_score()
        if pooled_score is None:
            return None
        return score_to_border_rgb(
            pooled_score,
            self.__pose_score_min,
            self.__pose_score_max,
            red_below_frac=self.__tl_red_below_frac,
            green_above_frac=self.__tl_green_above_frac,
            continuous=self.__traffic_light_continuous,
        )

    def predict_landmarks(self, frame):
        """4-chamber landmark detection. Returns (cla, landmarks_tensor)."""
        return run_fourch_core_prediction(self.__landmark_pipeline, frame)

    def predict_rotation(self, frame):
        """Probe rotation prediction, or None if the predictor is unavailable."""
        if self.__rotation_predictor is None:
            return None
        return self.__rotation_predictor.predict_rotation(frame)

    def predict_probe_pose(self, frame, target_view_index: int = 0, timestamp=None):
        """Standard-view probe-pose prediction, or None if unavailable."""
        if self.__probe_pose_predictor is None:
            return None
        return self.__probe_pose_predictor.predict(frame, target_view_index, timestamp)

    def run_display(self, frame, target_view_index: int = 0, timestamp=None):
        """Heavy path: landmark + rotation + probe-pose for display/guidance.

        Deliberately excludes pose classification and the LVEF buffer push
        (handled by ``classify_and_buffer``) so it can run off the critical
        per-frame path. Callers that distribute these stages across their own
        threads (e.g. the vigil cardiac node, which runs rotation on the IMU
        thread) can call the granular predict_* methods directly instead.

        Returns:
            tuple: (cla, landmarks_tensor, rotation_prediction, probe_pose_prediction)
        """
        cla, landmarks_tensor = self.predict_landmarks(frame)
        rotation_prediction = self.predict_rotation(frame)
        probe_pose_prediction = self.predict_probe_pose(frame, target_view_index, timestamp)
        return cla, landmarks_tensor, rotation_prediction, probe_pose_prediction

    # --- LVEF pass-throughs (so external orchestrators need not reach into the
    #     predictor or its buffer directly) ---
    def is_lvef_buffer_good(self) -> bool:
        """True when the LVEF buffer is full and clean enough to predict on."""
        return self.__lvef_predictor.is_buffer_video_good()

    def run_lvef_prediction(self):
        """Run a synchronous LVEF prediction over the current buffer."""
        return self.__lvef_predictor.run_prediction()

    def get_lvef_buffer_classifications(self):
        """Per-frame PoseResults currently in the LVEF buffer (diagnostics)."""
        return self.__lvef_predictor.get_buffer_classifications()

    def run_inference(self, frame, target_view_index: int = 0, timestamp=None):
        """Full per-frame pipeline (synchronous convenience).

        Runs the fast classify->buffer step FIRST so the LVEF buffer is fed
        before the heavier landmark/rotation/probe-pose work, then runs the
        display pipeline. Callers that need the two paths decoupled across
        threads should call ``classify_and_buffer()`` and ``run_display()``
        directly (see the vigil cardiac node).
        """
        timing_on = self.timing_enabled
        if timing_on:
            self._sync_for_timing()
            t0 = time.perf_counter()

        # Fast path: traffic-light classify + LVEF buffer push (before the rest).
        self.classify_and_buffer(frame)
        if timing_on:
            self._sync_for_timing()
            t1 = time.perf_counter()

        # Optional USGR quality classifier ("traffic light v2"). Skipped unless
        # enabled (it runs a DINOv3 backbone, so we don't pay for it otherwise).
        quality_score = None
        quality_rgb = None
        if self.quality_score_enabled:
            quality_result = self.classify_quality(frame)
            if quality_result is not None:
                quality_score = quality_result.score
                quality_rgb = self.get_pooled_quality_rgb()
        if timing_on:
            self._sync_for_timing()
            t_quality = time.perf_counter()

        # Heavy path: landmark + rotation + probe-pose.
        cla, landmarks_tensor, rotation_prediction, probe_pose_prediction = \
            self.run_display(frame, target_view_index, timestamp)
        if timing_on:
            self._sync_for_timing()
            t2 = time.perf_counter()
            self.last_timings = {
                "classify_buf": (t1 - t0) * 1e3,
                "quality": (t_quality - t1) * 1e3,
                "display": (t2 - t_quality) * 1e3,
            }

        return GuidancePipelineOutput(
            cla=cla,
            landmarks_tensor=landmarks_tensor,
            rotation_prediction=rotation_prediction,
            pose_classification_prediction_rgb=self.get_pooled_pose_rgb(),
            probe_pose_prediction=probe_pose_prediction,
            quality_score=quality_score,
            quality_rgb=quality_rgb,
        )


    def run_lvef_async(self) -> bool:
        """Submit LVEF prediction to thread pool. Returns True if submitted."""
        if not self.__lvef_predictor.is_buffer_video_good():
            return False
        if self.__lvef_future and not self.__lvef_future.done():
            return False
        self.__lvef_future = self.__executor.submit(self.__lvef_predictor.run_prediction)
        return True


    def get_latest_lvef(self):
        """Get result if ready."""
        if self.__lvef_future and self.__lvef_future.done():
            try:
                return self.__lvef_future.result()
            except Exception as e:
                print(f"[GuidancePipeline] LVEF error: {e}")
        return None