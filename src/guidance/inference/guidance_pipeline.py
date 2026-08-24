"""
Guidance pipeline for cardiac ultrasound analysis.
Provides unified interface to run:
- 4-chamber landmark detection 
- probe rotation prediction
- pose classification (traffic light)
- lvef estimation
"""


from dataclasses import dataclass
from typing import Optional
import os
import time

import cv2
from .inference_helper import get_device
from .fourch_landmark.fourch_landmark_pipeline import (
    configure_fourch_landmark_pipeline,
    run_fourch_core_prediction
)
from .rotation.rotation_predictor import create_rotation_predictor
from .probe_pose import create_probe_pose_predictor
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
    # Optional probe-pose (USGR PoseTransformer) standard-view guidance.
    # When any of these is missing, the probe-pose stage is disabled and the
    # pipeline falls back to rotation-only guidance.
    probe_pose_weights: Optional[str] = None
    probe_pose_norm_stats: Optional[str] = None
    probe_pose_dino_weights: Optional[str] = None
    # Encoder variant and input size used by the deployed invivo checkpoint.
    probe_pose_dino_model: str = "dinov3_vitl16"
    probe_pose_image_size: int = 512
    # H-flip the input frame before inference to match the training chirality
    # (USGR trains LV-on-left). Clarius/CUSG frames are LV-on-right, so the
    # flip-equivariant ema_11000 needs this ON for live use (verified: recovers
    # a converged A4C prediction from 178deg to 4.6deg). Keep OFF when the input
    # is already LV-on-left (e.g. the USGR invivo reproduction data).
    probe_pose_mirror_input: bool = False

    # Optional USGR quality-score classifier. This is an alternative to the
    # ResNet18 pose classifier. The runtime supplies these paths, and therefore
    # builds this stage, only when the USGR traffic-light source is selected.
    # It reuses the same DINOv3 ViT-L backbone file as probe pose.
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
    quality_score_score_min: float = -3.0
    quality_score_score_max: float = 3.0

    # Run BOTH USGR predictors' forwards (probe_pose + quality) under
    # torch.autocast(cuda, fp16): ~2.25x faster (32.9 -> 14.6 ms per ViT-L/16
    # @ 512 forward on a 4090) with no meaningful output change — validated on
    # 2,090 recorded frames (projects/usgr_integration/fp16_parity_test.py in
    # vigil-developer): quality |dScore| p95 = 0.007 on a 2.0 threshold span
    # with zero colour flips; probe raw rotation delta p95 = 0.6 deg vs the
    # model's ~5 deg converged error. CUDA-only; ignored on CPU.
    usgr_autocast_fp16: bool = False

    # --- Shared traffic-light border scale (applies to BOTH sources) ---------
    # Each source maps its raw score to a 0..1 "quality fraction" via its own
    # [score_min, score_max] range; these shared PERCENTAGE cut-points then pick
    # the colour, and one continuous toggle switches discrete<->gradient for
    # whichever source is active. Defaults (1/3, 2/3) reproduce the previously
    # hard-coded per-model thresholds. See inference/traffic_light.py.
    traffic_light_continuous: bool = False
    traffic_light_red_below_frac: float = 1.0 / 3.0
    traffic_light_green_above_frac: float = 2.0 / 3.0
    # Unified border-smoothing window (frames) for whichever source is active:
    # the border colour is the mean of the last N per-frame scores (moving
    # average). One knob drives BOTH poolers, so it follows the active source.
    traffic_light_pool_size: int = 15
    # Legacy pose-classifier raw-score range used to normalize to the 0..1
    # scale. [-2, 1] places the default 1/3, 2/3 cut-points at the classifier's
    # historical -1 (red) / 0 (green) raw thresholds.
    pose_classification_score_min: float = -2.0
    pose_classification_score_max: float = 1.0


class GuidancePipeline:
    def __init__(self, config: GuidancePipelineConfig):
        self.__device = get_device()
        self.__landmark_pipeline = configure_fourch_landmark_pipeline(
            weight_path=config.fourch_landmark_weights,
            device=self.__device,
            landmarks_per_chamber=config.landmarks_per_chamber,
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
                autocast_fp16=config.usgr_autocast_fp16,
            )
        else:
            print("[GuidancePipeline] Probe-pose assets not found; standard-view "
                  "guidance disabled (rotation-only fallback).")

        self.__pose_classifier_model = pose_classification.load_pretrained_model(
            self.__device, weight_path=config.pose_classification_weights
        )
        # One smoothing window follows whichever traffic-light source is active.
        pool_size = config.traffic_light_pool_size
        # Continuous-score pooler for the legacy classifier, mirroring the
        # quality pooler: average the raw score over a window, then map the
        # smoothed value through the shared traffic_light scale (discrete or
        # continuous). This replaces the old majority-vote-on-colours pooling so
        # the legacy border honours the same percentage cut-points + continuous
        # toggle as the quality border.
        self.__pose_score_pooler = QualityScorePooler(pool_size=pool_size)
        self.__pose_score_min = config.pose_classification_score_min
        self.__pose_score_max = config.pose_classification_score_max

        # Optional USGR quality-score ("traffic light v2") classifier. Built only
        # when both the head checkpoint and a DINOv3 backbone weights file are
        # configured and exist (it loads a vitl16 backbone, so we don't pay that
        # cost unless the user opts in). Loaded but inert until enabled via
        # set_quality_score_enabled(True) -- the consumer toggles it at runtime.
        self.__quality_score_predictor = None
        self.__quality_score_enabled = False
        self.__quality_score_pooler = QualityScorePooler(pool_size=pool_size)
        self.__quality_score_red_below = config.quality_score_red_below
        self.__quality_score_green_above = config.quality_score_green_above
        self.__quality_score_score_min = config.quality_score_score_min
        self.__quality_score_score_max = config.quality_score_score_max
        # Shared continuous-gradient toggle + percentage cut-points, applied to
        # BOTH the legacy and quality borders through inference/traffic_light.
        self.__traffic_light_continuous = bool(config.traffic_light_continuous)
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
                autocast_fp16=config.usgr_autocast_fp16,
            )
        else:
            print("[GuidancePipeline] Quality-score assets not found; the "
                  "'traffic light v2' classifier is unavailable (legacy "
                  "pose-classifier only).")

        self.__lvef_predictor = LVEFPredictor(device=self.__device, weight_path=config.lvef_weights)

        # Wall-clock meter for the actual rate (Hz) at which frames are consumed
        # into the LVEF buffer, updated every classify_and_buffer call and
        # averaged over a ~1s window. This is the TRUE buffer fill cadence — if
        # it sits below the publish rate, frames are being dropped upstream
        # before reaching the buffer (which temporally subsamples LVEF input).
        self.__lvef_buffer_fps = 0.0
        self.__lvef_buffer_frame_count = 0
        self.__lvef_buffer_fps_t0 = None
        self.__lvef_buffer_fps_window_s = 1.0

    @property
    def probe_pose_available(self) -> bool:
        """True when the optional standard-view probe-pose predictor is loaded."""
        return self.__probe_pose_predictor is not None and self.__probe_pose_predictor.model is not None

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
        )

    def _buffer_frame(self, frame, pose_result):
        """Push (frame, per-frame result) to the LVEF buffer + bump the meter.

        Shared by the legacy and quality fast paths so the LVEF buffer is
        filled by exactly ONE traffic-light classification per frame, whichever
        source is active.

        The EchoNet LVEF model (r2plus1d_18) was trained on 112x112. Callers
        pass the display/landmark-resolution frame (512x512 in the vigil
        cardiac node); feeding 512 to the model collapses LVEF to a near-
        constant (~48), destroying per-clip discrimination. Resize to 112 for
        the LVEF buffer only (classification/display keep their own sizes). The
        buffer keeps the RAW per-frame result so the LVEF gate + diagnostics
        see exactly what was classified each frame.
        """
        lvef_frame = cv2.resize(frame, (LVEF_INPUT_SIZE, LVEF_INPUT_SIZE),
                                interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(lvef_frame, cv2.COLOR_GRAY2RGB)
        self.__lvef_predictor.add_frame_to_buffer(frame_rgb, pose_result)
        self.__update_lvef_buffer_fps()

    def classify_and_buffer(self, frame):
        """Fast path (legacy source): classify the frame and buffer it.

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
        self._buffer_frame(frame, pose_result)
        return pose_result

    def classify_quality_and_buffer(self, frame):
        """Fast path (USGR source): ONE quality forward feeds gate + border.

        Runs the USGR quality classifier a single time, pushes the frame + its
        per-frame result to the LVEF buffer (so the LVEF gate uses the quality
        traffic light), and updates the quality pooler (so the border reads the
        same run via ``get_pooled_quality_rgb`` / ``get_pooled_border_rgb``).
        This replaces the previous split where the legacy classifier fed the
        LVEF buffer while a separate quality forward drove only the border.

        Falls back to the legacy ``classify_and_buffer`` when the quality
        classifier is unavailable or off, so the LVEF buffer is always fed.

        Returns:
            PoseResult: the raw per-frame classification (quality-derived).
        """
        if not self.quality_score_enabled:
            return self.classify_and_buffer(frame)
        qres = self.__quality_score_predictor.predict(frame)
        if qres is None:
            return self.classify_and_buffer(frame)
        self.__quality_score_pooler.add_score(qres.score)
        # Store the quality result on the buffer's per-frame PoseResult. rgb is
        # the DISCRETE traffic-light colour (score_to_traffic_rgb, red_below/
        # green_above) that is.video_good counts; the pooled/continuous border
        # is derived separately from the pooled score.
        pose_result = pose_classification.PoseResult(score=qres.score, rgb=qres.rgb)
        self._buffer_frame(frame, pose_result)
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

    def get_pooled_border_rgb(self):
        """Pooled traffic-light border for the ACTIVE source.

        Single source of truth so callers don't branch on the traffic-light
        source: the quality pooler when the USGR classifier is enabled, else
        the legacy pose pooler. Both go through the shared traffic_light scale
        (same cut-points + discrete/continuous toggle).
        """
        if self.quality_score_enabled:
            return self.get_pooled_quality_rgb()
        return self.get_pooled_pose_rgb()

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

    def is_lvef_buffer_good(self) -> bool:
        """True when the LVEF buffer is full and clean enough to predict on."""
        return self.__lvef_predictor.is_buffer_video_good()

    def is_lvef_buffer_full(self) -> bool:
        """True when a complete LVEF clip has been collected."""
        return self.__lvef_predictor.is_buffer_full()

    def reset_lvef_buffer(self) -> None:
        """Start a fresh LVEF clip."""
        self.__lvef_predictor.reset_buffer()

    def get_lvef_capture_progress(self):
        """Return total and acceptable frame counts and their requirements."""
        return self.__lvef_predictor.capture_progress()

    def run_lvef_prediction(self):
        """Run a synchronous LVEF prediction over the current buffer."""
        return self.__lvef_predictor.run_prediction()
