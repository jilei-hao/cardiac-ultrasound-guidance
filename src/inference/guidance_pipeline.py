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
from .pose_classification.pose_classification import PoseClassifierResultPooler
from .lvef.lvef_prediction import LVEFPredictor

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

@dataclass
class GuidancePipelineOutput:
    cla: np.ndarray
    landmarks_tensor: np.ndarray
    rotation_prediction: np.ndarray
    pose_classification_prediction_rgb: np.ndarray
    # Body-centric relative transform to the requested standard view (or None
    # when the probe-pose predictor is unavailable).
    probe_pose_prediction: Optional[ProbePosePrediction] = None


class GuidancePipeline:
    def __init__(self, config: GuidancePipelineConfig):
        self.__device = get_device()
        self.__landmark_pipeline = configure_fourch_landmark_pipeline(
            weight_path=config.fourch_landmark_weights,
            device=self.__device,
            landmarks_per_chamber=config.landmarks_per_chamber
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
        self.__pose_classifier_result_pooler = PoseClassifierResultPooler(pool_size=15, min_size=1)
        self.__lvef_predictor = LVEFPredictor(device=self.__device, weight_path=config.lvef_weights)
        self.__executor = ThreadPoolExecutor(max_workers=2)
        self.__lvef_future = None
        # Per-stage wall-clock for the last run_inference call, in ms.
        # Populated only when timing_enabled is True.
        self.timing_enabled = False
        self.last_timings: dict = {}


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
        # Smooth the colour for display/border stability (pooler keyed on rgb).
        self.__pose_classifier_result_pooler.add_result(pose_result.rgb)
        # Buffer keeps the RAW per-frame result so the LVEF model and any
        # diagnostics see exactly what was classified each frame.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        self.__lvef_predictor.add_frame_to_buffer(frame_rgb, pose_result)
        return pose_result

    def get_pooled_pose_rgb(self):
        """Current smoothed (pooled) traffic-light colour, or None if empty."""
        return self.__pose_classifier_result_pooler.get_pooled_result()

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

        # Heavy path: landmark + rotation + probe-pose.
        cla, landmarks_tensor, rotation_prediction, probe_pose_prediction = \
            self.run_display(frame, target_view_index, timestamp)
        if timing_on:
            self._sync_for_timing()
            t2 = time.perf_counter()
            self.last_timings = {
                "classify_buf": (t1 - t0) * 1e3,
                "display": (t2 - t1) * 1e3,
            }

        return GuidancePipelineOutput(
            cla=cla,
            landmarks_tensor=landmarks_tensor,
            rotation_prediction=rotation_prediction,
            pose_classification_prediction_rgb=self.get_pooled_pose_rgb(),
            probe_pose_prediction=probe_pose_prediction
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