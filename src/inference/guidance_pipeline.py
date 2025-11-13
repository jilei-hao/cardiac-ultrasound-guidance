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

import cv2
from .inference_helper import get_device
from .fourch_landmark.fourch_landmark_pipeline import (
    configure_fourch_landmark_pipeline,
    run_fourch_core_prediction
)
from .rotation.rotation_predictor import create_rotation_predictor
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

        _model, _preprocess = pose_classification.load_pretrained_model(
            self.__device, weight_path=config.pose_classification_weights
        )
        self.__pose_classifier_model = _model
        self.__pose_classifier_preprocess = _preprocess
        self.__pose_classifier_result_pooler = PoseClassifierResultPooler(pool_size=15, min_size=1)
        self.__lvef_predictor = LVEFPredictor(device=self.__device, weight_path=config.lvef_weights)
        self.__executor = ThreadPoolExecutor(max_workers=2)
        self.__lvef_future = None


    def run_inference(self, frame):
        # Run inference on the given frame using the landmark pipeline
        cla, landmarks_tensor = run_fourch_core_prediction(self.__landmark_pipeline, frame)

        # Run rotation prediction
        rotation_prediction = self.__rotation_predictor.predict_rotation(frame)
        
        # Run pose classification
        pose_classification_prediction = pose_classification.run_pose_classifier(
            self.__pose_classifier_model,
            self.__device,
            frame,
            landmarks_tensor,
            self.__pose_classifier_preprocess
        )
        self.__pose_classifier_result_pooler.add_result(pose_classification_prediction)
        pose_classification_prediction = self.__pose_classifier_result_pooler.get_pooled_result()

        # Add frame to LVEF predictor buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        self.__lvef_predictor.add_frame_to_buffer(frame_rgb, pose_classification_prediction)

        return cla, landmarks_tensor, rotation_prediction, pose_classification_prediction
    

    def run_lvef_async(self):
        if (self.__lvef_predictor.is_buffer_video_good() == False):
            # print("[GuidancePipeline] LVEF prediction skipped: insufficient good quality frames in buffer")
            return None
        
        """Submit LVEF prediction to thread pool."""
        if self.__lvef_future and not self.__lvef_future.done():
            return  # Still running
        
        self.__lvef_future = self.__executor.submit(self.__lvef_predictor.run_prediction)


    def get_latest_lvef(self):
        """Get result if ready."""
        if self.__lvef_future and self.__lvef_future.done():
            try:
                return self.__lvef_future.result()
            except Exception as e:
                print(f"[GuidancePipeline] LVEF error: {e}")
        return None