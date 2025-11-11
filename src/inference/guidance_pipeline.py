"""
Guidance pipeline for cardiac ultrasound analysis.
Provides unified interface to run:
- 4-chamber landmark detection 
- probe rotation prediction
- pose classification (traffic light)
- lvef estimation
"""

import os
from dataclasses import dataclass
import inference.inference_helper as ihp
from inference.fourch_landmark.fourch_landmark_pipeline import (
    configure_fourch_landmark_pipeline,
    run_fourch_core_prediction
)
from inference.rotation.rotation_predictor import create_rotation_predictor
from inference.pose_classification import pose_classification
from inference.pose_classification.pose_classification import PoseClassifierResultPooler

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
        self.__device = ihp.get_device()
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


        return cla, landmarks_tensor, rotation_prediction, pose_classification_prediction