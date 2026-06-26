"""USGR image-quality ("traffic light v2") scoring module.

A DINOv3-backed alternative to the legacy ResNet18 pose classifier
(``inference/pose_classification``). Both produce a red/yellow/green
traffic-light signal for the annotated-ultrasound border; this one is the
USGR ``ScoreTransformer`` (frozen DINOv3 encoder + a scalar regression head)
and is toggled on/off at runtime from the guidance UI ('m'). Off by default.
"""

from .quality_score_predictor import (
    ScoreTransformer,
    QualityScorePredictor,
    QualityResult,
    QualityScorePooler,
    create_quality_score_predictor,
    score_to_traffic_rgb,
)

__all__ = [
    "ScoreTransformer",
    "QualityScorePredictor",
    "QualityResult",
    "QualityScorePooler",
    "create_quality_score_predictor",
    "score_to_traffic_rgb",
]
