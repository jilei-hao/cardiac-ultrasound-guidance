"""Pose-classification helpers used only by the CUSG desktop app.

vigil-system's vendored core dropped these: its runtime pools quality scores
through GuidancePipeline instead. Keeping them here lets the core stay identical
to the vendored copy while legacy_main.py and research/ keep working.

Core symbols are re-exported so existing call sites are unaffected.
"""
import numpy as np

from guidance.inference.pose_classification.pose_classification import (  # noqa: F401
    PoseResult,
    load_pretrained_model,
    num_to_rgb,
    run_pose_classifier,
)



def plot_landmarks_as_images(landmarks, image_size=(112, 112), use_less_landmarks=True):
    device = landmarks.device
    
    # Determine image size based on landmark count
    # 48 landmarks = 4-chamber pipeline, 47 landmarks = old pipeline (112x112)
    if len(landmarks) == 48:
        # 4-chamber landmarks can be in any coordinate space (crop_frame size)
        # Detect the original coordinate space from landmark coordinates and scale to 112x112
        landmarks = landmarks.clone()
        max_coord = max(landmarks[:, 0].max().item(), landmarks[:, 1].max().item())
        
        # Scale landmarks to 112x112 if they're in a different coordinate space
        if max_coord > 112:
            scale_factor = 112 / max_coord
            landmarks[:, 0] = landmarks[:, 0] * scale_factor  # Scale X coordinates
            landmarks[:, 1] = landmarks[:, 1] * scale_factor  # Scale Y coordinates
        elif max_coord < 112 and max_coord > 0:
            # Scale up if landmarks are in smaller coordinate space
            scale_factor = 112 / max_coord
            landmarks[:, 0] = landmarks[:, 0] * scale_factor
            landmarks[:, 1] = landmarks[:, 1] * scale_factor
        
        image_size = (112, 112)
        # For 4-chamber landmarks, use a subset of landmarks for pose classification
        # Select landmarks from each chamber (first 3 landmarks from each chamber)
        landmarks_ind = [0, 1, 2, 12, 13, 14, 24, 25, 26, 36, 37, 38]  # 12 landmarks total
        landmarks = landmarks[landmarks_ind]
    else:
        # Old pipeline landmarks (already in 112x112 space)
        if use_less_landmarks:
            landmarks_ind = [42, 43, 44, 45, 46, 0, 40, 41, 20, 25, 10, 30, 11, 31]
            landmarks = landmarks[landmarks_ind]
    
    image = torch.zeros(image_size, device=device)

    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    landmarks_error = landmarks[:, 2]

    # create coordinate grid
    x = torch.arange(image_size[1], device=device)
    y = torch.arange(image_size[0], device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')  # (W, H) -> same as numpy meshgrid

    for i in range(len(landmarks_x)):
        sigma = 3 / landmarks_error[i]
        blob = torch.exp(-((x - landmarks_x[i])**2 + (y - landmarks_y[i])**2) / (2 * sigma**2))
        image += blob

    # Normalize twice to [0, 1]
    image = image / torch.max(image)
    image = image.float()
    image = image / torch.max(image) 
    # rotate 90 degrees clockwise
    image = torch.rot90(image, -1, [0, 1])
    # flip horizontally
    image = torch.flip(image, [1])

    return image


def _regression_to_num_classes(regression_val):
    # print("regression_val: ", regression_val)
    if regression_val <= -1:
        return -1 # red
    elif regression_val <= 0:
        return 0 # yellow
    else:
        return 1 # green


class PoseClassifierResultPooler:
    def __init__(self, pool_size=15, min_size=1):
        self.pool_size = pool_size
        self.min_size = min_size
        self.results_pool = []

    def add_result(self, rgb_prediction):
        self.results_pool.append(rgb_prediction)
        # Maintain sliding window by removing oldest element if over pool_size
        if len(self.results_pool) > self.pool_size:
            self.results_pool.pop(0)

    def get_pooled_result(self):
        if not self.results_pool:
            return None
        
        # Only return pooled result if we have at least min_size elements
        if len(self.results_pool) <= self.min_size:
            return self.results_pool[-1]  # Return most recent result
        
        # Find the most common result in the pool
        most_common_result = max(set(self.results_pool), key=self.results_pool.count)
        return most_common_result
