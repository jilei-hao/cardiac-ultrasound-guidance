import numpy as np
import cv2
import math
from .mesh_helper import load_stl_vertices, draw_stl_simplified

def draw_circle_on_image(center, radius, image, bgr):
    cv2.circle(image, center, radius, bgr, -1)
    return image

def draw_border_around_image(image, width, bgr):
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), bgr, width)
    return image

def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

def add_rgb_circle_to_result(input_frame, rgb):
    center = (90, 22); radius = 5
    return draw_circle_on_image(center, radius, input_frame, rgb)

def add_rgb_border_to_result(input_frame, rgb):
    width = 40
    return draw_border_around_image(input_frame, width, rgb)

def overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6, reference_size=112):
    """
    Overlay landmarks on the visualization frame (backward compatibility wrapper).
    
    For 4-chamber landmarks with labels, use FourChamberLandmarkRenderer directly.
    
    Args:
        vis_frame: Visualization frame (BGR format)
        landmarks_tensor: Tensor with landmark coordinates
        app_settings: Application settings
        ere_threshold: ERE threshold for filtering
        reference_size: Reference size for coordinate scaling (default 112 for backward compatibility)
                       If landmarks are already in vis_frame coordinates, set to None or vis_frame size
    """
    # Check if landmarks have label information (4 elements) - use new renderer
    has_labels = landmarks_tensor.shape[1] >= 4
    
    if has_labels:
        # Use FourChamberLandmarkRenderer for 4-chamber landmarks
        from .fourch_landmark_renderer import FourChamberLandmarkRenderer
        
        renderer = FourChamberLandmarkRenderer(
            landmark_radius=app_settings.landmark_radius,
            min_opacity=0.6,
            max_opacity=1.0,
            uncertainty_threshold=float(ere_threshold),
            reference_size=reference_size
        )
        return renderer.overlay_landmarks(vis_frame, landmarks_tensor, reference_size=reference_size)
    else:
        # Fallback to original ERE-based rendering for backward compatibility
        # Scale coordinates if landmarks are in reference_size coordinates
        if reference_size is not None:
            scale_x = vis_frame.shape[1] / reference_size
            scale_y = vis_frame.shape[0] / reference_size
        else:
            scale_x = 1.0
            scale_y = 1.0
        
        radius = app_settings.landmark_radius
        
        # Draw landmark points (legacy format)
        for c, lm in enumerate(landmarks_tensor):
            x, y, ere_val = lm[0], lm[1], lm[2]
            if c in [42, 45, 46]:
                ere_val = ere_val / 3
            if ere_val <= ere_threshold:
                # Convert ERE to opacity (lower ERE = higher opacity)
                ere_val_float = ere_val.item() if hasattr(ere_val, 'item') else float(ere_val)
                ere_normalized = ere_val_float / ere_threshold  # 0 to 1
                opacity = float(max(0.6, 1.0 - ere_normalized))
                color = app_settings.get_color_for_ere(ere_val_float, max_ere=ere_threshold)
                
                scaled_x = int(x.item() * scale_x)
                scaled_y = int(y.item() * scale_y)
                
                # Extract a small region around the landmark for efficient blending
                h, w = vis_frame.shape[:2]
                x1 = max(0, scaled_x - radius - 2)
                y1 = max(0, scaled_y - radius - 2)
                x2 = min(w, scaled_x + radius + 2)
                y2 = min(h, scaled_y + radius + 2)
                
                # Create small overlay for this landmark region
                region_original = vis_frame[y1:y2, x1:x2].copy()
                region_overlay = region_original.copy()
                
                # Draw circle relative to the region
                cv2.circle(region_overlay, (scaled_x - x1, scaled_y - y1), radius, color, -1)
                
                # Blend this region with calculated opacity
                region_blended = cv2.addWeighted(region_overlay, opacity, region_original, 1.0 - opacity, 0)
                vis_frame[y1:y2, x1:x2] = region_blended
    
    return vis_frame

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        return np.eye(3, dtype=float)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)
    return R

def draw_stl_model(input_frame, stl_path, position=(150, 150), size=80, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    return draw_stl_simplified(input_frame, stl_path, position, size, R, color=(180, 200, 220), thickness=2)
