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

def get_color_for_chamber(label):
    """Get vibrant color for each chamber based on label value (BGR format for OpenCV)"""
    # More vibrant, easily distinguishable colors
    chamber_colors = {
        50: (255, 100, 180),      # Left Ventricle (LV) - Bright Pink/Magenta
        100: (200, 200, 50),      # Right Ventricle (RV) - Bright Yellow
        150: (100, 255, 150),     # Left Atrium (LA) - Bright Green
        200: (255, 150, 100),     # Right Atrium (RA) - Bright Orange
    }
    return chamber_colors.get(int(label), (255, 255, 255))  # White for unknown labels

def overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6, reference_size=112):
    """
    Overlay landmarks on the visualization frame with connecting lines.
    
    Args:
        vis_frame: Visualization frame (BGR format)
        landmarks_tensor: Tensor with landmark coordinates
        app_settings: Application settings
        ere_threshold: ERE threshold for filtering
        reference_size: Reference size for coordinate scaling (default 112 for backward compatibility)
                       If landmarks are already in vis_frame coordinates, set to None or vis_frame size
    """
    # Check if landmarks have label information (4 elements) or just (x, y, ere) format
    has_labels = landmarks_tensor.shape[1] >= 4
    
    # Scale coordinates if landmarks are in reference_size coordinates, otherwise use 1:1
    if reference_size is not None:
        scale_x = vis_frame.shape[1] / reference_size
        scale_y = vis_frame.shape[0] / reference_size
    else:
        scale_x = 1.0
        scale_y = 1.0
    
    radius = app_settings.landmark_radius
    line_thickness = max(1, radius // 2)  # Line thickness based on landmark radius
    
    # Group landmarks by chamber for drawing connecting lines
    if has_labels:
        landmarks_by_chamber = {}
        for c, lm in enumerate(landmarks_tensor):
            x, y, uncertainty, label = lm[0], lm[1], lm[2], lm[3]
            uncertainty_val = uncertainty.item() if hasattr(uncertainty, 'item') else float(uncertainty)
            
            if float(uncertainty_val) <= ere_threshold:
                label_int = int(label.item() if hasattr(label, 'item') else label)
                if label_int not in landmarks_by_chamber:
                    landmarks_by_chamber[label_int] = []
                scaled_x = int(x.item() * scale_x)
                scaled_y = int(y.item() * scale_y)
                uncertainty_float = float(uncertainty_val)
                landmarks_by_chamber[label_int].append((scaled_x, scaled_y, uncertainty_float))
        
        # Draw connecting lines for each chamber
        for label, chamber_landmarks in landmarks_by_chamber.items():
            if len(chamber_landmarks) < 2:
                continue
            
            color = get_color_for_chamber(label)
            chamber_overlay = vis_frame.copy()
            
            # Draw lines connecting consecutive landmarks
            for i in range(len(chamber_landmarks)):
                x1, y1, _ = chamber_landmarks[i]
                x2, y2, _ = chamber_landmarks[(i + 1) % len(chamber_landmarks)]  # Connect last to first
                
                # Draw line on overlay
                cv2.line(chamber_overlay, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)
            
            # Blend entire chamber overlay with average opacity
            avg_chamber_opacity = float(max(0.6, 1.0 - np.mean([unc for _, _, unc in chamber_landmarks])))
            cv2.addWeighted(chamber_overlay, avg_chamber_opacity, vis_frame, 1.0 - avg_chamber_opacity, 0, vis_frame)
    
    # Draw landmark points
    for c, lm in enumerate(landmarks_tensor):
        if has_labels:
            x, y, uncertainty, label = lm[0], lm[1], lm[2], lm[3]
            # Use chamber-specific vibrant color
            color = get_color_for_chamber(label)
            # Convert uncertainty to opacity: lower uncertainty = higher opacity
            # Uncertainty range: 0 (certain) to 1 (uncertain)
            # Opacity range: 0.6 (uncertain) to 1.0 (certain) - higher base opacity
            uncertainty_val = uncertainty.item() if hasattr(uncertainty, 'item') else float(uncertainty)
            opacity = float(max(0.6, 1.0 - uncertainty_val))  # Higher base opacity
        else:
            # Fallback to original ERE-based coloring for backward compatibility
            x, y, ere_val = lm[0], lm[1], lm[2]
            if c in [42, 45, 46]:
                ere_val = ere_val / 3
            if ere_val <= ere_threshold:
                # Convert ERE to opacity (lower ERE = higher opacity)
                ere_val_float = ere_val.item() if hasattr(ere_val, 'item') else float(ere_val)
                ere_normalized = ere_val_float / ere_threshold  # 0 to 1
                opacity = float(max(0.6, 1.0 - ere_normalized))  # Higher base opacity
                color = app_settings.get_color_for_ere(ere_val_float, max_ere=ere_threshold)
            else:
                continue  # Skip landmarks with high ERE
        
        # Only draw if uncertainty/ERE is below threshold
        uncertainty_val = lm[2].item() if has_labels else (ere_val.item() if hasattr(ere_val, 'item') else float(ere_val))
        if float(uncertainty_val) <= ere_threshold:
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
            
            # Blend this region with calculated opacity (higher base opacity)
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
