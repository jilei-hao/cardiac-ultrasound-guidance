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

def add_rgb_border_to_result(input_frame_bgr, rgb):
    width = 40
    bgr = rgb_to_bgr(rgb)
    return draw_border_around_image(input_frame_bgr, width, bgr)

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

def overlay_color_legend(input_frame, legend_bgra, top=10, left=10, pct_width=0.2, margin=20):
    """Overlay a BGRA color legend onto the input frame at specified position."""
    h, w = input_frame.shape[:2]
    lh, lw = legend_bgra.shape[:2]

    # resample the legend to fit the specified percentage of the input frame
    new_lw = int(w * pct_width)
    # compute new lh to maintain aspect ratio
    aspect_ratio = lw / lh
    new_lh = int(new_lw / aspect_ratio)
    legend_resampled = cv2.resize(legend_bgra, (new_lw, new_lh), interpolation=cv2.INTER_AREA)
    legend_bgra = legend_resampled
    lh, lw = legend_bgra.shape[:2]
    
    # Calculate position
    y1 = top + margin
    x1 = left + margin
    y2 = min(h, y1 + lh)
    x2 = min(w, x1 + lw)
    
    # Adjust legend size if it exceeds frame boundaries
    legend_crop = legend_bgra[0:(y2 - y1), 0:(x2 - x1)]
    
    # Split legend into BGR and Alpha channels
    legend_bgr = legend_crop[:, :, :3]
    alpha_mask = legend_crop[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
    

    # Overlay legend onto input frame
    for c in range(3):  # For B, G, R channels
        input_frame[y1:y2, x1:x2, c] = (
            alpha_mask * legend_bgr[:, :, c] +
            (1 - alpha_mask) * input_frame[y1:y2, x1:x2, c]
        ).astype(np.uint8)

    return input_frame



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


def draw_text_on_frame(input_frame, text):
    # Define the font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = input_frame.shape[0] / 400
    font_color = (255, 255, 255)  # White color
    thickness = int(font_scale * 2)

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position to the lower right corner
    text_x = input_frame.shape[1] - text_size[0] - 20  # 20 pixels from the right
    text_y = input_frame.shape[0] - text_size[1] - 20  # 20 pixels from the bottom

    # Draw the text on the frame
    cv2.putText(input_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    return input_frame