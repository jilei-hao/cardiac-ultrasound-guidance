import numpy as np
import cv2
import math
from .mesh_helper import draw_stl_simplified

def draw_circle_on_image(center, radius, image, bgr):
    # Draw a circle on the image
    cv2.circle(image, center, radius, bgr, -1)
    return image

def draw_border_around_image(image, width, bgr):
    # Draw a rectangle around the image
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), bgr, width)
    return image

def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

def add_rgb_circle_to_result(input_frame, rgb):
    center = (90, 22)
    radius = 5
    return draw_circle_on_image(center, radius, input_frame, rgb)

def add_rgb_border_to_result(input_frame, rgb):
    width = 40
    return draw_border_around_image(input_frame, width, rgb)


def overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6):
    scale_x = vis_frame.shape[1] / 112
    scale_y = vis_frame.shape[0] / 112

    overlay = vis_frame.copy()

    for c, lm in enumerate(landmarks_tensor):
        x, y, ere_val = lm
        if c in [42, 45, 46]:
            ere_val = ere_val / 3
        if ere_val <= ere_threshold:
            color = app_settings.get_color_for_ere(ere_val, max_ere=ere_threshold)
            scaled_x = int(x.item() * scale_x)
            scaled_y = int(y.item() * scale_y)
            cv2.circle(overlay, (scaled_x, scaled_y), app_settings.landmark_radius, color, -1)

    cv2.addWeighted(overlay, app_settings.landmark_alpha, vis_frame, 1 - app_settings.landmark_alpha, 0, vis_frame)

    return vis_frame

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        return np.eye(3)
    
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def draw_3axis_vector(input_frame, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """
    Draw a 3-axis vector using a right-handed convention:
      +X: right, +Z: up, +Y: into screen (shown as back-diagonal).
    """
    start_x, start_y = 100, 100
    axis_length = 60

    # Rotation from quaternion
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    # World axes (right-handed) - mirrored against YZ plane
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # # Apply -90 degree rotation around Z-axis to initial view
    # R_z_90 = np.array([
    #     [0, 1, 0],   # X -> Y
    #     [1, 0, 0],  # Y -> -X
    #     [0, 0, 1]    # Z -> Z
    # ], dtype=np.float32)
    
    # Apply X-axis rotation reversal to match STL model
    R_flipped = R.copy()
    # R_flipped[0, :] = -R_flipped[0, :]  # Flip X row
    # R_flipped[:, 0] = -R_flipped[:, 0]  # Flip X column
    
    # Combine rotations: first -90° around Z, then quaternion rotation
    R_combined = R_flipped
    
    # Rotate
    x_rot = R_combined @ x_axis
    y_rot = R_combined @ y_axis
    z_rot = R_combined @ z_axis

    # Simple axonometric projection: (x, y, z) -> (x + α*y, -z)
    alpha = 0.35  # controls how "diagonal" the Y arrow appears
    def proj(v):
        px = start_x + int((v[0] + alpha * v[1]) * axis_length)
        py = start_y - int(v[2] * axis_length)  # screen Y grows downward
        return (px, py)

    origin = (start_x, start_y)
    x_end = proj(x_rot)
    y_end = proj(y_rot)
    z_end = proj(z_rot)

    # Draw axes
    cv2.arrowedLine(input_frame, origin, x_end, (200, 150, 100), 3)
    cv2.putText(input_frame, 'x', (x_end[0] + 6, x_end[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 100), 2)

    cv2.arrowedLine(input_frame, origin, y_end, (200, 180, 220), 3)
    cv2.putText(input_frame, 'y', (y_end[0] + 6, y_end[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 220), 2)

    cv2.arrowedLine(input_frame, origin, z_end, (150, 220, 200), 3)
    cv2.putText(input_frame, 'z', (z_end[0] + 6, z_end[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 200), 2)

    return input_frame


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

def draw_stl_model(input_frame, stl_path, position=(150, 150), size=80, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """Draw STL model as object of the axis with quaternion rotation"""
    # Convert quaternion to full 3D rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Pass the rotation matrix directly - coordinate transformation is handled in mesh_helper.py
    return draw_stl_simplified(input_frame, stl_path, position, size, R, color=(180, 200, 220), thickness=2)
    