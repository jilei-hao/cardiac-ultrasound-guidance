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

# Optional legacy helper (kept simple, no extra transforms)
def draw_3axis_vector(input_frame, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    start_x, start_y = 100, 100
    axis_length = 60
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    x_rot = R @ x_axis
    y_rot = R @ y_axis
    z_rot = R @ z_axis

    alpha = 0.35
    def proj(v):
        px = start_x + int((v[0] + alpha * v[1]) * axis_length)
        py = start_y - int(v[2] * axis_length)
        return (px, py)

    origin = (start_x, start_y)
    x_end = proj(x_rot); y_end = proj(y_rot); z_end = proj(z_rot)

    cv2.arrowedLine(input_frame, origin, x_end, (200, 150, 100), 3)
    cv2.putText(input_frame, 'x', (x_end[0] + 6, x_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 100), 2)
    cv2.arrowedLine(input_frame, origin, y_end, (200, 180, 220), 3)
    cv2.putText(input_frame, 'y', (y_end[0] + 6, y_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 220), 2)
    cv2.arrowedLine(input_frame, origin, z_end, (150, 220, 200), 3)
    cv2.putText(input_frame, 'z', (z_end[0] + 6, z_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 200), 2)
    return input_frame

def draw_text_on_frame(input_frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = input_frame.shape[0] / 400
    font_color = (255, 255, 255)
    thickness = int(font_scale * 2)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = input_frame.shape[1] - text_size[0] - 20
    text_y = input_frame.shape[0] - text_size[1] - 20
    cv2.putText(input_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
    return input_frame

class STLAxisRenderer:
    """STL and axis renderer with pre-computed decimation; no coord swaps."""
    def __init__(self, stl_path, position=(100, 100), size=80):
        self.stl_path = stl_path
        self.position = position
        self.size = size
        self.vertices = None
        self.normalized_vertices = None
        self.lod_vertices = {}
        self.load_stl()
        self.precompute()

    def load_stl(self):
        try:
            self.vertices = load_stl_vertices(self.stl_path)
            print(f"✅ Loaded STL with {len(self.vertices)} vertices")
        except Exception as e:
            print(f"❌ Failed to load STL: {e}")
            self.vertices = None

    def precompute(self):
        if self.vertices is None:
            return
        self.normalized_vertices = self._normalize_vertices()
        self._precompute_lod_levels()
        print("✅ Pre-computed normalization and LOD levels")

    def _normalize_vertices(self):
        v = self.vertices.copy()
        center = (v.min(axis=0) + v.max(axis=0)) / 2.0
        v = v - center
        max_range = float(np.max(v.max(axis=0) - v.min(axis=0)))
        if max_range > 0:
            v = v / max_range
        return v

    def _precompute_lod_levels(self):
        if self.normalized_vertices is None:
            return
        for lod in [1.0, 0.5, 0.25, 0.1]:
            if lod == 1.0:
                self.lod_vertices[lod] = self.normalized_vertices.copy()
            else:
                num_vertices = max(1, int(len(self.normalized_vertices) * lod))
                step = max(1, len(self.normalized_vertices) // num_vertices)
                self.lod_vertices[lod] = self.normalized_vertices[::step]

    def get_lod_level(self, fps=None):
        if fps is not None:
            if fps < 5:   return 0.1
            if fps < 15:  return 0.25
            if fps < 20:  return 0.5
            return 1.0
        return 1.0

    def draw_combined(self, input_frame, qw=1.0, qx=0.0, qy=0.0, qz=0.0, fps=None, predicted_rotation_matrix=None):
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        input_frame = self.draw_3axis_vector(input_frame, R)
        if predicted_rotation_matrix is not None:
            input_frame = self.draw_3axis_vector_predicted(input_frame, predicted_rotation_matrix)
        if self.vertices is not None:
            input_frame = self.draw_stl_model_optimized(input_frame, R, fps)
        return input_frame

    def draw_3axis_vector(self, input_frame, R):
        start_x, start_y = self.position
        axis_length = self.size * 0.75

        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        x_rot = R @ x_axis
        y_rot = R @ y_axis
        z_rot = R @ z_axis

        alpha = 0.35
        def proj(v):
            px = start_x + int((v[0] + alpha * v[1]) * axis_length)
            py = start_y - int(v[2] * axis_length)
            return (px, py)

        origin = (start_x, start_y)
        x_end = proj(x_rot); y_end = proj(y_rot); z_end = proj(z_rot)

        cv2.line(input_frame, origin, x_end, (200, 150, 100), 3)
        cv2.putText(input_frame, 'x', (x_end[0] + 6, x_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 100), 2)
        cv2.line(input_frame, origin, y_end, (200, 180, 220), 3)
        cv2.putText(input_frame, 'y', (y_end[0] + 6, y_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 220), 2)
        cv2.line(input_frame, origin, z_end, (150, 220, 200), 3)
        cv2.putText(input_frame, 'z', (z_end[0] + 6, z_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 220, 200), 2)
        return input_frame

    def draw_3axis_vector_predicted(self, input_frame, R_pred):
        start_x, start_y = self.position
        axis_length = self.size * 0.75

        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        x_rot = R_pred @ x_axis
        y_rot = R_pred @ y_axis
        z_rot = R_pred @ z_axis

        alpha = 0.5
        def proj(v):
            px = start_x + int((v[0] + alpha * v[1]) * axis_length)
            py = start_y - int(v[2] * axis_length)
            return (px, py)

        origin = (start_x, start_y)
        x_end = proj(x_rot); y_end = proj(y_rot); z_end = proj(z_rot)

        overlay = input_frame.copy()
        self._draw_dashed_line(overlay, origin, x_end, (200, 150, 100), 3)
        self._draw_dashed_line(overlay, origin, y_end, (200, 180, 220), 3)
        self._draw_dashed_line(overlay, origin, z_end, (150, 220, 200), 3)
        cv2.addWeighted(overlay, 0.4, input_frame, 0.6, 0, input_frame)
        return input_frame

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness):
        x1, y1 = pt1; x2, y2 = pt2
        dx = x2 - x1; dy = y2 - y1
        distance = int(np.sqrt(dx*dx + dy*dy))
        if distance == 0:
            return
        dx /= distance; dy /= distance
        dash_length, gap_length = 8, 4
        current = 0
        while current < distance:
            start = current
            end = min(current + dash_length, distance)
            sx = int(x1 + start * dx); sy = int(y1 + start * dy)
            ex = int(x1 + end   * dx); ey = int(y1 + end   * dy)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            current += dash_length + gap_length

    def draw_stl_model_optimized(self, input_frame, R, fps=None):
        if self.normalized_vertices is None:
            return input_frame
        lod = self.get_lod_level(fps=fps)
        vertices = self.lod_vertices.get(lod, self.normalized_vertices)

        # No coord transform—just rotate the (normalized) model and project.
        v = vertices @ R.T

        projected = self._project_vertices_optimized(v)
        self._draw_triangles_optimized(input_frame, projected)
        return input_frame

    def _project_vertices_optimized(self, vertices):
        if len(vertices) == 0:
            return np.empty((0, 2), dtype=np.int32)
        alpha = 0.35
        px = vertices[:, 0] + alpha * vertices[:, 1]
        py = -vertices[:, 2]
        v2d = np.stack([px, py], axis=1) * float(self.size)
        v2d = v2d + np.asarray(self.position, dtype=np.float32)
        v2d = np.clip(v2d, 0, 10000)
        return v2d.astype(np.int32)

    def _draw_triangles_optimized(self, input_frame, projected):
        color = (180, 200, 220); thickness = 2
        for i in range(0, len(projected), 3):
            if i + 2 < len(projected):
                p0, p1, p2 = projected[i], projected[i+1], projected[i+2]
                cv2.line(input_frame, tuple(p0), tuple(p1), color, thickness)
                cv2.line(input_frame, tuple(p1), tuple(p2), color, thickness)
                cv2.line(input_frame, tuple(p2), tuple(p0), color, thickness)

    def draw_stl_model(self, input_frame, R):
        return self.draw_stl_model_optimized(input_frame, R, fps=None)

def draw_stl_model(input_frame, stl_path, position=(150, 150), size=80, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    return draw_stl_simplified(input_frame, stl_path, position, size, R, color=(180, 200, 220), thickness=2)
