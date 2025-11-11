import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
from ..mesh_helper import load_stl_vertices
from .guidance_widget_renderer import GuidanceWidgetRenderer

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

class GuidanceWidgetRenderer2D(GuidanceWidgetRenderer):
    """Probe and axis renderer with configurable parameters."""
    
    def __init__(self, probe_path, 
                 background_color=(0, 0, 0),
                 predicted_axis_opacity=0.6, #0-1
                 axis_thickness=2,
                 axis_colors=((200, 150, 100), (200, 180, 220), (150, 220, 200)),
                 axis_labels=('x', 'y', 'z'),
                 lod_level=0.25, #mesh simplification level, 0-1, 1 = full detail
                 probe_scale=0.7, #probe size relative to axes, 0-1
                 rotation_amplification=3.0,  # Amplification factor for rotation prediction
                 render_mode='fullscreen',
                 window_size=(512, 512),
                 **kwargs):
        # Initialize parent class
        super().__init__(probe_path, **kwargs)
        
        # Store 2D-specific parameters
        self.background_color = background_color
        self.predicted_axis_opacity = predicted_axis_opacity
        self.axis_thickness = axis_thickness
        self.axis_colors = axis_colors
        self.axis_labels = axis_labels
        self.lod_level = lod_level
        self.probe_scale = probe_scale
        self.rotation_amplification = rotation_amplification
        self.render_mode = render_mode
        self.window_size = window_size
        
        self.vertices = None
        self.normalized_vertices = None
        self.lod_vertices = None
        
        # Pre-compute axis vectors (optimization 1)
        self.base_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # Pre-compute projection constants (optimization 2)
        self.projection_alpha = 0.35
        self.projection_scale = np.array([1.0, self.projection_alpha, -1.0], dtype=np.float32)  # [x, y, z] scaling
        
        self.load_probe()
        self.precompute()

    def load_probe(self):
        """Load probe vertices from file"""
        try:
            self.vertices = load_stl_vertices(self.probe_path)
            print(f"Loaded probe with {len(self.vertices)} vertices")
        except Exception as e:
            print(f"Failed to load probe: {e}")
            self.vertices = None

    def precompute(self):
        """Pre-compute normalized and LOD vertices"""
        if self.vertices is None:
            return
        self.normalized_vertices = self._normalize_vertices()
        # Apply LOD once during loading
        self.lod_vertices = self._apply_lod(self.normalized_vertices, self.lod_level)
        print(f"Pre-computed normalization and LOD ({len(self.lod_vertices)} vertices)")

    def _normalize_vertices(self):
        """Normalize vertices to unit cube"""
        v = self.vertices.copy()
        center = (v.min(axis=0) + v.max(axis=0)) / 2.0
        v = v - center
        max_range = float(np.max(v.max(axis=0) - v.min(axis=0)))
        if max_range > 0:
            v = v / max_range
        return v

    def _apply_lod(self, vertices, lod_level):
        """Apply Level of Detail by decimating vertices"""
        if lod_level >= 1.0:
            return vertices
        num_vertices = max(1, int(len(vertices) * lod_level))
        step = max(1, len(vertices) // num_vertices)
        return vertices[::step]

    def amplify_rotation_prediction(self, rotation_matrix, current_imu_rotation):
        """Amplify rotation prediction and combine with current IMU rotation"""
        if rotation_matrix is None:
            return None
            
        # Extract rotation angles and amplify
        r = R.from_matrix(rotation_matrix)
        angles = r.as_euler('xyz', degrees=True)
        exaggerated_angles = angles * self.rotation_amplification
        exaggerated_matrix = R.from_euler('xyz', exaggerated_angles, degrees=True).as_matrix()
        
        # Apply inverse transformation
        exaggerated_inverse = np.linalg.inv(exaggerated_matrix)
        
        # Combine with current IMU rotation
        return current_imu_rotation @ exaggerated_inverse

    def render_widget(self, input_frame: np.ndarray = None, qw=1.0, qx=0.0, qy=0.0, qz=0.0, predicted_rotation_matrix=None):
        """Main rendering method - renders widget on input frame"""
        # Apply amplification to predicted rotation if provided
        amplified_predicted_rotation = None
        if predicted_rotation_matrix is not None:
            current_imu_rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            amplified_predicted_rotation = self.amplify_rotation_prediction(predicted_rotation_matrix, current_imu_rotation)
        
        if self.render_mode == 'fullscreen':
            return self._render_fullscreen(qw, qx, qy, qz, amplified_predicted_rotation)
        else:  # overlay mode
            return self._render_overlay(input_frame, qw, qx, qy, qz, amplified_predicted_rotation)

    def _render_fullscreen(self, qw, qx, qy, qz, predicted_rotation_matrix):
        """Render fullscreen widget on background"""
        widget_width, widget_height = self.window_size
        widget_size = min(widget_width, widget_height)
        
        # Create background
        result_frame = np.full((widget_height, widget_width, 3), self.background_color, dtype=np.uint8)
        
        # Center the square widget
        center_x, center_y = widget_width // 2, widget_height // 2
        position = (center_x, center_y)
        size = widget_size // 2
        
        # Draw widget
        return self._draw_combined_with_params(result_frame, qw, qx, qy, qz, predicted_rotation_matrix, position, size)

    def _render_overlay(self, input_frame, qw, qx, qy, qz, predicted_rotation_matrix):
        """Render overlay widget in top-left corner"""
        h, w = input_frame.shape[:2]
        min_dim = min(h, w)
        widget_size = int(min_dim * 0.2)
        
        # Position widget in top-left corner with margin
        top_left_x, top_left_y = widget_size, widget_size
        position = (top_left_x, top_left_y)
        size = widget_size
        
        # Draw on input frame
        return self._draw_combined_with_params(input_frame, qw, qx, qy, qz, predicted_rotation_matrix, position, size)

    def _draw_combined_with_params(self, input_frame, qw, qx, qy, qz, predicted_rotation_matrix, position, size):
        """Draw combined widget with specific position and size"""
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        
        # Draw IMU axes
        input_frame = self._draw_3axis_vector_with_params(input_frame, R, position, size)
        
        # Draw predicted axes if available
        if predicted_rotation_matrix is not None:
            input_frame = self._draw_3axis_vector_predicted_with_params(input_frame, predicted_rotation_matrix, position, size)
        
        # Draw probe model if available
        if self.vertices is not None:
            input_frame = self._draw_probe_model_with_params(input_frame, R, position, size)
        
        return input_frame

    def _draw_3axis_vector_with_params(self, input_frame, R, position, size):
        """Draw 3D axis vectors with parameters"""
        start_x, start_y = position
        axis_length = size * 0.75

        # Use pre-computed axis vectors (optimization 1)
        # Apply rotation to all axes at once
        rotated_axes = R @ self.base_axes.T  # Shape: (3, 3)
        x_rot = rotated_axes[:, 0]
        y_rot = rotated_axes[:, 1] 
        z_rot = rotated_axes[:, 2]

        # Project to 2D using pre-computed constants (optimization 2)
        def proj(v):
            px = start_x + int((v[0] + self.projection_alpha * v[1]) * axis_length)
            py = start_y - int(v[2] * axis_length)
            return (px, py)

        origin = (start_x, start_y)
        x_end = proj(x_rot)
        y_end = proj(y_rot)
        z_end = proj(z_rot)

        # Draw axes
        x_color, y_color, z_color = self.axis_colors
        x_label, y_label, z_label = self.axis_labels

        cv2.line(input_frame, origin, x_end, x_color, self.axis_thickness)
        cv2.putText(input_frame, x_label, (x_end[0] + 6, x_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, x_color, 2)
        
        cv2.line(input_frame, origin, y_end, y_color, self.axis_thickness)
        cv2.putText(input_frame, y_label, (y_end[0] + 6, y_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, y_color, 2)
        
        cv2.line(input_frame, origin, z_end, z_color, self.axis_thickness)
        cv2.putText(input_frame, z_label, (z_end[0] + 6, z_end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, z_color, 2)
        
        return input_frame

    def _draw_3axis_vector_predicted_with_params(self, input_frame, R_pred, position, size):
        """Draw predicted 3D axis vectors with dashed lines"""
        start_x, start_y = position
        axis_length = size * 0.75

        # Use pre-computed axis vectors (optimization 1)
        # Apply rotation to all axes at once
        rotated_axes = R_pred @ self.base_axes.T  # Shape: (3, 3)
        x_rot = rotated_axes[:, 0]
        y_rot = rotated_axes[:, 1] 
        z_rot = rotated_axes[:, 2]

        # Project to 2D using pre-computed constants (optimization 2)
        def proj(v):
            px = start_x + int((v[0] + self.projection_alpha * v[1]) * axis_length)
            py = start_y - int(v[2] * axis_length)
            return (px, py)

        origin = (start_x, start_y)
        x_end = proj(x_rot)
        y_end = proj(y_rot)
        z_end = proj(z_rot)

        # Draw dashed axes
        x_color, y_color, z_color = self.axis_colors

        overlay = input_frame.copy()
        self._draw_dashed_line(overlay, origin, x_end, x_color, self.axis_thickness)
        self._draw_dashed_line(overlay, origin, y_end, y_color, self.axis_thickness)
        self._draw_dashed_line(overlay, origin, z_end, z_color, self.axis_thickness)
        
        cv2.addWeighted(overlay, self.predicted_axis_opacity, input_frame, 1.0 - self.predicted_axis_opacity, 0, input_frame)
        return input_frame

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness):
        """Draw dashed line between two points"""
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        distance = int(np.sqrt(dx*dx + dy*dy))
        
        if distance == 0:
            return
            
        dx /= distance
        dy /= distance
        dash_length, gap_length = 8, 4
        current = 0
        
        while current < distance:
            start = current
            end = min(current + dash_length, distance)
            sx = int(x1 + start * dx)
            sy = int(y1 + start * dy)
            ex = int(x1 + end * dx)
            ey = int(y1 + end * dy)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            current += dash_length + gap_length
    
    def set_render_mode(self, mode: str):
        """Set the rendering mode."""
        if mode in ['fullscreen', 'overlay']:
            self.render_mode = mode
        else:
            raise ValueError(f"Invalid render mode: {mode}")
    
    def get_render_mode(self) -> str:
        """Get the current rendering mode."""
        return self.render_mode
    
    def destroy(self):
        """Clean up resources (2D renderer doesn't need special cleanup)."""
        pass

    def _draw_probe_model_with_params(self, input_frame, R, position, size):
        """Draw probe model with parameters"""
        if self.lod_vertices is None:
            return input_frame
        
        # 
        # Apply rotation
        v = self.lod_vertices @ R.T

        # Project and draw
        projected = self._project_vertices_with_params(v, position, size)
        self._draw_triangles_optimized(input_frame, projected)
        return input_frame

    def _project_vertices_with_params(self, vertices, position, size):
        """Project 3D vertices to 2D with parameters"""
        if len(vertices) == 0:
            return np.empty((0, 2), dtype=np.int32)
            
        # Use pre-computed projection constants (optimization 2)
        # Vectorized projection: [x + alpha*y, -z]
        projected_xy = vertices[:, :2] * self.projection_scale[:2]  # [x, alpha*y]
        px = projected_xy[:, 0] + projected_xy[:, 1]  # x + alpha*y
        py = -vertices[:, 2]  # -z
        
        v2d = np.stack([px, py], axis=1) * float(size) * self.probe_scale
        v2d = v2d + np.asarray(position, dtype=np.float32)
        v2d = np.clip(v2d, 0, 10000)
        return v2d.astype(np.int32)

    def _draw_triangles_optimized(self, input_frame, projected):
        """Draw probe triangles as wireframe"""
        color = (180, 200, 220)
        thickness = 2
        
        for i in range(0, len(projected), 3):
            if i + 2 < len(projected):
                p0, p1, p2 = projected[i], projected[i+1], projected[i+2]
                cv2.line(input_frame, tuple(p0), tuple(p1), color, thickness)
                cv2.line(input_frame, tuple(p1), tuple(p2), color, thickness)
                cv2.line(input_frame, tuple(p2), tuple(p0), color, thickness)