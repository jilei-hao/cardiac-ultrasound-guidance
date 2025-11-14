from dataclasses import dataclass
from .fourch_landmark_renderer import FourChLandmarkRenderer

@dataclass
class GuidanceRenderingConfig:
    landmark_line_thickness: int = 2
    landmark_point_radius: int = 8
    landmark_color_legend_path: str = None
    guidance_widget_legend_path: str = None

class GuidanceRenderingPipeline:
    def __init__(self, ui_config: GuidanceRenderingConfig):
        self.ui_config = ui_config
        self.landmark_renderer = FourChLandmarkRenderer(
            line_thickness=ui_config.landmark_line_thickness,
            landmark_radius=ui_config.landmark_point_radius,
            color_legend_path=ui_config.landmark_color_legend_path
        )

    def render_frame(self, frame_bgr, landmarks_tensor):
        """
        Render the guidance frame with overlaid landmarks.
        
        Args:
            frame_bgr: Input BGR image frame as a NumPy array
            landmarks_tensor: Landmarks tensor as a NumPy array of shape (N, 4)
                              where each landmark is (x, y, uncertainty, label)
        
        Returns:
            Rendered BGR image frame with landmarks overlaid
        """
        rendered_frame_bgr = self.landmark_renderer.overlay_landmarks(frame_bgr, landmarks_tensor)
        return rendered_frame_bgr