"""
4-Chamber Landmark Renderer
Configurable class for rendering 4-chamber landmarks with connecting lines.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from .image_helpers import overlay_color_legend


class FourChamberLandmarkRenderer:
    """
    Configurable renderer for 4-chamber landmarks.
    
    Features:
    - Tunable color map for each chamber
    - Configurable opacity (base and uncertainty-based)
    - Adjustable landmark size (radius)
    - Configurable number of landmarks per chamber
    - Connecting lines between landmarks
    """
    
    def __init__(
        self,
        landmark_radius: int = 5,
        min_opacity: float = 0.6,
        max_opacity: float = 1.0,
        uncertainty_threshold: float = 1.0,
        line_thickness_ratio: float = 0.5,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        landmarks_per_chamber: Optional[Dict[int, int]] = None,
        reference_size: Optional[int] = None, 
        color_legend_png_path: Optional[str] = None
    ):
        """
        Initialize the 4-chamber landmark renderer.
        
        Args:
            landmark_radius: Radius of landmark circles (pixels)
            min_opacity: Minimum opacity (0-1) for landmarks with high uncertainty
            max_opacity: Maximum opacity (0-1) for landmarks with low uncertainty
            uncertainty_threshold: Maximum uncertainty value to display landmarks
            line_thickness_ratio: Line thickness as ratio of landmark radius
            color_map: Dictionary mapping chamber label to BGR color tuple
                      Default: {50: LV, 100: RV, 150: LA, 200: RA}
            landmarks_per_chamber: Dictionary mapping chamber label to number of landmarks
                                 Default: {50: 24 (LV), 100: 12, 150: 12, 200: 12}
            reference_size: Reference size for coordinate scaling (None = use frame coordinates)
        """
        self.landmark_radius = landmark_radius
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.uncertainty_threshold = uncertainty_threshold
        self.line_thickness_ratio = line_thickness_ratio
        self.line_thickness = max(1, int(landmark_radius * line_thickness_ratio))
        self.reference_size = reference_size
        self.color_legend_bgra = None  # Cached legend image
        
        # Default color map (BGR format for OpenCV)
        # Original RGB colors: LV=(255,100,180), RV=(200,200,50), LA=(100,255,150), RA=(255,150,100)
        if color_map is None:
            self.color_map = {
                50: (180, 100, 255),      # Left Ventricle (LV) - Bright Pink/Magenta (RGB: 255,100,180)
                100: (50, 200, 200),      # Right Ventricle (RV) - Bright Yellow (RGB: 200,200,50)
                150: (150, 255, 100),     # Left Atrium (LA) - Bright Green (RGB: 100,255,150)
                200: (100, 150, 255),     # Right Atrium (RA) - Bright Orange (RGB: 255,150,100)
            }
        else:
            self.color_map = color_map
        
        # Default landmarks per chamber
        if landmarks_per_chamber is None:
            self.landmarks_per_chamber = {
                50: 24,   # Left Ventricle: 24 landmarks
                100: 12,  # Right Ventricle: 12 landmarks
                150: 12,  # Left Atrium: 12 landmarks
                200: 12,  # Right Atrium: 12 landmarks
            }
        else:
            self.landmarks_per_chamber = landmarks_per_chamber


        # Read legend png and cache bgra image if path provided
        if color_legend_png_path is not None:
            legend_bgra = cv2.imread(color_legend_png_path, cv2.IMREAD_UNCHANGED)

            if legend_bgra is not None:
                self.color_legend_bgra = legend_bgra
            else:
                print(f"Warning: Could not read color legend PNG from {color_legend_png_path}")
    
    def get_color_for_chamber(self, label: int) -> Tuple[int, int, int]:
        """Get color for a chamber based on label value (BGR format for OpenCV)."""
        return self.color_map.get(int(label), (255, 255, 255))  # White for unknown labels
    
    def uncertainty_to_opacity(self, uncertainty: float) -> float:
        """
        Convert uncertainty to opacity.
        
        Args:
            uncertainty: Uncertainty value (0 = certain, 1 = uncertain)
            
        Returns:
            Opacity value between min_opacity and max_opacity
        """
        # Invert: lower uncertainty = higher opacity
        opacity = self.min_opacity + (self.max_opacity - self.min_opacity) * (1.0 - uncertainty)
        return float(np.clip(opacity, self.min_opacity, self.max_opacity))
    

    def _overlay_color_legend(self, input_frame):
        # Overlay the color legend on the input frame (top-left corner)
        if self.color_legend_bgra is None:
            return input_frame
        
        overlay_color_legend(input_frame, self.color_legend_bgra, top=10, left=10, pct_width=0.2, margin=20)
        return input_frame
    
    def overlay_landmarks(
        self,
        vis_frame: np.ndarray,
        landmarks_tensor: np.ndarray,
        reference_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Overlay landmarks on the visualization frame with connecting lines.
        
        Args:
            vis_frame: Visualization frame (BGR format)
            landmarks_tensor: Tensor with landmark coordinates (N, 4) format:
                            [x, y, uncertainty, label]
            reference_size: Override reference_size if provided
            
        Returns:
            Frame with landmarks overlaid
        """
        if landmarks_tensor.shape[0] == 0:
            return vis_frame
        
        # Check if landmarks have label information (4 elements)
        has_labels = landmarks_tensor.shape[1] >= 4
        if not has_labels:
            # Fallback: treat as (x, y, ere) format
            return self._overlay_landmarks_legacy(vis_frame, landmarks_tensor, reference_size)
        
        # Use provided reference_size or instance default
        ref_size = reference_size if reference_size is not None else self.reference_size
        
        # Scale coordinates if landmarks are in reference_size coordinates
        if ref_size is not None:
            scale_x = vis_frame.shape[1] / ref_size
            scale_y = vis_frame.shape[0] / ref_size
        else:
            scale_x = 1.0
            scale_y = 1.0
        
        # Group landmarks by chamber for drawing connecting lines
        landmarks_by_chamber = {}
        for c, lm in enumerate(landmarks_tensor):
            x, y, uncertainty, label = lm[0], lm[1], lm[2], lm[3]
            uncertainty_val = uncertainty.item() if hasattr(uncertainty, 'item') else float(uncertainty)
            
            if float(uncertainty_val) <= self.uncertainty_threshold:
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
            
            color = self.get_color_for_chamber(label)
            chamber_overlay = vis_frame.copy()
            
            # Draw lines connecting consecutive landmarks
            for i in range(len(chamber_landmarks)):
                x1, y1, _ = chamber_landmarks[i]
                x2, y2, _ = chamber_landmarks[(i + 1) % len(chamber_landmarks)]  # Connect last to first
                
                # Draw line on overlay
                cv2.line(chamber_overlay, (x1, y1), (x2, y2), color, self.line_thickness, lineType=cv2.LINE_AA)
            
            # Blend entire chamber overlay with average opacity
            avg_uncertainty = np.mean([unc for _, _, unc in chamber_landmarks])
            avg_chamber_opacity = self.uncertainty_to_opacity(avg_uncertainty)
            cv2.addWeighted(chamber_overlay, avg_chamber_opacity, vis_frame, 1.0 - avg_chamber_opacity, 0, vis_frame)
        
        # Draw landmark points
        for c, lm in enumerate(landmarks_tensor):
            x, y, uncertainty, label = lm[0], lm[1], lm[2], lm[3]
            color = self.get_color_for_chamber(label)
            
            # Convert uncertainty to opacity
            uncertainty_val = uncertainty.item() if hasattr(uncertainty, 'item') else float(uncertainty)
            opacity = self.uncertainty_to_opacity(uncertainty_val)
            
            # Only draw if uncertainty is below threshold
            if float(uncertainty_val) <= self.uncertainty_threshold:
                scaled_x = int(x.item() * scale_x)
                scaled_y = int(y.item() * scale_y)
                
                # Extract a small region around the landmark for efficient blending
                h, w = vis_frame.shape[:2]
                x1 = max(0, scaled_x - self.landmark_radius - 2)
                y1 = max(0, scaled_y - self.landmark_radius - 2)
                x2 = min(w, scaled_x + self.landmark_radius + 2)
                y2 = min(h, scaled_y + self.landmark_radius + 2)
                
                # Create small overlay for this landmark region
                region_original = vis_frame[y1:y2, x1:x2].copy()
                region_overlay = region_original.copy()
                
                # Draw circle relative to the region
                cv2.circle(region_overlay, (scaled_x - x1, scaled_y - y1), self.landmark_radius, color, -1)
                
                # Blend this region with calculated opacity
                region_blended = cv2.addWeighted(region_overlay, opacity, region_original, 1.0 - opacity, 0)
                vis_frame[y1:y2, x1:x2] = region_blended

        # Overlay color legend if available
        vis_frame = self._overlay_color_legend(vis_frame)
        
        return vis_frame
    
    def _overlay_landmarks_legacy(
        self,
        vis_frame: np.ndarray,
        landmarks_tensor: np.ndarray,
        reference_size: Optional[int] = None,
        ere_threshold: float = 6.0
    ) -> np.ndarray:
        """
        Legacy overlay for landmarks without label information (backward compatibility).
        
        Args:
            vis_frame: Visualization frame (BGR format)
            landmarks_tensor: Tensor with (x, y, ere) format
            reference_size: Reference size for coordinate scaling
            ere_threshold: ERE threshold for filtering
            
        Returns:
            Frame with landmarks overlaid
        """
        # This is a fallback for old landmark format
        # For now, return the frame unchanged if no labels are present
        # Can be extended if needed for backward compatibility
        return vis_frame
    
    def set_landmark_radius(self, radius: int):
        """Update landmark radius and recalculate line thickness."""
        self.landmark_radius = radius
        self.line_thickness = max(1, int(radius * self.line_thickness_ratio))
    
    def set_opacity_range(self, min_opacity: float, max_opacity: float):
        """Update opacity range."""
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
    
    def set_color_map(self, color_map: Dict[int, Tuple[int, int, int]]):
        """Update color map for chambers."""
        self.color_map = color_map
    
    def set_landmarks_per_chamber(self, landmarks_per_chamber: Dict[int, int]):
        """Update expected number of landmarks per chamber."""
        self.landmarks_per_chamber = landmarks_per_chamber

