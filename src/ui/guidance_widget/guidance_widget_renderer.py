"""
Abstract base class for guidance widget renderers with factory pattern.
Provides a unified interface for both 2D and 3D guidance widget rendering.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Union
import os


class GuidanceWidgetRenderer(ABC):
    """
    Abstract base class for guidance widget renderers.
    
    Defines the common interface for both 2D and 3D guidance widget rendering,
    including probe visualization and coordinate axes display.
    """
    
    def __init__(self, probe_path: str, **kwargs):
        """
        Initialize the guidance widget renderer.
        
        Args:
            probe_path: Path to the probe mesh/model file
            **kwargs: Renderer-specific configuration parameters
        """
        self.probe_path = probe_path
        self.config = kwargs
        
    @abstractmethod
    def render_widget(self, input_frame: Optional[np.ndarray] = None, 
                     qw: float = 1.0, qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
                     predicted_rotation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render the guidance widget with current probe orientation.
        
        Args:
            input_frame: Input frame to overlay on (optional for fullscreen mode)
            qw, qx, qy, qz: Quaternion components representing probe orientation
            predicted_rotation_matrix: Optional predicted rotation for visualization
            
        Returns:
            np.ndarray: Rendered frame with guidance widget
        """
        pass
    
    @abstractmethod
    def set_render_mode(self, mode: str):
        """
        Set the rendering mode.
        
        Args:
            mode: Rendering mode ('fullscreen' or 'overlay')
        """
        pass
    
    @abstractmethod
    def get_render_mode(self) -> str:
        """
        Get the current rendering mode.
        
        Returns:
            str: Current rendering mode
        """
        pass
    
    @abstractmethod
    def destroy(self):
        """Clean up resources used by the renderer."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.destroy()


class GuidanceWidgetRendererFactory:
    """
    Factory class for creating guidance widget renderers.
    
    Automatically selects between 2D and 3D renderers based on availability
    and system capabilities, with 2D as fallback.
    """
    
    @staticmethod
    def create_renderer(probe_path: str, 
                       renderer_type: str = 'auto',
                       fallback_to_2d: bool = True,
                       **kwargs) -> GuidanceWidgetRenderer:
        """
        Create a guidance widget renderer with automatic fallback.
        
        Args:
            probe_path: Path to the probe mesh/model file
            renderer_type: Type of renderer ('2d', '3d', or 'auto')
            fallback_to_2d: Whether to fallback to 2D if 3D fails
            **kwargs: Renderer-specific configuration parameters
            
        Returns:
            GuidanceWidgetRenderer: Instance of appropriate renderer
            
        Raises:
            RuntimeError: If no suitable renderer can be created
        """
        
        if renderer_type == '2d':
            return GuidanceWidgetRendererFactory._create_2d_renderer(probe_path, **kwargs)
        elif renderer_type == '3d':
            return GuidanceWidgetRendererFactory._create_3d_renderer(probe_path, **kwargs)
        elif renderer_type == 'auto':
            # Try 3D first, fallback to 2D if needed
            try:
                return GuidanceWidgetRendererFactory._create_3d_renderer(probe_path, **kwargs)
            except Exception as e:
                if fallback_to_2d:
                    print(f"⚠️  3D renderer failed ({e}), falling back to 2D renderer")
                    return GuidanceWidgetRendererFactory._create_2d_renderer(probe_path, **kwargs)
                else:
                    raise RuntimeError(f"Failed to create 3D renderer and fallback disabled: {e}")
        else:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
    
    @staticmethod
    def _create_3d_renderer(probe_path: str, **kwargs) -> GuidanceWidgetRenderer:
        """Create 3D renderer instance."""
        try:
            from .guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
            return GuidanceWidgetRenderer3D(probe_path, **kwargs)
        except ImportError as e:
            raise RuntimeError(f"3D renderer dependencies not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize 3D renderer: {e}")
    
    @staticmethod
    def _create_2d_renderer(probe_path: str, **kwargs) -> GuidanceWidgetRenderer:
        """Create 2D renderer instance."""
        try:
            from .guidance_widget_renderer_2d import GuidanceWidgetRenderer2D
            return GuidanceWidgetRenderer2D(probe_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize 2D renderer: {e}")
    
    @staticmethod
    def get_available_renderers() -> list[str]:
        """
        Get list of available renderer types.
        
        Returns:
            list[str]: List of available renderer types
        """
        available = []
        
        # Check 2D renderer
        try:
            from .guidance_widget_renderer_2d import GuidanceWidgetRenderer2D
            available.append('2d')
        except ImportError:
            pass
        
        # Check 3D renderer
        try:
            from .guidance_widget_renderer_3d import GuidanceWidgetRenderer3D
            available.append('3d')
        except ImportError:
            pass
        
        return available


# Convenience function for backward compatibility
def create_guidance_widget_renderer(probe_path: str, 
                                   renderer_type: str = 'auto',
                                   **kwargs) -> GuidanceWidgetRenderer:
    """
    Convenience function to create a guidance widget renderer.
    
    Args:
        probe_path: Path to the probe mesh/model file
        renderer_type: Type of renderer ('2d', '3d', or 'auto')
        **kwargs: Renderer-specific configuration parameters
        
    Returns:
        GuidanceWidgetRenderer: Instance of appropriate renderer
    """
    return GuidanceWidgetRendererFactory.create_renderer(probe_path, renderer_type, **kwargs)
