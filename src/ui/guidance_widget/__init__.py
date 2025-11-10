"""
Guidance Widget Renderer Package

Provides unified interface for both 2D and 3D guidance widget rendering
with automatic fallback capabilities.
"""

from .guidance_widget_renderer import (
    GuidanceWidgetRenderer,
    GuidanceWidgetRendererFactory,
    create_guidance_widget_renderer
)

# For backward compatibility, expose factory create function as main class
GuidanceWidgetRenderer = create_guidance_widget_renderer

__all__ = [
    'GuidanceWidgetRenderer',
    'GuidanceWidgetRendererFactory', 
    'create_guidance_widget_renderer'
]
