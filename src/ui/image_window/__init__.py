"""
Cross-platform image window implementations for cardiac ultrasound guidance.

This module provides abstract and platform-specific image window classes
that can display frames, handle keyboard input, and manage window properties.
"""

from .abstract_image_window import AbstractImageWindow
from .window_factory import (
    get_image_window_class,
    create_image_window,
    create_dual_window_setup,
    WindowManager
)

# Platform-specific implementations (imported on-demand)
__all__ = [
    'AbstractImageWindow',
    'get_image_window_class', 
    'create_image_window',
    'create_dual_window_setup',
    'WindowManager'
]
