"""
Cross-platform image window implementations for cardiac ultrasound guidance.

This module provides abstract and platform-specific image window classes
that can display frames, handle keyboard input, and manage window properties.

Default implementation uses QtImageWindow (PySide6/PyQt6) for cross-platform support.
"""

from .abstract_image_window import AbstractImageWindow
from .window_factory import (
    get_image_window_class,
    create_image_window,
    create_dual_window_setup,
    WindowManager
)

# Qt implementation (cross-platform default)
try:
    from .qt_image_window import QtImageWindow
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Platform-specific implementations (imported on-demand)
__all__ = [
    'AbstractImageWindow',
    'get_image_window_class', 
    'create_image_window',
    'create_dual_window_setup',
    'WindowManager',
    'QtImageWindow' if QT_AVAILABLE else None,
]

# Remove None from __all__
__all__ = [item for item in __all__ if item is not None]
