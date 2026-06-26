"""
Qt-based image window implementation for cross-platform support.
Works on Linux, macOS, and Windows using PySide6/PyQt6.
"""
from typing import Tuple
import numpy as np

try:
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap, QKeyEvent
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QImage, QPixmap, QKeyEvent
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        print("Warning: Neither PySide6 nor PyQt6 is available. QtImageWindow will not work.")

from .abstract_image_window import AbstractImageWindow


class QtImageWindow(AbstractImageWindow):
    """
    Qt-based image window implementation.
    Cross-platform support for Linux, macOS, and Windows.
    """
    
    # Class-level QApplication instance (shared across all windows)
    _qapp = None
    _window_count = 0
    
    def __init__(self, 
                 title: str = "Image Window",
                 width: int = 512,
                 height: int = 512,
                 x_pos: int = 100,
                 y_pos: int = 100,
                 background_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize the Qt image window.
        
        Args:
            title: Window title
            width: Initial window width
            height: Initial window height
            x_pos: Initial window x position
            y_pos: Initial window y position
            background_color: RGB background color tuple
        """
        if not QT_AVAILABLE:
            raise RuntimeError("Qt (PySide6 or PyQt6) is not available. Cannot create QtImageWindow.")
        
        super().__init__(title, width, height, x_pos, y_pos, background_color)
        
        self.window = None
        self.label = None
        self.current_pixmap = None
        
    @classmethod
    def _ensure_qapp(cls):
        """Ensure QApplication instance exists."""
        if cls._qapp is None:
            import sys
            cls._qapp = QApplication.instance()
            if cls._qapp is None:
                cls._qapp = QApplication(sys.argv)
        return cls._qapp
    
    def initialize(self) -> bool:
        """
        Initialize the Qt window.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
        
        try:
            # Ensure QApplication exists
            self._ensure_qapp()
            
            # Create main window
            self.window = QMainWindow()
            self.window.setWindowTitle(self.title)
            self.window.resize(self.width, self.height)
            self.window.move(self.x_pos, self.y_pos)
            
            # Create central widget with label for image display
            central_widget = QWidget()
            self.window.setCentralWidget(central_widget)
            
            # Create label for displaying images
            self.label = QLabel(central_widget)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setScaledContents(False)
            
            # Set background color
            bg_color = f"rgb({self.background_color[0]}, {self.background_color[1]}, {self.background_color[2]})"
            self.label.setStyleSheet(f"QLabel {{ background-color: {bg_color}; }}")
            
            # Make label fill the window
            self.label.setGeometry(0, 0, self.width, self.height)
            
            # Install event filter for keyboard events
            self.window.keyPressEvent = self._on_key_press
            
            # Handle window close event
            self.window.closeEvent = self._on_close_event
            # Handle window resize event to rescale displayed image
            self.window.resizeEvent = self._on_resize_event
            
            # Show window
            self.window.show()
            
            QtImageWindow._window_count += 1
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Qt window: {e}")
            return False
    
    def destroy(self):
        """
        Clean up and destroy the window.
        """
        if self.window is not None:
            try:
                self.window.close()
                self.window.deleteLater()
                QtImageWindow._window_count -= 1
            except Exception as e:
                print(f"Error destroying window: {e}")
            finally:
                self.window = None
                self.label = None
                self.current_pixmap = None
        
        self.is_initialized = False
    
    def display_frame_rgb(self, frame_rgb: np.ndarray) -> bool:
        """
        Display an RGB frame in the window.
        
        Args:
            frame_rgb: NumPy array representing the image in RGB format (H, W, 3)
            
        Returns:
            bool: True if display successful, False otherwise
        """
        if not self.is_initialized or self.label is None:
            return False
        
        try:
            # Get frame dimensions
            height, width = frame_rgb.shape[:2]
            
            # Handle grayscale images
            if len(frame_rgb.shape) == 2:
                # Convert grayscale to RGB
                frame_rgb = np.stack([frame_rgb] * 3, axis=-1)
            elif frame_rgb.shape[2] == 4:
                # Convert RGBA to RGB
                frame_rgb = frame_rgb[:, :, :3]
            
            # Ensure the array is contiguous and in the right format
            frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
            
            # Create QImage from numpy array
            bytes_per_line = 3 * width
            q_image = QImage(
                frame_rgb.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.label.setPixmap(scaled_pixmap)
            self.current_pixmap = pixmap  # Keep reference to avoid garbage collection
            
            return True
            
        except Exception as e:
            print(f"Failed to display frame: {e}")
            return False
    
    def update(self) -> bool:
        """
        Update the window and process events.
        
        Returns:
            bool: True if window is still active, False if should close
        """
        if not self.is_initialized:
            return False
        
        try:
            # Process Qt events
            QApplication.processEvents()
            return self.is_window_active()
        except Exception as e:
            print(f"Error updating window: {e}")
            return False
    
    def set_window_size(self, width: int, height: int):
        """
        Set the window size.
        
        Args:
            width: New window width
            height: New window height
        """
        if self.window is not None:
            self.window.resize(width, height)
            if self.label is not None:
                self.label.setGeometry(0, 0, width, height)
            self.width = width
            self.height = height
    
    def set_window_position(self, x: int, y: int):
        """
        Set the window position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.window is not None:
            self.window.move(x, y)
            self.x_pos = x
            self.y_pos = y
    
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get current window size.
        
        Returns:
            Tuple of (width, height)
        """
        if self.window is not None:
            size = self.window.size()
            return (size.width(), size.height())
        return (self.width, self.height)
    
    def set_title(self, title: str):
        """
        Set the window title.
        
        Args:
            title: New window title
        """
        if self.window is not None:
            self.window.setWindowTitle(title)
        self.title = title
    
    def set_resizable(self, resizable: bool):
        """
        Set whether the window can be resized.
        
        Args:
            resizable: True to allow resizing, False otherwise
        """
        if self.window is not None:
            if resizable:
                self.window.setFixedSize(16777215, 16777215)  # Max size
            else:
                size = self.window.size()
                self.window.setFixedSize(size)
    
    def set_always_on_top(self, on_top: bool):
        """
        Set whether the window should stay on top.
        
        Args:
            on_top: True to keep on top, False otherwise
        """
        if self.window is not None:
            if on_top:
                self.window.setWindowFlags(
                    self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
                )
            else:
                self.window.setWindowFlags(
                    self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint
                )
            self.window.show()  # Need to show again after changing flags
    
    def is_window_active(self) -> bool:
        """
        Check if the window is still active/open.
        
        Returns:
            bool: True if window is active, False if closed
        """
        return self.window is not None and self.window.isVisible()
    
    def _on_key_press(self, event: QKeyEvent):
        """
        Handle key press events.
        
        Args:
            event: Qt key event
        """
        # Map Qt key to string representation
        key_map = {
            Qt.Key.Key_Escape: 'Escape',
            Qt.Key.Key_Return: 'Return',
            Qt.Key.Key_Enter: 'Return',
            Qt.Key.Key_Space: 'space',
            Qt.Key.Key_Up: 'Up',
            Qt.Key.Key_Down: 'Down',
            Qt.Key.Key_Left: 'Left',
            Qt.Key.Key_Right: 'Right',
            Qt.Key.Key_Q: 'q',
            Qt.Key.Key_E: 'e',
            Qt.Key.Key_F: 'f',
            Qt.Key.Key_R: 'r',
            Qt.Key.Key_S: 's',
            Qt.Key.Key_W: 'w',
            Qt.Key.Key_A: 'a',
            Qt.Key.Key_D: 'd',
            # Letters with SHIFTED bindings ('Shift+c' reset-calibration,
            # 'Shift+m' continuous-border) must be mapped explicitly so they
            # lowercase here; otherwise the Shift fallback uses event.text(),
            # which returns 'C'/'M' and dispatches as 'Shift+C'/'Shift+M',
            # missing the lowercase handler keys.
            Qt.Key.Key_C: 'c',
            Qt.Key.Key_M: 'm',
            # Punctuation used by camera nudge / reset hotkeys. Mapping
            # these explicitly bypasses ``event.text()``, which on some
            # macOS keyboard layouts can return empty for punctuation
            # (the symptom: hotkey looks dead even though the key is
            # being pressed).
            Qt.Key.Key_Comma: ',',
            Qt.Key.Key_Period: '.',
            Qt.Key.Key_Apostrophe: "'",
            Qt.Key.Key_Slash: '/',
            Qt.Key.Key_Semicolon: ';',
        }
        
        qt_key = event.key()

        # Try to get mapped key
        if qt_key in key_map:
            key_str = key_map[qt_key]
        else:
            # Try to get text representation for other keys
            key_str = event.text()

        if not key_str:
            return

        # Prepend "Shift+" when shift is held. Arrow/named keys pass
        # through ``key_map`` and never carry shift state in their text
        # form, so the prefix is the only signal. Letter keys also pass
        # through ``key_map`` (always lowercased), so we need the prefix
        # there too — without it, ``Shift+r`` would dispatch to the
        # ``r`` handler, masking the shifted binding.
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            key_str = f'Shift+{key_str}'

        self._handle_key_event(key_str)
    
    def _on_close_event(self, event):
        """
        Handle window close event.
        
        Args:
            event: Qt close event
        """
        self.is_initialized = False
        QtImageWindow._window_count -= 1
        event.accept()

    def _on_resize_event(self, event):
        """
        Handle window resize event: update label geometry and rescale image.
        
        Args:
            event: Qt resize event
        """
        try:
            if self.window is None:
                return
            # Update tracked size
            new_size = self.window.size()
            self.width, self.height = new_size.width(), new_size.height()
            # Ensure label fills the client area
            if self.label is not None:
                self.label.setGeometry(0, 0, self.width, self.height)
                # Rescale current image to fit new size while preserving aspect ratio
                if self.current_pixmap is not None and not self.current_pixmap.isNull():
                    scaled = self.current_pixmap.scaled(
                        self.label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.label.setPixmap(scaled)
        finally:
            event.accept()
