from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np


class AbstractImageWindow(ABC):
    """
    Abstract base class for cross-platform image window implementations.
    Provides a common interface for displaying frames, handling keyboard input,
    and managing window properties.
    """
    
    def __init__(self, 
                 title: str = "Image Window",
                 width: int = 512,
                 height: int = 512,
                 x_pos: int = 100,
                 y_pos: int = 100,
                 background_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize the abstract image window.
        
        Args:
            title: Window title
            width: Initial window width
            height: Initial window height
            x_pos: Initial window x position
            y_pos: Initial window y position
            background_color: RGB background color tuple
        """
        self.title = title
        self.width = width
        self.height = height
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.background_color = background_color
        self.is_initialized = False
        self.key_handlers = {}  # Dictionary mapping keys to handler functions
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the window and underlying GUI framework.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def destroy(self):
        """
        Clean up and destroy the window.
        """
        pass
    
    @abstractmethod
    def display_frame(self, frame: np.ndarray) -> bool:
        """
        Display a frame in the window.
        
        Args:
            frame: NumPy array representing the image (BGR or RGB format)
            
        Returns:
            bool: True if display successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self) -> bool:
        """
        Update the window and process events.
        
        Returns:
            bool: True if window is still active, False if should close
        """
        pass
    
    @abstractmethod
    def set_window_size(self, width: int, height: int):
        """
        Set the window size.
        
        Args:
            width: New window width
            height: New window height
        """
        pass
    
    @abstractmethod
    def set_window_position(self, x: int, y: int):
        """
        Set the window position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        pass
    
    @abstractmethod
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get current window size.
        
        Returns:
            Tuple of (width, height)
        """
        pass
    
    @abstractmethod
    def set_title(self, title: str):
        """
        Set the window title.
        
        Args:
            title: New window title
        """
        pass
    
    def register_key_handler(self, key: str, handler: Callable[[], None]):
        """
        Register a keyboard event handler.
        
        Args:
            key: Key identifier (e.g., 'e', 'f', 'Up', 'Down', etc.)
            handler: Function to call when key is pressed
        """
        self.key_handlers[key] = handler
    
    def unregister_key_handler(self, key: str):
        """
        Unregister a keyboard event handler.
        
        Args:
            key: Key identifier to unregister
        """
        if key in self.key_handlers:
            del self.key_handlers[key]
    
    def _handle_key_event(self, key: str):
        """
        Internal method to handle key events by calling registered handlers.
        
        Args:
            key: Key that was pressed
        """
        if key in self.key_handlers:
            try:
                self.key_handlers[key]()
            except Exception as e:
                print(f"Error in key handler for '{key}': {e}")
    
    @abstractmethod
    def set_resizable(self, resizable: bool):
        """
        Set whether the window can be resized.
        
        Args:
            resizable: True to allow resizing, False otherwise
        """
        pass
    
    @abstractmethod
    def set_always_on_top(self, on_top: bool):
        """
        Set whether the window should stay on top.
        
        Args:
            on_top: True to keep on top, False otherwise
        """
        pass
    
    @abstractmethod
    def is_window_active(self) -> bool:
        """
        Check if the window is still active/open.
        
        Returns:
            bool: True if window is active, False if closed
        """
        pass
    
    def get_display_info(self) -> Dict[str, Any]:
        """
        Get information about the current display state.
        
        Returns:
            Dictionary with display information
        """
        return {
            'title': self.title,
            'size': (self.width, self.height),
            'position': (self.x_pos, self.y_pos),
            'background_color': self.background_color,
            'initialized': self.is_initialized,
            'active': self.is_window_active() if self.is_initialized else False
        }
