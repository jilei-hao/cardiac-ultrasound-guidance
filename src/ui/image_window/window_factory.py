import platform
import sys
from typing import Type, Dict, Any, Optional
from .abstract_image_window import AbstractImageWindow


def _normalize_backend(backend: Optional[str]) -> Optional[str]:
    if backend is None:
        return None
    backend = backend.lower()
    if backend in {"tkinter", "tk"}:
        return "tk"
    if backend in {"opencv", "cv2", "highgui"}:
        return "opencv"
    return backend


def get_image_window_class(backend: Optional[str] = None) -> Type[AbstractImageWindow]:
    """
    Factory function to get the appropriate image window class for the current platform.
    Allows explicitly selecting a backend when needed.
    
    Returns:
        Class: Platform-specific image window class
    """
    backend = _normalize_backend(backend)
    system = platform.system().lower()

    # Handle explicit backend selection
    if backend == "qt":
        from .qt_image_window import QtImageWindow
        return QtImageWindow
    if backend == "tk":
        from .macos_image_window import MacOSImageWindow
        return MacOSImageWindow
    if backend == "opencv":
        from .linux_image_window import LinuxImageWindow
        return LinuxImageWindow
    
    # Default to Qt for all platforms (cross-platform solution)
    # Qt works on macOS, Linux, and Windows
    try:
        from .qt_image_window import QtImageWindow
        return QtImageWindow
    except ImportError:
        # Fall back to platform-specific implementations if Qt is not available
        if system == 'darwin':  # macOS
            from .macos_image_window import MacOSImageWindow
            return MacOSImageWindow
        elif system == 'linux':  # Linux/Ubuntu
            from .linux_image_window import LinuxImageWindow
            return LinuxImageWindow
        else:
            # Windows or other platforms - try Linux implementation as last resort
            print(f"Warning: Platform '{system}' not specifically supported and Qt not available. Using Linux implementation as fallback.")
            from .linux_image_window import LinuxImageWindow
            return LinuxImageWindow


def create_image_window(*, backend: Optional[str] = None, **kwargs) -> AbstractImageWindow:
    """
    Factory function to create an appropriate image window for the current platform.
    
    Args:
        backend: Optional explicit backend ('tk' or 'opencv')
        **kwargs: Keyword arguments passed to the window constructor
        
    Returns:
        AbstractImageWindow: Platform-specific image window instance
    """
    WindowClass = get_image_window_class(backend=backend)
    return WindowClass(**kwargs)


def create_window_row(
        titles: list[str],
        width: int = 512,
        height: int = 512,
        x_pos: int = 100,
        y_pos: int = 100,
        **kwargs) -> list[AbstractImageWindow]:
    """
    Create N equally sized windows laid out left-to-right in one row.

    Args:
        titles: Window titles, in left-to-right display order
        width: Width for every window
        height: Height for every window
        x_pos: X position of the leftmost window
        y_pos: Y position of the row
        **kwargs: Additional keyword arguments passed to window constructors

    Returns:
        List of windows, in the same order as ``titles``
    """
    return [
        create_image_window(
            title=title,
            width=width,
            height=height,
            x_pos=x_pos + i * width,
            y_pos=y_pos,
            **kwargs
        )
        for i, title in enumerate(titles)
    ]


def create_dual_window_setup(
        main_title: str = "Main Window",
        guidance_title: str = "Guidance Window",
        width: int = 512,
        height: int = 512,
        x_pos: int = 100,
        y_pos: int = 100,
        **kwargs) -> tuple[AbstractImageWindow, AbstractImageWindow]:
    """
    Create a dual window setup similar to the existing main.py implementation.

    Args:
        main_title: Title for the main window
        guidance_title: Title for the guidance window
        width: Width for both windows
        height: Height for both windows
        x_pos: Starting x position
        y_pos: Starting y position
        **kwargs: Additional keyword arguments passed to window constructors

    Returns:
        Tuple of (main_window, guidance_window)
    """
    main_window, guidance_window = create_window_row(
        [main_title, guidance_title],
        width=width,
        height=height,
        x_pos=x_pos,
        y_pos=y_pos,
        **kwargs
    )
    return main_window, guidance_window


class WindowManager:
    """
    Helper class to manage multiple image windows.
    Useful for applications that need to coordinate multiple windows.
    """
    
    def __init__(self):
        self.windows: Dict[str, AbstractImageWindow] = {}
        
    def create_window(self, name: str, **kwargs) -> AbstractImageWindow:
        """
        Create and register a new window.
        
        Args:
            name: Unique name for the window
            **kwargs: Arguments passed to window constructor
            
        Returns:
            AbstractImageWindow: Created window instance
        """
        if name in self.windows:
            raise ValueError(f"Window with name '{name}' already exists")
            
        window = create_image_window(**kwargs)
        self.windows[name] = window
        return window
    
    def get_window(self, name: str) -> AbstractImageWindow:
        """
        Get a window by name.
        
        Args:
            name: Window name
            
        Returns:
            AbstractImageWindow: Window instance
            
        Raises:
            KeyError: If window name doesn't exist
        """
        return self.windows[name]
    
    def initialize_all(self) -> bool:
        """
        Initialize all registered windows.
        
        Returns:
            bool: True if all windows initialized successfully
        """
        success = True
        for name, window in self.windows.items():
            if not window.initialize():
                print(f"Failed to initialize window '{name}'")
                success = False
        return success
    
    def update_all(self) -> bool:
        """
        Update all active windows.
        
        Returns:
            bool: True if all windows are still active
        """
        active_windows = {}
        for name, window in self.windows.items():
            if window.update():
                active_windows[name] = window
            else:
                # Window has been closed, clean it up
                window.destroy()
        
        self.windows = active_windows
        return len(self.windows) > 0
    
    def destroy_all(self):
        """
        Destroy all windows and clean up resources.
        """
        for window in self.windows.values():
            window.destroy()
        self.windows.clear()
    
    def register_global_key_handlers(self, key_handlers: Dict[str, callable]):
        """
        Register the same key handlers for all windows.
        
        Args:
            key_handlers: Dictionary mapping keys to handler functions
        """
        for window in self.windows.values():
            for key, handler in key_handlers.items():
                window.register_key_handler(key, handler)
    
    def get_window_count(self) -> int:
        """
        Get the number of active windows.
        
        Returns:
            int: Number of active windows
        """
        return len(self.windows)
    
    def list_windows(self) -> list[str]:
        """
        Get a list of all window names.
        
        Returns:
            List of window names
        """
        return list(self.windows.keys())