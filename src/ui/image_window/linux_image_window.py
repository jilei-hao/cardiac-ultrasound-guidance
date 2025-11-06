import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
import threading
from .abstract_image_window import AbstractImageWindow


class LinuxImageWindow(AbstractImageWindow):
    """
    Linux (Ubuntu) image window implementation using OpenCV's highgui.
    Provides cross-platform compatibility for Linux systems.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window_name = None
        self.current_frame = None
        self.key_check_thread = None
        self.should_stop = False
        self._window_flags = cv2.WINDOW_NORMAL  # Allow resizing by default
        
    def initialize(self) -> bool:
        """
        Initialize the OpenCV window for Linux.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Use the title as the window name
            self.window_name = self.title
            
            # Create the window
            cv2.namedWindow(self.window_name, self._window_flags)
            
            # Set window size and position
            cv2.resizeWindow(self.window_name, self.width, self.height)
            cv2.moveWindow(self.window_name, self.x_pos, self.y_pos)
            
            # Create a blank frame with background color
            blank_frame = np.full((self.height, self.width, 3), 
                                self.background_color[::-1], dtype=np.uint8)  # BGR format
            cv2.imshow(self.window_name, blank_frame)
            
            # Start the key monitoring thread
            self._start_key_monitoring()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Linux window: {e}")
            return False
    
    def destroy(self):
        """
        Clean up and destroy the OpenCV window.
        """
        self.should_stop = True
        
        # Wait for key monitoring thread to finish
        if self.key_check_thread and self.key_check_thread.is_alive():
            self.key_check_thread.join(timeout=1.0)
        
        if self.window_name is not None:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass  # Window may already be destroyed
            self.window_name = None
        
        self.is_initialized = False
    
    def display_frame(self, frame: np.ndarray) -> bool:
        """
        Display a frame in the window.
        
        Args:
            frame: NumPy array representing the image (BGR or RGB format)
            
        Returns:
            bool: True if display successful
        """
        if not self.is_initialized or self.window_name is None:
            return False
            
        try:
            # Get current window size
            current_width, current_height = self.get_window_size()
            
            # Resize frame to fit window (maintain aspect ratio)
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate the minimum dimension to maintain square aspect ratio
            display_size = min(current_width, current_height)
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (display_size, display_size), 
                                     interpolation=cv2.INTER_AREA)
            
            # Convert RGB to BGR if necessary (OpenCV uses BGR by default)
            if len(resized_frame.shape) == 3 and resized_frame.shape[2] == 3:
                # Assume input is RGB, convert to BGR for OpenCV
                display_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = resized_frame
            
            # Display the frame
            cv2.imshow(self.window_name, display_frame)
            self.current_frame = display_frame.copy()
            
            return True
            
        except Exception as e:
            print(f"Failed to display frame: {e}")
            return False
    
    def update(self) -> bool:
        """
        Update the window and process events.
        
        Returns:
            bool: True if window is still active
        """
        if not self.is_initialized or self.window_name is None:
            return False
            
        try:
            # Process OpenCV events (non-blocking)
            cv2.waitKey(1)  # 1ms timeout to keep responsive
            
            # Check if window still exists
            try:
                # Try to get window property - will fail if window is closed
                cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                return True
            except cv2.error:
                # Window has been closed
                self.is_initialized = False
                return False
                
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
        self.width = width
        self.height = height
        
        if self.is_initialized and self.window_name is not None:
            cv2.resizeWindow(self.window_name, width, height)
    
    def set_window_position(self, x: int, y: int):
        """
        Set the window position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.x_pos = x
        self.y_pos = y
        
        if self.is_initialized and self.window_name is not None:
            cv2.moveWindow(self.window_name, x, y)
    
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get current window size.
        
        Returns:
            Tuple of (width, height)
        """
        if self.is_initialized and self.window_name is not None:
            try:
                # Get window rectangle (x, y, width, height)
                rect = cv2.getWindowImageRect(self.window_name)
                return rect[2], rect[3]  # width, height
            except cv2.error:
                pass
        
        return self.width, self.height
    
    def set_title(self, title: str):
        """
        Set the window title (creates new window with new name in OpenCV).
        
        Args:
            title: New window title
        """
        if self.is_initialized and title != self.title:
            old_window_name = self.window_name
            
            # Create new window with new title
            self.title = title
            self.window_name = title
            cv2.namedWindow(self.window_name, self._window_flags)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            cv2.moveWindow(self.window_name, self.x_pos, self.y_pos)
            
            # Display current frame in new window
            if self.current_frame is not None:
                cv2.imshow(self.window_name, self.current_frame)
            
            # Destroy old window
            if old_window_name:
                cv2.destroyWindow(old_window_name)
    
    def set_resizable(self, resizable: bool):
        """
        Set whether the window can be resized.
        
        Args:
            resizable: True to allow resizing, False otherwise
        """
        new_flags = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
        
        if new_flags != self._window_flags:
            self._window_flags = new_flags
            
            # Recreate window with new flags if already initialized
            if self.is_initialized:
                old_frame = self.current_frame
                self.destroy()
                self.initialize()
                if old_frame is not None:
                    cv2.imshow(self.window_name, old_frame)
    
    def set_always_on_top(self, on_top: bool):
        """
        Set whether the window should stay on top.
        Note: This feature may not be available on all Linux window managers.
        
        Args:
            on_top: True to keep on top, False otherwise
        """
        # OpenCV doesn't have direct support for always-on-top
        # This would require platform-specific implementation using X11
        print("Warning: Always-on-top is not supported in OpenCV Linux implementation")
    
    def is_window_active(self) -> bool:
        """
        Check if the window is still active/open.
        
        Returns:
            bool: True if window is active
        """
        if not self.is_initialized or self.window_name is None:
            return False
            
        try:
            # Try to get window property - will raise exception if window is closed
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            return visible >= 0  # -1 indicates window is closed
        except cv2.error:
            return False
        except:
            return False
    
    def _start_key_monitoring(self):
        """
        Start a thread to monitor keyboard input.
        OpenCV key detection works differently than Tkinter.
        """
        def monitor_keys():
            while not self.should_stop and self.is_initialized:
                try:
                    # Check for key press (with timeout)
                    key = cv2.waitKey(30) & 0xFF  # 30ms timeout
                    
                    if key != 255:  # 255 means no key pressed
                        # Convert key code to key name
                        key_name = self._convert_key_code(key)
                        if key_name:
                            self._handle_key_event(key_name)
                            
                except:
                    break
                    
                time.sleep(0.01)  # Small delay to prevent high CPU usage
        
        self.should_stop = False
        self.key_check_thread = threading.Thread(target=monitor_keys, daemon=True)
        self.key_check_thread.start()
    
    def _convert_key_code(self, key_code: int) -> Optional[str]:
        """
        Convert OpenCV key code to string representation.
        
        Args:
            key_code: OpenCV key code
            
        Returns:
            String representation of the key, or None if not recognized
        """
        # Common key mappings
        key_map = {
            27: 'Escape',
            13: 'Return',
            8: 'BackSpace',
            9: 'Tab',
            32: 'space',
            # Arrow keys (these may vary by system)
            81: 'Left',    # or 2424832 depending on system
            82: 'Up',      # or 2490368
            83: 'Right',   # or 2555904
            84: 'Down',    # or 2621440
            # Function keys
            59: 'F1', 60: 'F2', 61: 'F3', 62: 'F4',
            63: 'F5', 64: 'F6', 65: 'F7', 66: 'F8',
            67: 'F9', 68: 'F10', 69: 'F11', 70: 'F12',
        }
        
        # Check special keys first
        if key_code in key_map:
            return key_map[key_code]
        
        # Handle extended key codes for arrow keys (Linux specific)
        if key_code == 2424832:  # Left arrow
            return 'Left'
        elif key_code == 2490368:  # Up arrow
            return 'Up'
        elif key_code == 2555904:  # Right arrow
            return 'Right'
        elif key_code == 2621440:  # Down arrow
            return 'Down'
        
        # Regular ASCII characters
        if 32 <= key_code <= 126:
            return chr(key_code).lower()
        
        return None
    
    def create_child_window(self, 
                           title: str = "Child Window",
                           width: int = 512, 
                           height: int = 512,
                           x_offset: int = 50,
                           y_offset: int = 50) -> 'LinuxImageWindow':
        """
        Create a child window associated with this window.
        
        Args:
            title: Child window title
            width: Child window width
            height: Child window height
            x_offset: X offset from parent window
            y_offset: Y offset from parent window
            
        Returns:
            LinuxImageWindow: New child window instance
        """
        child = LinuxImageWindow(
            title=title,
            width=width,
            height=height,
            x_pos=self.x_pos + x_offset,
            y_pos=self.y_pos + y_offset,
            background_color=self.background_color
        )
        
        child.initialize()
        return child