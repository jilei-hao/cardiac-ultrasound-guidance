import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple
import threading  # NEW
from .abstract_image_window import AbstractImageWindow

# Optional cv2 import for advanced image processing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MacOSImageWindow(AbstractImageWindow):
    """
    macOS-specific image window implementation using Tkinter.
    Compatible with the existing codebase that uses Tkinter.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = None
        self.label = None
        self.current_image = None
        
    def initialize(self) -> bool:
        """
        Initialize the Tkinter window for macOS.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create the main window
            self.root = tk.Tk()
            self.root.title(self.title)
            
            # Set window size and position
            geometry = f"{self.width}x{self.height}+{self.x_pos}+{self.y_pos}"
            self.root.geometry(geometry)
            
            # Set background color
            bg_color = f"#{self.background_color[0]:02x}{self.background_color[1]:02x}{self.background_color[2]:02x}"
            self.root.configure(bg=bg_color)
            
            # Create label for image display
            self.label = Label(self.root, bg=bg_color)
            self.label.pack(expand=True, fill='both')
            
            # Bind keyboard events
            self.root.bind("<Key>", self._on_key_press)
            self.root.focus_set()  # Enable keyboard focus
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize macOS window: {e}")
            return False
    
    def destroy(self):
        """
        Clean up and destroy the Tkinter window.
        """
        if self.root is not None:
            try:
                # Drop image refs before destroying root
                if self.label is not None:
                    self.label.image = None
                self.current_image = None
                self.root.destroy()
            except:
                pass
            self.root = None
            self.label = None
        self.is_initialized = False
    
    def display_frame(self, frame: np.ndarray) -> bool:
        """
        Display a frame in the window.
        """
        if not self.is_initialized or self.label is None or self.root is None:
            return False

        # Ensure UI updates occur on the Tk main thread
        if threading.current_thread() is not threading.main_thread():
            # Make a shallow copy to avoid race conditions on the array
            fcopy = frame.copy()
            try:
                self.root.after(0, lambda: self._display_frame_on_ui(fcopy))
                return True
            except tk.TclError:
                return False
        else:
            return self._display_frame_on_ui(frame)

    def _display_frame_on_ui(self, frame: np.ndarray) -> bool:
        """
        Internal: perform the actual image conversion and widget update on the UI thread.
        """
        if not self.is_initialized or self.label is None or self.root is None:
            return False

        try:
            # Get current window dimensions
            current_width, current_height = self.get_window_size()

            # Resize frame to fit window (maintain aspect ratio via square min side)
            frame_height, frame_width = frame.shape[:2]
            display_size = min(current_width, current_height)
            display_size = max(1, display_size)  # guard against 0

            if HAS_CV2:
                resized_frame = cv2.resize(frame, (display_size, display_size), interpolation=cv2.INTER_AREA)
                if len(resized_frame.shape) == 3 and resized_frame.shape[2] == 3:
                    display_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                else:
                    display_frame = resized_frame
            else:
                if len(frame.shape) == 3:
                    pil_img = Image.fromarray(frame)
                else:
                    pil_img = Image.fromarray(frame, mode='L')
                pil_img = pil_img.resize((display_size, display_size), Image.Resampling.LANCZOS)
                display_frame = np.array(pil_img)

            # Convert to PIL Image, then ImageTk PhotoImage bound to this window's root
            img = Image.fromarray(display_frame)
            img_tk = ImageTk.PhotoImage(img, master=self.root)  # IMPORTANT: bind to the correct Tk instance

            # Update the label
            self.label.config(image=img_tk)
            self.label.image = img_tk        # keep strong reference
            self.current_image = img_tk
            return True

        except tk.TclError:
            # Root may be closed or image context invalid
            return False
        except Exception as e:
            print(f"Failed to display frame: {e}")
            return False
    
    def update(self) -> bool:
        """
        Update the window and process events.
        
        Returns:
            bool: True if window is still active
        """
        if not self.is_initialized or self.root is None:
            return False
            
        try:
            self.root.update_idletasks()
            self.root.update()
            return True
        except tk.TclError:
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
        
        if self.is_initialized and self.root is not None:
            geometry = f"{width}x{height}+{self.x_pos}+{self.y_pos}"
            self.root.geometry(geometry)
    
    def set_window_position(self, x: int, y: int):
        """
        Set the window position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.x_pos = x
        self.y_pos = y
        
        if self.is_initialized and self.root is not None:
            geometry = f"{self.width}x{self.height}+{x}+{y}"
            self.root.geometry(geometry)
    
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get current window size.
        
        Returns:
            Tuple of (width, height)
        """
        if self.is_initialized and self.root is not None:
            try:
                geometry = self.root.geometry()
                size_part = geometry.split('+')[0]  # Get the "WIDTHxHEIGHT" part
                width, height = map(int, size_part.split('x'))
                return width, height
            except:
                pass
        
        return self.width, self.height
    
    def set_title(self, title: str):
        """
        Set the window title.
        
        Args:
            title: New window title
        """
        self.title = title
        if self.is_initialized and self.root is not None:
            self.root.title(title)
    
    def set_resizable(self, resizable: bool):
        """
        Set whether the window can be resized.
        
        Args:
            resizable: True to allow resizing, False otherwise
        """
        if self.is_initialized and self.root is not None:
            self.root.resizable(resizable, resizable)
    
    def set_always_on_top(self, on_top: bool):
        """
        Set whether the window should stay on top.
        
        Args:
            on_top: True to keep on top, False otherwise
        """
        if self.is_initialized and self.root is not None:
            self.root.attributes('-topmost', on_top)
    
    def is_window_active(self) -> bool:
        """
        Check if the window is still active/open.
        
        Returns:
            bool: True if window is active
        """
        if not self.is_initialized or self.root is None:
            return False
            
        try:
            # Try to get window state - will raise exception if window is closed
            self.root.state()
            return True
        except tk.TclError:
            return False
        except:
            return False
    
    def _on_key_press(self, event):
        """
        Internal handler for key press events.
        
        Args:
            event: Tkinter key event
        """
        key = event.keysym
        self._handle_key_event(key)
    
    def create_child_window(self, 
                           title: str = "Child Window",
                           width: int = 512, 
                           height: int = 512,
                           x_offset: int = 50,
                           y_offset: int = 50) -> 'MacOSImageWindow':
        """
        Create a child window (Toplevel) associated with this window.
        Useful for creating guidance windows or additional displays.
        
        Args:
            title: Child window title
            width: Child window width
            height: Child window height
            x_offset: X offset from parent window
            y_offset: Y offset from parent window
            
        Returns:
            MacOSImageWindow: New child window instance
        """
        if not self.is_initialized:
            raise RuntimeError("Parent window must be initialized first")
        
        child = MacOSImageWindow(
            title=title,
            width=width,
            height=height,
            x_pos=self.x_pos + x_offset,
            y_pos=self.y_pos + y_offset,
            background_color=self.background_color
        )
        
        # Override the root creation to use Toplevel
        child.root = tk.Toplevel(self.root)
        child.root.title(title)
        
        geometry = f"{width}x{height}+{child.x_pos}+{child.y_pos}"
        child.root.geometry(geometry)
        
        # Set background color
        bg_color = f"#{child.background_color[0]:02x}{child.background_color[1]:02x}{child.background_color[2]:02x}"
        child.root.configure(bg=bg_color)
        
        # Create label for image display
        child.label = Label(child.root, bg=bg_color)
        child.label.pack(expand=True, fill='both')
        
        # Bind keyboard events
        child.root.bind("<Key>", child._on_key_press)
        child.root.focus_set()
        
        child.is_initialized = True
        return child