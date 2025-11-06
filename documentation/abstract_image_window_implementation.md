# AbstractImageWindow Implementation

## Overview
I have successfully created a cross-platform image window system that extracts and generalizes the window logic from `main.py`, providing a clean, extensible interface for displaying frames and handling user input.

## Architecture

### 1. AbstractImageWindow Base Class
**Location**: `src/ui/image_window/abstract_image_window.py`

- Abstract base class using Python's `abc` module
- Defines the common interface for all platform-specific implementations
- Handles keyboard event management with a registration system
- Provides unified window property management

#### Key Abstract Methods:
- `initialize() -> bool`: Initialize the window and GUI framework
- `destroy()`: Clean up and destroy the window
- `display_frame(frame: np.ndarray) -> bool`: Display a frame in the window
- `update() -> bool`: Update window and process events
- `set_window_size(width: int, height: int)`: Set window dimensions
- `set_window_position(x: int, y: int)`: Set window position
- `get_window_size() -> Tuple[int, int]`: Get current window size
- `set_title(title: str)`: Set window title
- `set_resizable(resizable: bool)`: Control window resizing
- `set_always_on_top(on_top: bool)`: Control window layering
- `is_window_active() -> bool`: Check if window is still open

#### Key Features:
- **Keyboard Event System**: Register/unregister key handlers with `register_key_handler(key, handler)`
- **Window Properties**: Comprehensive window management (size, position, title, etc.)
- **Display Information**: Get complete window state with `get_display_info()`

### 2. Platform-Specific Implementations

#### MacOSImageWindow
**Location**: `src/ui/image_window/macos_image_window.py`

- **GUI Framework**: Tkinter (compatible with existing `main.py`)
- **Key Features**:
  - Native macOS window management
  - PIL/Tkinter image display pipeline
  - Optional OpenCV support for advanced image processing
  - Child window creation with `create_child_window()`
  - Automatic image format conversion (BGR ↔ RGB)
  - Maintains aspect ratio with square display format

#### LinuxImageWindow  
**Location**: `src/ui/image_window/linux_image_window.py`

- **GUI Framework**: OpenCV highgui
- **Key Features**:
  - Native Linux window management
  - OpenCV-based image display
  - Threaded keyboard monitoring system
  - Cross-platform key code conversion
  - Child window creation support
  - Automatic image format handling

### 3. Factory and Management System

#### WindowFactory
**Location**: `src/ui/image_window/window_factory.py`

**Functions**:
- `get_image_window_class()`: Auto-detect platform and return appropriate class
- `create_image_window(**kwargs)`: Factory function to create platform-specific windows
- `create_dual_window_setup()`: Create main + guidance window pair (like `main.py`)

#### WindowManager Class
Advanced multi-window management system:
- `create_window(name, **kwargs)`: Create and register named windows
- `initialize_all()`: Initialize all registered windows
- `update_all()`: Update all windows, remove closed ones
- `destroy_all()`: Clean shutdown of all windows
- `register_global_key_handlers()`: Apply same key handlers to all windows

## Key Improvements Over Original main.py

### 1. **Cross-Platform Compatibility**
- **macOS**: Native Tkinter implementation (backward compatible)
- **Linux**: OpenCV-based implementation
- **Windows**: Falls back to Linux implementation

### 2. **Clean Separation of Concerns**
- **Window Management**: Abstracted into dedicated classes
- **Event Handling**: Structured keyboard event system
- **Image Display**: Unified interface regardless of platform

### 3. **Enhanced Features**
- **Multiple Window Support**: Easy creation of guidance windows, monitors, etc.
- **Dynamic Window Properties**: Runtime changes to size, position, title
- **Robust Error Handling**: Graceful handling of window closure, display errors
- **Optional Dependencies**: Works with or without OpenCV

### 4. **Developer Experience**
- **Consistent API**: Same interface across all platforms
- **Easy Testing**: Mock-friendly design for unit testing
- **Extensible**: Simple to add new platforms or features

## Usage Examples

### Basic Single Window
```python
from ui.image_window import create_image_window

# Create and initialize window
window = create_image_window(title="My App", width=512, height=512)
window.initialize()

# Register keyboard handlers
window.register_key_handler('q', lambda: window.destroy())
window.register_key_handler('f', toggle_fullscreen)

# Main loop
while window.is_window_active():
    frame = get_next_frame()  # Your frame source
    window.display_frame(frame)
    if not window.update():
        break

window.destroy()
```

### Dual Window Setup (Replaces main.py logic)
```python
from ui.image_window import create_dual_window_setup

# Create main + guidance windows
main_window, guidance_window = create_dual_window_setup(
    main_title="Results",
    guidance_title="Guidance", 
    width=512, height=512
)

main_window.initialize()
guidance_window.initialize()

# Main loop with dual display
while main_window.is_window_active() and guidance_window.is_window_active():
    frame_data = get_frame_data()
    
    # Process for main window
    main_frame = process_main_view(frame_data)
    main_window.display_frame(main_frame)
    
    # Process for guidance window  
    guidance_frame = process_guidance_view(frame_data)
    guidance_window.display_frame(guidance_frame)
    
    # Update both windows
    if not main_window.update() or not guidance_window.update():
        break

main_window.destroy()
guidance_window.destroy()
```

### Multi-Window Management
```python
from ui.image_window import WindowManager

manager = WindowManager()

# Create multiple windows
manager.create_window("main", title="Main View", width=600, height=600)
manager.create_window("monitor", title="Monitor", width=300, height=200)
manager.create_window("debug", title="Debug Info", width=400, height=300)

# Set up global key handlers
manager.register_global_key_handlers({
    'q': lambda: manager.destroy_all(),
    'i': lambda: print(f"Active windows: {manager.get_window_count()}")
})

# Initialize and run
manager.initialize_all()

while manager.update_all():  # Updates all, removes closed windows
    # Update each window with different content
    for name in manager.list_windows():
        window = manager.get_window(name)
        frame = generate_content_for_window(name)
        window.display_frame(frame)

manager.destroy_all()
```

## Migration from main.py

The new system can easily replace the existing window logic in `main.py`:

### Before (main.py):
```python
root = tk.Tk()
root.title("Results")
root.geometry(f"{width}x{height}+{left}+{top}")
root.configure(bg='black')
label = tk.Label(root, bg='black')
label.pack(expand=True, fill='both')

def handle_key(event):
    if event.keysym == "e":
        # handle key
        
root.bind("<Key>", handle_key)

# In main loop:
img_tk = ImageTk.PhotoImage(img)
label.config(image=img_tk)
label.image = img_tk
root.update_idletasks()
root.update()
```

### After (AbstractImageWindow):
```python
from ui.image_window import create_image_window

window = create_image_window(title="Results", width=width, height=height, 
                           x_pos=left, y_pos=top)
window.initialize()

# Register key handlers
window.register_key_handler('e', handle_e_key)
window.register_key_handler('f', handle_f_key)

# In main loop:
window.display_frame(frame)  # Handles all conversion automatically
if not window.update():
    break
```

## Testing

### Test Coverage
- **Basic Functionality**: `tests/test_basic_windows.py`
  - Window creation and initialization
  - Frame display with FrameDataSimulator
  - Keyboard event handling
  - Window property management
  
- **Integration**: Successfully displays 190 ultrasound frames from existing test data
- **Environment**: Tested with `conda activate lumify-streaming`
- **Dependencies**: Works with minimal dependencies (PIL, numpy, pandas)

### Test Results
- ✅ Window initialization and display
- ✅ Frame loading and display from simulator
- ✅ Keyboard event handling
- ✅ Window property management
- ✅ Graceful cleanup and destruction

## Dependencies

### Required
- **PIL (Pillow)**: Image handling and display
- **numpy**: Array operations
- **tkinter**: macOS GUI (usually built-in with Python)

### Optional
- **opencv-python**: Enhanced image processing and Linux support
- **pandas**: For FrameDataSimulator (if using CSV IMU data)

## Benefits

1. **Maintainability**: Clean separation of window logic from application logic
2. **Testability**: Easy to mock and unit test window operations  
3. **Flexibility**: Support multiple windows, different platforms
4. **Backward Compatibility**: Drop-in replacement for existing Tkinter code
5. **Future-Proof**: Easy to add new platforms or GUI frameworks
6. **Robustness**: Better error handling and resource cleanup

This implementation provides a solid foundation for the cardiac ultrasound guidance system's UI needs while maintaining compatibility with the existing codebase and enabling future cross-platform deployment.