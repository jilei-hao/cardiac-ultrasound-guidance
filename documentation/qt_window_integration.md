# Qt Window Integration Summary

## What Changed

The image window system now uses **QtImageWindow** (PySide6/PyQt6) as the default cross-platform implementation instead of platform-specific Tkinter or OpenCV windows.

## Files Modified

1. **`src/ui/image_window/qt_image_window.py`** - Created
   - Full QtImageWindow implementation using PySide6/PyQt6
   - Implements all AbstractImageWindow methods
   - Cross-platform: Linux, macOS, Windows

2. **`src/ui/image_window/window_factory.py`** - Updated
   - Now prioritizes QtImageWindow as default
   - Falls back to platform-specific implementations if Qt unavailable
   - Supports explicit backend selection via `backend='qt'` parameter

3. **`src/ui/image_window/__init__.py`** - Updated
   - Exports QtImageWindow class
   - Exports QT_AVAILABLE flag for runtime checks

4. **`requirements.txt`** - Updated
   - Added PySide6 dependency

5. **`tests/test_qt_window.py`** - Created
   - Comprehensive Qt window tests
   - Dual window support tests
   - Animation and keyboard handling demos

6. **`tests/verify_qt_integration.py`** - Created
   - Quick verification script
   - Confirms Qt is being used by default

## Benefits

✅ **Cross-platform**: Works on macOS, Linux, and Windows without platform-specific code
✅ **No Tkinter issues**: Avoids the macOS Tk 8.6 / macOS 15.x compatibility crash
✅ **Modern**: Uses Qt6, actively maintained and widely supported
✅ **Backward compatible**: Existing code using `create_dual_window_setup()` works unchanged
✅ **Flexible**: Can still force platform-specific backends if needed

## Usage

### Default (Qt windows, recommended)
```python
from ui.image_window import create_dual_window_setup

main_window, guidance_window = create_dual_window_setup(
    main_title="Main View",
    guidance_title="Guidance View"
)
```

### Explicit backend selection
```python
from ui.image_window import create_image_window

# Force Qt
qt_window = create_image_window(backend='qt', title="Qt Window")

# Force Tkinter (macOS)
tk_window = create_image_window(backend='tk', title="Tk Window")

# Force OpenCV (Linux)
cv_window = create_image_window(backend='opencv', title="OpenCV Window")
```

## Testing

### Run Qt window tests
```bash
conda activate vigil
python tests/test_qt_window.py
```

### Verify integration
```bash
conda activate vigil
python tests/verify_qt_integration.py
```

### Run guidance tests (now using Qt)
```bash
conda activate vigil
python -m tests.test_guidance frames /path/to/frames
```

## No Changes Required

**`test_guidance.py` requires NO modifications** - it already uses `create_dual_window_setup()` which now automatically returns Qt windows.

All existing code continues to work as before, but now uses the robust Qt implementation instead of Tkinter.
