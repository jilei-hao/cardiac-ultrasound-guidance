#!/usr/bin/env python3
"""
Simple test for QtImageWindow to verify cross-platform functionality.
"""
import sys
import os
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.image_window.qt_image_window import QtImageWindow


def test_basic_window():
    """Test basic window creation and display."""
    print("Testing QtImageWindow...")
    
    # Create window
    window = QtImageWindow(
        title="Qt Test Window",
        width=512,
        height=512,
        x_pos=100,
        y_pos=100
    )
    
    # Initialize
    if not window.initialize():
        print("Failed to initialize window")
        return False
    
    print("Window initialized successfully")
    
    # Create test pattern
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Red square
    test_image[50:150, 50:150] = [255, 0, 0]
    # Green square
    test_image[50:150, 250:350] = [0, 255, 0]
    # Blue square
    test_image[250:350, 50:150] = [0, 0, 255]
    # White square
    test_image[250:350, 250:350] = [255, 255, 255]
    
    # Display initial frame
    if not window.display_frame_rgb(test_image):
        print("Failed to display frame")
        return False
    
    print("Displayed test pattern")
    
    # Register key handler
    def on_escape():
        print("Escape pressed - closing window")
        window.destroy()
    
    window.register_key_handler('Escape', on_escape)
    window.register_key_handler('q', on_escape)
    
    print("Press 'Escape' or 'q' to close the window")
    
    # Animation loop
    frame_count = 0
    start_time = time.time()
    
    while window.is_window_active():
        # Create animated frame (rotating colors)
        angle = (frame_count * 5) % 360
        hue = angle / 360.0
        
        # Simple color animation
        r = int(255 * abs(np.sin(hue * np.pi)))
        g = int(255 * abs(np.sin((hue + 0.33) * np.pi)))
        b = int(255 * abs(np.sin((hue + 0.66) * np.pi)))
        
        animated_image = test_image.copy()
        animated_image[150:250, 150:250] = [r, g, b]
        
        # Display frame
        window.display_frame_rgb(animated_image)
        
        # Update window
        if not window.update():
            break
        
        frame_count += 1
        
        # Limit frame rate
        time.sleep(0.033)  # ~30 FPS
        
        # Auto-close after 10 seconds for automated testing
        if time.time() - start_time > 10:
            print("Auto-closing after 10 seconds")
            break
    
    # Cleanup
    window.destroy()
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Test completed: {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
    
    return True


def test_dual_windows():
    """Test creating multiple windows."""
    print("\nTesting dual windows...")
    
    window1 = QtImageWindow(title="Window 1", width=400, height=400, x_pos=100, y_pos=100)
    window2 = QtImageWindow(title="Window 2", width=400, height=400, x_pos=550, y_pos=100)
    
    if not window1.initialize() or not window2.initialize():
        print("Failed to initialize windows")
        return False
    
    # Different test patterns
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1[:, :] = [255, 100, 100]  # Red-ish
    
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    img2[:, :] = [100, 100, 255]  # Blue-ish
    
    window1.display_frame_rgb(img1)
    window2.display_frame_rgb(img2)
    
    print("Displaying dual windows for 3 seconds...")
    
    start = time.time()
    while time.time() - start < 3:
        if not window1.update() or not window2.update():
            break
        time.sleep(0.033)
    
    window1.destroy()
    window2.destroy()
    
    print("Dual window test completed")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Qt Image Window Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic window
        if not test_basic_window():
            print("Basic window test FAILED")
            return 1
        
        # Test 2: Dual windows
        if not test_dual_windows():
            print("Dual window test FAILED")
            return 1
        
        print("\n" + "=" * 50)
        print("All tests PASSED")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
