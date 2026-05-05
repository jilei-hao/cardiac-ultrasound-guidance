#!/usr/bin/env python3
"""
Simple test for the AbstractImageWindow system using only basic functionality.
This test focuses on the macOS Tkinter implementation.
"""

import sys
import os
import time
import numpy as np

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.image_window import create_image_window
from data_access.simulator import FrameDataSimulator


def test_basic_window():
    """Test basic window functionality without OpenCV dependencies."""
    print("=== Basic Window Test ===")

    frames_folder = os.environ.get("CUSG_TEST_FRAMES_FOLDER")
    if not frames_folder or not os.path.isdir(frames_folder):
        pytest.skip(
            "Set CUSG_TEST_FRAMES_FOLDER to a frames folder to run this "
            "interactive demo."
        )
    
    # Create window
    window = create_image_window(
        title="Basic Test Window",
        width=400,
        height=400,
        x_pos=100,
        y_pos=100
    )
    
    # Set up a simple key handler
    def handle_quit():
        print("Quit key pressed!")
        window.destroy()
    
    window.register_key_handler('q', handle_quit)
    
    # Initialize window
    if not window.initialize():
        print("❌ Failed to initialize window")
        return
    
    print("✅ Window initialized successfully")
    print("Window info:", window.get_display_info())
    
    # Create simulator
    simulator = FrameDataSimulator(frames_folder=frames_folder)
    
    try:
        simulator.start()
        
        frame_count = 0
        print("Starting display loop. Press 'q' to quit, or close the window.")
        
        while window.is_window_active():
            # Get frame from simulator
            frame_data = simulator.get_frame()
            
            if frame_data.image is None:
                # End of sequence, reset
                simulator.reset()
                continue
            
            # Convert PIL image to numpy array (RGB format)
            frame = np.array(frame_data.image)
            
            # For macOS window, we need RGB format (will be converted internally)
            # The window will handle the conversion to the appropriate format
            
            # Display frame
            if not window.display_frame(frame):
                print("❌ Failed to display frame")
                break
            
            # Update window
            if not window.update():
                print("🛑 Window closed")
                break
            
            frame_count += 1
            
            # Limit to ~10 FPS for testing
            time.sleep(0.1)
            
            # Show progress every 10 frames
            if frame_count % 10 == 0:
                print(f"Displayed {frame_count} frames")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulator.stop()
        window.destroy()
        print("✅ Test completed")


def test_window_properties():
    """Test window property management."""
    print("\n=== Window Properties Test ===")
    
    window = create_image_window(
        title="Property Test",
        width=300,
        height=300,
        x_pos=200,
        y_pos=200
    )
    
    if not window.initialize():
        print("❌ Failed to initialize window")
        return
    
    try:
        # Test property getters/setters
        print(f"Initial size: {window.get_window_size()}")
        
        # Test resizing
        window.set_window_size(400, 400)
        print(f"After resize: {window.get_window_size()}")
        
        # Test repositioning
        window.set_window_position(300, 300)
        
        # Test title change
        window.set_title("Updated Title")
        
        # Test resizable setting
        window.set_resizable(True)
        
        # Show window for a few seconds
        for i in range(30):  # 3 seconds at 10 FPS
            if not window.update():
                break
            time.sleep(0.1)
        
        print("✅ Window properties test completed")
        
    except Exception as e:
        print(f"❌ Error in properties test: {e}")
        
    finally:
        window.destroy()


def main():
    """Run basic tests."""
    print("AbstractImageWindow Basic Tests")
    print("=" * 40)
    
    try:
        # Test basic window functionality
        test_basic_window()
        
        time.sleep(1)
        
        # Test window properties
        test_window_properties()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Basic tests completed!")


if __name__ == "__main__":
    main()