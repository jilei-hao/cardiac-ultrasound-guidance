#!/usr/bin/env python3
"""
Demo script showing how to use the new AbstractImageWindow system.
This demonstrates replacing the window logic from main.py with the new system.
"""

import sys
import os
import time
import numpy as np
import cv2

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.image_window import create_image_window, create_dual_window_setup, WindowManager
from data_access.simulator import FrameDataSimulator


def demo_single_window():
    """Demo with a single window displaying frames from the simulator."""
    print("=== Single Window Demo ===")
    
    # Create window
    window = create_image_window(
        title="Ultrasound Display",
        width=600,
        height=600,
        x_pos=100,
        y_pos=100
    )
    
    # Set up keyboard handlers (similar to main.py)
    def handle_fps_toggle():
        print("FPS toggle pressed!")
    
    def handle_debug_toggle():
        print("Debug toggle pressed!")
    
    def handle_radius_increase():
        print("Landmark radius increased!")
    
    def handle_radius_decrease():
        print("Landmark radius decreased!")
    
    # Register key handlers
    window.register_key_handler('f', handle_fps_toggle)
    window.register_key_handler('d', handle_debug_toggle)
    window.register_key_handler('Up', handle_radius_increase)
    window.register_key_handler('Down', handle_radius_decrease)
    
    # Initialize window
    if not window.initialize():
        print("Failed to initialize window")
        return
    
    # Create simulator
    simulator = FrameDataSimulator(
        frames_folder="/Users/jileihao/dev/arpa-h-dev/cardiac-ultrasound-guidance/output/20250614-012501_155563293680"
    )
    
    try:
        simulator.start()
        
        frame_count = 0
        start_time = time.time()
        
        print("Starting display loop. Press 'q' to quit, or try other keys (f, d, Up, Down)")
        
        while window.is_window_active():
            # Get frame from simulator
            frame_data = simulator.get_frame()
            
            if frame_data.image is None:
                # End of sequence, reset
                simulator.reset()
                continue
            
            # Convert PIL image to numpy array
            frame = np.array(frame_data.image)
            
            # Add some processing (similar to main.py)
            if len(frame.shape) == 3:
                # Convert RGB to BGR for display
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Convert grayscale to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Add frame counter text
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            if not window.display_frame(frame):
                print("Failed to display frame")
                break
            
            # Update window
            if not window.update():
                print("Window closed")
                break
            
            frame_count += 1
            
            # Limit to ~30 FPS
            time.sleep(0.033)
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}")
    
    finally:
        simulator.stop()
        window.destroy()


def demo_dual_window():
    """Demo with dual windows (like main.py with guidance window)."""
    print("\n=== Dual Window Demo ===")
    
    # Create dual window setup
    main_window, guidance_window = create_dual_window_setup(
        main_title="Results",
        guidance_title="Guidance",
        width=512,
        height=512,
        x_pos=150,
        y_pos=150
    )
    
    # Initialize both windows
    if not main_window.initialize():
        print("Failed to initialize main window")
        return
    
    if not guidance_window.initialize():
        print("Failed to initialize guidance window")
        main_window.destroy()
        return
    
    # Create simulator
    simulator = FrameDataSimulator(
        frames_folder="/Users/jileihao/dev/arpa-h-dev/cardiac-ultrasound-guidance/output/20250614-012501_155563293680"
    )
    
    try:
        simulator.start()
        
        frame_count = 0
        print("Starting dual window display. Close either window to exit.")
        
        while main_window.is_window_active() and guidance_window.is_window_active():
            # Get frame from simulator
            frame_data = simulator.get_frame()
            
            if frame_data.image is None:
                simulator.reset()
                continue
            
            # Convert PIL image to numpy array
            original_frame = np.array(frame_data.image)
            
            # Process main window frame
            main_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(main_frame, "Main View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process guidance window frame (simulate different processing)
            guidance_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            # Add some visual difference (e.g., edge detection)
            edges = cv2.Canny(cv2.cvtColor(guidance_frame, cv2.COLOR_BGR2GRAY), 50, 150)
            guidance_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(guidance_frame, "Guidance View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frames
            main_window.display_frame(main_frame)
            guidance_window.display_frame(guidance_frame)
            
            # Update windows
            if not main_window.update() or not guidance_window.update():
                break
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
    
    finally:
        simulator.stop()
        main_window.destroy()
        guidance_window.destroy()


def demo_window_manager():
    """Demo using the WindowManager for managing multiple windows."""
    print("\n=== Window Manager Demo ===")
    
    manager = WindowManager()
    
    # Create multiple windows
    manager.create_window("main", title="Main Window", width=400, height=400, x_pos=100, y_pos=100)
    manager.create_window("secondary", title="Secondary Window", width=300, height=300, x_pos=550, y_pos=100)
    manager.create_window("monitor", title="Monitor Window", width=300, height=200, x_pos=100, y_pos=550)
    
    # Set up global key handlers
    def quit_all():
        print("Quit key pressed - closing all windows")
        manager.destroy_all()
    
    def print_info():
        print(f"Active windows: {manager.get_window_count()}")
        print(f"Window names: {manager.list_windows()}")
    
    manager.register_global_key_handlers({
        'q': quit_all,
        'i': print_info
    })
    
    # Initialize all windows
    if not manager.initialize_all():
        print("Failed to initialize some windows")
        manager.destroy_all()
        return
    
    try:
        frame_count = 0
        print("Window Manager Demo running. Press 'i' for info, 'q' to quit all.")
        
        while manager.update_all():  # This updates all windows and removes closed ones
            # Create different content for each window
            for i, name in enumerate(manager.list_windows()):
                window = manager.get_window(name)
                
                # Create a simple colored frame with text
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
                frame = np.full((200, 200, 3), color, dtype=np.uint8)
                
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                window.display_frame(frame)
            
            frame_count += 1
            time.sleep(0.05)  # 20 FPS
    
    finally:
        manager.destroy_all()


def main():
    """Run all demos."""
    print("AbstractImageWindow Demos")
    print("=" * 40)
    
    try:
        # Run single window demo
        demo_single_window()
        
        time.sleep(1)  # Brief pause between demos
        
        # Run dual window demo
        demo_dual_window()
        
        time.sleep(1)  # Brief pause between demos
        
        # Run window manager demo
        demo_window_manager()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll demos completed!")


if __name__ == "__main__":
    main()