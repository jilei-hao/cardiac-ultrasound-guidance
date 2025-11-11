#!/usr/bin/env python3
"""
Example usage of the FrameDataInput classes
"""

from data_access.simulator import FrameDataSimulator
from data_access.clarius import ClariusStreamer  
from data_access.screen_capture import ScreenCapture
import time


def example_simulator_usage():
    """Example of using the FrameDataSimulator"""
    print("=== FrameDataSimulator Example ===")
    
    # Create simulator instance
    # Note: Replace with actual paths to your PNG frames and IMU CSV file
    simulator = FrameDataSimulator(
        frames_folder="./output/sample_frames",  # Folder with PNG files
        imu_csv_file="./output/sample_imu.csv"   # CSV with IMU data (optional)
    )
    
    try:
        # Start the simulator
        simulator.start()
        
        # Get some frames
        for i in range(5):
            frame = simulator.get_frame()
            if frame.image is not None:
                print(f"Frame {i}: timestamp={frame.timestamp}, has_imu={frame.imu is not None}")
                if frame.imu:
                    print(f"  IMU data keys: {list(frame.imu.keys())}")
            else:
                print(f"Frame {i}: No more frames available")
                break
            
        # Show simulator info
        print(f"Total frames: {simulator.get_total_frames()}")
        print(f"Current index: {simulator.get_current_frame_index()}")
        print(f"Has IMU data: {simulator.has_imu_data()}")
        
    finally:
        # Always stop the simulator
        simulator.stop()


def example_clarius_usage():
    """Example of using the ClariusStreamer (commented out as it requires actual hardware)"""
    print("=== ClariusStreamer Example (Hardware Required) ===")
    
    # Uncomment and modify these lines if you have Clarius hardware
    # clarius = ClariusStreamer(
    #     ip="192.168.1.1",
    #     port=5828, 
    #     path="research"
    # )
    # 
    # try:
    #     clarius.start()
    #     
    #     # Get frames in a loop
    #     for i in range(10):
    #         frame = clarius.get_frame()
    #         if frame.image is not None:
    #             print(f"Clarius frame: {frame.timestamp}")
    #         time.sleep(0.1)
    #         
    # finally:
    #     clarius.stop()
    
    print("ClariusStreamer example skipped (requires hardware)")


def example_screen_capture_usage():
    """Example of using the ScreenCapture"""
    print("=== ScreenCapture Example ===")
    
    # Note: Replace with an actual window title on your system
    # You can get window titles using: osascript -e 'tell application "System Events" to get name of every process'
    screen_capture = ScreenCapture(
        window_title="Finder",  # Change this to an actual window
        app_settings=None       # You would pass actual app settings here
    )
    
    # Uncomment to test (requires proper app_settings)
    # try:
    #     screen_capture.start()
    #     
    #     frame = screen_capture.get_frame()
    #     if frame.image is not None:
    #         print(f"Screen capture frame: timestamp={frame.timestamp}")
    #     else:
    #         print("Failed to capture screen")
    #         
    # finally:
    #     screen_capture.stop()
    
    print("ScreenCapture example skipped (requires proper app_settings)")


if __name__ == "__main__":
    print("FrameDataInput Examples")
    print("=" * 40)
    
    # Run the simulator example
    # example_simulator_usage()
    
    # Run other examples
    example_clarius_usage() 
    example_screen_capture_usage()
    
    print("\nNote: Uncomment example_simulator_usage() and provide actual paths to test the simulator")