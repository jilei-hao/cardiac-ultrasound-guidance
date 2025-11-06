#!/usr/bin/env python3
"""
Test script for FrameDataSimulator to verify it works with existing data
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_access.simulator.frame_data_simulator import FrameDataSimulator


def test_simulator():
    """Test the FrameDataSimulator with existing data"""
    print("Testing FrameDataSimulator...")
    
    # Use the existing data
    frames_folder = "/Users/jileihao/dev/arpa-h-dev/cardiac-ultrasound-guidance/output/20250614-012501_155563293680"
    imu_csv_file = "/Users/jileihao/dev/arpa-h-dev/cardiac-ultrasound-guidance/output/20250614-012501_155563293680/frame.imu.csv"
    
    print(f"Frames folder: {frames_folder}")
    print(f"IMU CSV file: {imu_csv_file}")
    
    # Create simulator instance
    simulator = FrameDataSimulator(frames_folder, imu_csv_file)
    
    try:
        # Start the simulator
        print("\nStarting simulator...")
        simulator.start()
        
        print(f"Total frames available: {simulator.get_total_frames()}")
        print(f"Has IMU data: {simulator.has_imu_data()}")
        
        # Get first few frames
        print("\nTesting frame retrieval:")
        for i in range(5):
            frame = simulator.get_frame()
            if frame.image is not None:
                print(f"Frame {i}:")
                print(f"  - Image size: {frame.image.size}")
                print(f"  - Timestamp: {frame.timestamp}")
                print(f"  - Has IMU: {frame.imu is not None}")
                if frame.imu:
                    print(f"  - IMU keys: {list(frame.imu.keys())[:5]}...")  # Show first 5 keys
                    print(f"  - IMU timestamp: {frame.imu.get('timestamp', 'N/A')}")
            else:
                print(f"Frame {i}: No image data")
                break
                
        # Test current position
        print(f"\nCurrent frame index: {simulator.get_current_frame_index()}")
        
        # Test reset
        print("\nTesting reset...")
        simulator.reset()
        print(f"After reset, current index: {simulator.get_current_frame_index()}")
        
        # Get one more frame after reset
        frame = simulator.get_frame()
        if frame.image is not None:
            print(f"First frame after reset: image size = {frame.image.size}")
        
        print("\n✅ FrameDataSimulator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always stop the simulator
        simulator.stop()


if __name__ == "__main__":
    test_simulator()