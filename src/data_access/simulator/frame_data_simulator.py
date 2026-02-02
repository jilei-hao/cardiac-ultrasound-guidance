import os
import time
import pandas as pd
from PIL import Image
from ..frame_data import FrameData, TrackerData
from ..frame_data_input import FrameDataInput
import bisect


class FrameDataSimulator(FrameDataInput):
    """
    Simulator for frame data that loads PNG images from a folder and optionally
    corresponding IMU data from a CSV file.
    """
    
    def __init__(self, frames_folder: str, imu_csv_file: str = None):
        """
        Initialize the frame data simulator.
        
        Args:
            frames_folder: Path to folder containing PNG frame images
            imu_csv_file: Path to CSV file containing IMU data (optional)
        """
        self.frames_folder = frames_folder
        if (imu_csv_file is not None) and (not os.path.exists(imu_csv_file)):
            raise FileNotFoundError(f"IMU CSV file not found: {imu_csv_file}")
        if imu_csv_file is None:
            imu_csv_file = os.path.join(frames_folder, "frame.imu.csv")
        
        self.imu_csv_file = imu_csv_file
        self.frame_files = []
        self.frame_data = None
        self.current_frame_index = 0
        self._is_started = False
        
    def _load_frame_files(self):
        """Load and sort PNG files from the frames folder."""
        if not os.path.exists(self.frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {self.frames_folder}")
            
        # Get all PNG files and sort them
        png_files = [f for f in os.listdir(self.frames_folder) if f.lower().endswith('.png')]
        png_files.sort()  # Sort to ensure consistent ordering
        
        self.frame_files = [os.path.join(self.frames_folder, f) for f in png_files]
        
        if not self.frame_files:
            raise ValueError(f"No PNG files found in folder: {self.frames_folder}")
            
        print(f"✅ Loaded {len(self.frame_files)} frame files from {self.frames_folder}")
        
    def _load_imu_data(self):
        """Load IMU data from CSV file if provided."""
        if self.imu_csv_file is None:
            return
            
        if not os.path.exists(self.imu_csv_file):
            print(f"🟡 Warning: IMU CSV file not found: {self.imu_csv_file}. Continuing without IMU data.")
            self.imu_csv_file = None
            return
            
        try:
            self.frame_data = pd.read_csv(self.imu_csv_file).to_dict(orient='records')
            self.start_timestamp = self.frame_data[0]['timestamp']
            print(f"✅ Loaded Frame data from {self.imu_csv_file} ({len(self.frame_data)} records)")
        except Exception as e:
            print(f"🟡 Warning: Failed to load IMU data: {str(e)}. Continuing without IMU data.")
            self.imu_csv_file = None
            self.frame_data = None
    
    def start(self):
        """Initialize the simulator by loading frame files and IMU data."""
        if self._is_started:
            print("🟡 Warning: Simulator is already started")
            return
            
        self._load_frame_files()
        self._load_imu_data()
        self.current_frame_index = 0
        self._is_started = True
        print("✅ FrameDataSimulator started")
    
    def stop(self):
        """Stop the simulator and reset state."""
        self._is_started = False
        self.current_frame_index = 0
        print("🛑 FrameDataSimulator stopped")
    
    def get_frame(self, index=None, relative_time=None, no_image=False) -> FrameData:
        """
        Get the next frame from the simulator.
        
        Returns:
            FrameData: Frame data with image, dummy timestamp, and IMU data (if available)
        """
        if not self._is_started:
            print("🛑 Error: Simulator not started. Call start() first.")
            return FrameData(None, None, None)
        
        if index is not None:
            self.current_frame_index = index
        elif relative_time is not None:
            self.current_frame_index = bisect.bisect_left(self.frame_data, relative_time, key=lambda x: x['timestamp'] - self.start_timestamp)
        else:
            # Advance to next frame
            self.current_frame_index += 1

        if self.current_frame_index >= len(self.frame_files):
            # Reached end of frames, restart the sequence
            print("[FrameDataSimulator] Reached end of frame sequence")
            self.current_frame_index = -1
            
        try:
            if not no_image:
                # Load the current frame image
                frame_path = self.frame_files[self.current_frame_index]
                image = Image.open(frame_path)
                
                # Convert to RGB format consistently
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                image = None
            
            # Get IMU data for current frame if available
            tracker_data = None
            if self.frame_data is not None and self.current_frame_index < len(self.frame_data):
                row = self.frame_data[self.current_frame_index]
                tracker_data = TrackerData.from_dict(row)  # Convert to TrackerData instance
                        
            # return FrameData(image, timestamp, frame_data)
            return FrameData(image, row['timestamp'], tracker_data, microns_per_pixel=0)
            
        except Exception as e:
            print(f"🛑 Error loading frame {self.current_frame_index}: {str(e)}")
            return FrameData(None, None, None, microns_per_pixel=0)
    
    def reset(self):
        """Reset the simulator to the beginning of the sequence."""
        self.current_frame_index = 0
        print("🔄 FrameDataSimulator reset to beginning")
    
    def get_total_frames(self) -> int:
        """Get the total number of frames available."""
        return len(self.frame_files)
    
    def get_current_frame_index(self) -> int:
        """Get the current frame index."""
        return self.current_frame_index
    
    def has_imu_data(self) -> bool:
        """Check if IMU data is available."""
        return self.frame_data is not None