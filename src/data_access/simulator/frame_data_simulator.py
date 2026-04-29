import os
import time
import pandas as pd
from PIL import Image
from ..frame_data import FrameData, IMUData
from ..frame_data_input import FrameDataInput


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
        
        # try two default CSV file names if not explicitly provided
        if imu_csv_file is None:
            imu_csv_file = os.path.join(frames_folder, "frame.imu.csv")
        if (imu_csv_file is not None) and (not os.path.exists(imu_csv_file)):
            imu_csv_file = os.path.join(frames_folder, "frame.meta.csv")
        if (imu_csv_file is not None) and (not os.path.exists(imu_csv_file)):
            raise FileNotFoundError(f"IMU CSV file not found: {imu_csv_file}")
        
        self.imu_csv_file = imu_csv_file
        self.frame_files = []
        self.imu_data = None
        self.current_frame_index = 0
        self.start_time = None
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
            self.imu_data = pd.read_csv(self.imu_csv_file)
            print(f"✅ Loaded IMU data from {self.imu_csv_file} ({len(self.imu_data)} records)")
        except Exception as e:
            print(f"🟡 Warning: Failed to load IMU data: {str(e)}. Continuing without IMU data.")
            self.imu_csv_file = None
            self.imu_data = None
    
    def start(self):
        """Initialize the simulator by loading frame files and IMU data."""
        if self._is_started:
            print("🟡 Warning: Simulator is already started")
            return
            
        self._load_frame_files()
        self._load_imu_data()
        self.current_frame_index = 0
        self.start_time = time.time_ns()
        self._is_started = True
        print("✅ FrameDataSimulator started")
    
    def stop(self):
        """Stop the simulator and reset state."""
        self._is_started = False
        self.current_frame_index = 0
        self.start_time = None
        print("🛑 FrameDataSimulator stopped")
    
    def get_frame(self) -> FrameData:
        """
        Get the next frame from the simulator.
        
        Returns:
            FrameData: Frame data with image, dummy timestamp, and IMU data (if available)
        """
        if not self._is_started:
            print("🛑 Error: Simulator not started. Call start() first.")
            return FrameData(None, None, None)
            
        if self.current_frame_index >= len(self.frame_files):
            # Reached end of frames, restart the sequence
            print("[FrameDataSimulator] Reached end of frame sequence, restarting from beginning")
            self.current_frame_index = 0
            self.start_time = time.time_ns()
            
        try:
            # Load the current frame image
            frame_path = self.frame_files[self.current_frame_index]
            image = Image.open(frame_path)
            
            # Convert to RGB format consistently
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate dummy timestamp (nanoseconds since start)
            current_time = time.time_ns()
            timestamp = current_time
            
            # Get IMU data for current frame if available
            imu_data = None
            if self.imu_data is not None and self.current_frame_index < len(self.imu_data):
                imu_row = self.imu_data.iloc[self.current_frame_index]
                imu_data = IMUData.from_dict(imu_row.to_dict())  # Convert to IMUData instance
            
            # Advance to next frame
            self.current_frame_index += 1
            
            return FrameData(image, timestamp, imu_data)
            
        except Exception as e:
            print(f"🛑 Error loading frame {self.current_frame_index}: {str(e)}")
            return FrameData(None, None, None)
    
    def reset(self):
        """Reset the simulator to the beginning of the sequence."""
        self.current_frame_index = 0
        self.start_time = time.time_ns()
        print("🔄 FrameDataSimulator reset to beginning")
    
    def get_total_frames(self) -> int:
        """Get the total number of frames available."""
        return len(self.frame_files)
    
    def get_current_frame_index(self) -> int:
        """Get the current frame index."""
        return self.current_frame_index
    
    def has_imu_data(self) -> bool:
        """Check if IMU data is available."""
        return self.imu_data is not None