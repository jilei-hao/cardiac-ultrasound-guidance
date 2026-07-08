import os
import re
import time
import pandas as pd
from PIL import Image
from ..frame_data import FrameData, IMUData
from ..frame_data_input import FrameDataInput


# Capture frame files carry their source timestamp in the name:
# frame_<seq>_<timestamp_ns>.png (written by record.py and the USGR
# recording utility alike).
_FRAME_TS_RE = re.compile(r"^frame_\d+_(\d+)\.png$", re.IGNORECASE)


class FrameDataSimulator(FrameDataInput):
    """
    Simulator for frame data that loads PNG images from a folder and optionally
    corresponding IMU data from a CSV file.

    Accepts either a folder of frame PNGs (with an optional frame.imu.csv /
    frame.meta.csv alongside) or a capture root whose frames live in an
    images/ subfolder — the layout the USGR recording utility produces.
    """

    def __init__(self, frames_folder: str, imu_csv_file: str = None,
                 realtime: bool = False):
        """
        Initialize the frame data simulator.

        Args:
            frames_folder: Path to folder containing PNG frame images, or a
                capture root containing an images/ subfolder of PNGs
            imu_csv_file: Path to CSV file containing IMU data (optional)
            realtime: Pace get_frame() to the source timestamps so replay
                runs at the recorded frame rate instead of free-running
        """
        # USGR recording-utility captures keep the PNGs (and the meta CSV)
        # in an images/ subfolder; descend so callers can pass the capture
        # root itself.
        if os.path.isdir(frames_folder):
            has_pngs = any(
                f.lower().endswith('.png') for f in os.listdir(frames_folder)
            )
            images_subdir = os.path.join(frames_folder, "images")
            if not has_pngs and os.path.isdir(images_subdir):
                print(f"📂 No PNGs at capture root; using {images_subdir}")
                frames_folder = images_subdir

        self.frames_folder = frames_folder
        if (imu_csv_file is not None) and (not os.path.exists(imu_csv_file)):
            raise FileNotFoundError(f"IMU CSV file not found: {imu_csv_file}")

        # try two default CSV file names if not explicitly provided
        if imu_csv_file is None:
            imu_csv_file = os.path.join(frames_folder, "frame.imu.csv")
            if not os.path.exists(imu_csv_file):
                imu_csv_file = os.path.join(frames_folder, "frame.meta.csv")
            if not os.path.exists(imu_csv_file):
                print(f"🟡 Warning: no frame.imu.csv/frame.meta.csv in "
                      f"{frames_folder}. Continuing without IMU data.")
                imu_csv_file = None

        self.imu_csv_file = imu_csv_file
        self.frame_files = []
        self.imu_data = None
        self.current_frame_index = 0
        self.start_time = None
        self._is_started = False
        # Real-time pacing state: wall-clock and source-timestamp anchors
        # of the previously served frame.
        self._realtime = realtime
        self._pace_wall = None
        self._pace_src_ns = None
        
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
        self._pace_wall = None
        self._pace_src_ns = None
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

            # Get IMU data and per-frame capture metadata if available
            imu_data = None
            timestamp = None
            microns_per_pixel = 0.0
            if self.imu_data is not None and self.current_frame_index < len(self.imu_data):
                row = self.imu_data.iloc[self.current_frame_index].to_dict()
                imu_data = IMUData.from_dict(row)  # Convert to IMUData instance
                ts_val = row.get('timestamp')
                if ts_val is not None and pd.notna(ts_val):
                    timestamp = int(ts_val)
                mpp = row.get('microns_per_pixel')
                if mpp is not None and pd.notna(mpp):
                    microns_per_pixel = float(mpp)

            # Source timestamp: prefer the CSV column, then the ns suffix in
            # the filename, then wall clock. Serving source time keeps
            # downstream dt-based consumers (e.g. pose smoothing) consistent
            # with the recorded motion, and the jump backwards on sequence
            # wraparound lets them reset.
            if timestamp is None:
                m = _FRAME_TS_RE.match(os.path.basename(frame_path))
                if m:
                    timestamp = int(m.group(1))
            if timestamp is None:
                timestamp = time.time_ns()

            if self._realtime:
                self._pace(timestamp)

            # Advance to next frame
            self.current_frame_index += 1

            return FrameData(image, timestamp, imu_data, microns_per_pixel)

        except Exception as e:
            print(f"🛑 Error loading frame {self.current_frame_index}: {str(e)}")
            return FrameData(None, None, None)

    def _pace(self, src_ts_ns: int):
        """Sleep so consecutive frames are served at the source frame rate.

        Anchored to the previous frame's (wall clock, source timestamp)
        pair. Gaps are capped at 1 s so recording pauses don't stall
        playback; wraparound (source time going backwards) just re-anchors.
        """
        now = time.monotonic()
        if self._pace_src_ns is not None and src_ts_ns > self._pace_src_ns:
            src_dt = min((src_ts_ns - self._pace_src_ns) / 1e9, 1.0)
            target = self._pace_wall + src_dt
            delay = target - now
            if delay > 0:
                time.sleep(delay)
                now = target
        self._pace_wall = now
        self._pace_src_ns = src_ts_ns
    
    def reset(self):
        """Reset the simulator to the beginning of the sequence."""
        self.current_frame_index = 0
        self.start_time = time.time_ns()
        self._pace_wall = None
        self._pace_src_ns = None
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