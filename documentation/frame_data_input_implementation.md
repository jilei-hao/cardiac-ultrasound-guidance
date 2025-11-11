# FrameDataInput Implementation Summary

## Overview
I have successfully implemented a base class architecture for all data access classes with the following components:

## 1. FrameDataInput Base Class
**Location**: `src/data_access/frame_data_input.py`

- Abstract base class using Python's `abc` module
- Defines the interface that all frame data sources must implement
- Required methods:
  - `get_frame() -> FrameData`: Retrieve next frame
  - `start()`: Initialize and start data source
  - `stop()`: Stop and cleanup data source

## 2. Updated Existing Classes

### ClariusStreamer 
**Location**: `src/data_access/clarius/clarius_streamer.py`
- Now inherits from `FrameDataInput`
- Wraps existing global functions into class-based interface
- Constructor takes: `ip`, `port`, `path`, `width`, `height`
- Maintains backward compatibility with existing code

### ScreenCapture
**Location**: `src/data_access/screen_capture/screen_capture.py`
- Now inherits from `FrameDataInput`
- Already had the required methods, just needed to add inheritance
- No functional changes required

## 3. New FrameDataSimulator Class
**Location**: `src/data_access/simulator/frame_data_simulator.py`

### Features:
- **PNG Frame Loading**: Loads PNG images from a specified folder
- **IMU Data Integration**: Optionally loads IMU data from CSV file
- **Dummy Timestamps**: Generates nanosecond timestamps for each frame
- **Sequential Playback**: Maintains frame order and current position
- **Reset Functionality**: Can reset to beginning of sequence
- **Error Handling**: Graceful handling of missing files/data

### Constructor Parameters:
```python
FrameDataSimulator(frames_folder: str, imu_csv_file: str = None)
```

### Key Methods:
- `start()`: Load frames and IMU data
- `stop()`: Cleanup and reset state
- `get_frame()`: Get next frame with image, timestamp, and IMU data
- `reset()`: Return to beginning of sequence
- `get_total_frames()`: Get total frame count
- `get_current_frame_index()`: Get current position
- `has_imu_data()`: Check if IMU data is available

## 4. Testing
**Simulator Testing**: `tests/test_simulator.py`
**Window Testing**: `tests/test_basic_windows.py`

Successfully tested with existing data:
- **Test Data**: `output/20250614-012501_155563293680/`
- **Frame Count**: 190 PNG files (512x512 pixels each)
- **IMU Data**: 190 records with timestamp, gyroscope, accelerometer, magnetometer, and quaternion data
- **Results**: All functionality working correctly

## 5. Updated Module Exports
Updated `__init__.py` files in:
- `src/data_access/`
- `src/data_access/simulator/`
- `src/data_access/clarius/`
- `src/data_access/screen_capture/`

## Usage Examples

### FrameDataSimulator
```python
from data_access.simulator import FrameDataSimulator

# Create simulator with frames and IMU data
simulator = FrameDataSimulator(
    frames_folder="./output/sample_frames",
    imu_csv_file="./output/sample_imu.csv"
)

simulator.start()

# Get frames
while True:
    frame = simulator.get_frame()
    if frame.image is None:
        break
    
    print(f"Frame: {frame.timestamp}, IMU: {frame.imu is not None}")
    # Process frame.image, frame.timestamp, frame.imu

simulator.stop()
```

### ClariusStreamer
```python
from data_access.clarius import ClariusStreamer

clarius = ClariusStreamer(ip="192.168.1.1", port=5828, path="research")
clarius.start()

frame = clarius.get_frame()
# Process frame

clarius.stop()
```

### ScreenCapture
```python
from data_access.screen_capture import ScreenCapture

screen = ScreenCapture(window_title="MyWindow", app_settings=settings)
screen.start()

frame = screen.get_frame()
# Process frame

screen.stop()
```

## Dependencies
- **PIL (Pillow)**: For image loading
- **pandas**: For CSV processing
- **numpy**: For numerical operations

All dependencies have been installed in the virtual environment.

## Benefits
1. **Consistent Interface**: All data sources follow the same pattern
2. **Polymorphism**: Can swap data sources easily
3. **Testing**: Simulator enables development without hardware
4. **Extensibility**: Easy to add new data sources
5. **Backward Compatibility**: Existing code continues to work