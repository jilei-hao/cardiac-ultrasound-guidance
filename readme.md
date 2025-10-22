# Real-time Screen Capture and Deep Learning Inference

## Prerequisites

-- (Optional) Install `scrcpy` from: https://github.com/Genymobile/scrcpy

-- (Optional) Create links or add `adb` and `scrcpy` to PATH, or use absolute path to the executables in the following steps

-- Run following code to initialize the environment:

 `conda create -n lumify-streaming python=3.12`

 `conda activate lumify-streaming`

 `pip install -r requirements.txt`

 **Alternative:** Install packages individually:
 ```
 pip install torch
 pip install mss
 pip install pyobjc
 pip install "numpy<2"
 pip install opencv-python
 pip install Pillow
 pip install requests
 ```

## Documentations

### Main Program

Go to `documentation/run_main.md` for instructions running the main program for frame capture and inference

### Exam Recording Utility

Go to `documentation/run_recording.md` for instructions running the recording utility for capturing the clarius frames with imu data

### Clarius Setup

Go to `documentation/clarius_setup.md` for instructions setting up connection to the clarius scanner

### Dependencies and Version Management

Go to `documentation/requirements.md` for information about managing dependencies, checking package versions (including NumPy), and troubleshooting
