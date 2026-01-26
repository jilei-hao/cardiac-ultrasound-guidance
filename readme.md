# Real-time Screen Capture and Deep Learning Inference

## Installation

### Quick Install (Recommended)

1. Create and activate a conda environment:
   ```bash
   conda create -n cardiac-guidance python=3.10
   conda activate cardiac-guidance
   ```

2. Download cast API
   ```bash
   ./scripts/download_cast_api.sh
   ```

3. Install the package:
   ```bash
   # Standard installation
   pip install .
   
   # Or for development (editable install)
   pip install -e .
   
   # With development dependencies (pytest, black, etc.)
   pip install -e .[dev]
   ```

4. After installation, you can use the following commands, always source `scripts/setup.sh` before any run:
   ```bash
   # Main guidance application
   source scripts/setup.sh && cusg-main --help
   
   # Guidance UI application  
   source scripts/setup.sh && cusg-guidance-ui --help
   
   # IMU recording utility
   source scripts/setup.sh && cusg-record-imu --help
   ```

### Manual Installation (Alternative)

If you prefer to install dependencies manually:

```bash
conda create -n lumify-streaming python=3.12
conda activate lumify-streaming
pip install -r requirements.txt
```

## Prerequisites

-- (Optional) Install `scrcpy` from: https://github.com/Genymobile/scrcpy

-- (Optional) Create links or add `adb` and `scrcpy` to PATH, or use absolute path to the executables in the following steps


## Documentations

### Main Program

Go to `documentation/run_main.md` for instructions running the main program for frame capture and inference

### Exam Recording Utility

Go to `documentation/run_recording.md` for instructions running the recording utility for capturing the clarius frames with imu data

### Clarius Setup

Go to `documentation/clarius_setup.md` for instructions setting up connection to the clarius scanner
