# Clarius Exam with IMU Recording Utility

## Setup

- Read `documentation/clarius_setup` for setting up the clarius connection
- Modify argument values in `run_recording.sh`
  - `ip` and `port` for clarius connection
  - `duration` the recording duration of each exam in seconds
  - `outputdir` the directory storing the output recordings
- Make sure to turn on position data for the scanning session or imu data will be empty (Use the option button below the scanning window)

## Run

Run `bash run_recording.sh` in a terminal
