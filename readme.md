# Real-time Screen Capture and Deep Learning Inference

## Prerequisites

-- Install `scrcpy` from: https://github.com/Genymobile/scrcpy

-- (Optional) Create links or add `adb` and `scrcpy` to PATH, or use absolute path to the executables in the following steps

-- Run following code to initialize the environment:

 `conda create -n lumify-streaming python=3.12`

 `conda activate lumify-streaming`

 `pip install torch`
 `pip install mss`
 `pip install pyobjc`
 `pip install "numpy<2"`
 `pip install opencv-python`
 `pip install Pillow`
 `pip install requests`

## Run the Program

1. Run `bash run_scrcpy.sh` (see Additional Instructions for details)

2. In another terminal window, run `bash run.sh`


## Additional Instructions

### How to configure scrcpy using cable

1. Connect android device using cable

2. Run `adb start-server`

3. Run `scrcpy`

### How to configure scrcpy using wifi

1. Connect android device using cable

2. Run `adb tcpip 5555`

3. Run `adb shell ip route` to get device ip

4. Run `adb connect <device_ip>:5555`

5. Disconnect cable

6. Run `scrcpy`
