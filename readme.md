# Script to capture real-time frame from an android device

## How to start scrcpy using cable

1. Connect android device using cable

2. Run `adb start-server`

3. Run `scrcpy`

## How to start scrcpy using wifi

1. Connect android device using cable

2. Run `adb tcpip 5555`

3. Run `adb shell ip route` to get device ip

4. Run `adb connect <device_ip>:5555`

5. Disconnect cable

6. Run `scrcpy`