#!/bin/bash

# Restart adb server
adb kill-server
adb start-server

# Switch to TCP/IP mode on port 5555
adb tcpip 5555

# wait for 5 seconds
sleep 5

# Get device IP
ip=$(adb shell ip route | awk '{print $NF}')
echo "Connecting to $ip"

# Connect over Wi-Fi
adb connect "$ip:5555"