#!/bin/bash

# prompt user to connect device with cable
echo "Please connect your Android device to the computer via USB cable. Press Enter to continue..."
read -r _

# Restart adb server
adb kill-server
adb start-server

# Switch to TCP/IP mode on port 5555
adb tcpip 5555

# wait for 5 seconds
sleep 2

# Get device IP
ip=$(adb shell ip route | awk '{print $NF}')
echo "Connecting to $ip"

# Connect over Wi-Fi
adb connect "$ip:5555"


echo "Please unplug the USB cable. Press Enter to run adb devices to check connection..."
read -r _

adb devices