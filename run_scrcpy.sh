#! /bin/bash

# if not in PATH, replace with the path to your scrcpy executable
path_scrcpy=scrcpy 

# execute scrcpy
$path_scrcpy \
  --window-title "SM-T860" \
  --video-bit-rate 8M \
  --no-control \
  --no-audio \
  --crop 1600:1600:0:400 \
  --max-size 1000


# --max-size 1280 \
