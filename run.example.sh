#!/bin/bash

# run recording utility
# -- duration in seconds
# -- interval in milliseconds
source ./scripts/setup.sh && cusg-record \
--ip 192.168.1.1 --port 5828 \
--outputdir ./output/test_recording \
--duration 10 \
--interval 50 \
--preserve-date