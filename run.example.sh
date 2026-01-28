#!/bin/bash

# run recording utility
# -- duration in seconds
# -- interval in milliseconds
source ./scripts/setup.sh && cusg-record \
--ip 66.66.66.160 --port 5828 \
--outputdir ./output \
--duration 10 \
--interval 50 \
--preserve-date