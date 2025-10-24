#! /bin/bash

source initialize_envs.sh

# echo $DYLD_LIBRARY_PATH


python3 src/record_imu.py \
--duration 10 \
--outputdir ./output \
--ip "66.66.66.160"