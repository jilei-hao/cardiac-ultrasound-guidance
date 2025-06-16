#! /bin/bash

source initialize_envs.sh

# echo $DYLD_LIBRARY_PATH


python3 src/record_imu.py --duration 10 --outputdir ./output \
--ip "2607:f470:6:20fc:11d4:7ab9:9c90:b6f"