#! /bin/bash

source initialize_envs.sh

# echo $DYLD_LIBRARY_PATH


python3 src/record.py --duration 10 --outputdir ./output --ip "192.168.50.192"