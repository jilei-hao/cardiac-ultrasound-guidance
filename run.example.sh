#!/bin/bash

# run recording utility
# -- duration in seconds
# -- interval in milliseconds
# source ./scripts/setup.sh && cusg-record \
# --ip 66.66.66.160 --port 5828 \
# --outputdir ./output \
# --duration 30 \
# --interval 50 \
# --preserve-date


##################################
# run probe rendering test
##################################
# source ./scripts/setup.sh && cusg-probe-render \
# --source clarius --clarius-ip 192.168.50.192 --clarius-port 5828

# source ./scripts/setup.sh && cusg-probe-render \
# --source clarius --clarius-ip 10.102.9.225 --clarius-port 5828


source ./scripts/setup.sh && cusg-probe-render \
--source simulator \
--frames-folder ./output/1777469821799232000


##################################
# Run cusg guidance ui
##################################
# source ./scripts/setup.sh && cusg-guidance-ui \
# --source simulator \
# --frames-folder /Users/jileihao/data/vigil/recorded_exams/tte001/01543274692500

