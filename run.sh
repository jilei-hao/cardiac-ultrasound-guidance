#! /bin/bash

dir_out=/Users/jileihao/dev/arpa-h-dev/lumify_streaming/output/250129/capture


python3 src/capture.py --render \
  --frame_interval 0.01 \
  --duration 600 \
  --window_title "SM-T860"