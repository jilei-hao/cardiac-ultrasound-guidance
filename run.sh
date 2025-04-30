#! /bin/bash

dir_out=/Users/jileihao/dev/arpa-h-dev/lumify_streaming/output/250129/capture


python3 src/capture.py --render \
  --frame_interval 0 \
  --window_title "Input" \
  # --no_landmarks_for_pose \

# --duration 600