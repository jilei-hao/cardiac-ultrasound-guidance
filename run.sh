#! /bin/bash

dir_out=/Users/jileihao/dev/arpa-h-dev/lumify_streaming/output/250129/capture


python3 src/capture.py --render \
  --frame_interval 0 \
  --window_title "Input" \
  --show_lvef true
  # --upload_url http://141.212.114.250:8000/upload
  # --use_landmarks_for_pose false \

# --duration 600