#! /bin/bash

server_ip=10.102.149.196

python3 src/capture.py \
  --window_title "Input" \
  --upload_url http://$server_ip:8000/upload

# --save --save_folder /path/to/output/folder