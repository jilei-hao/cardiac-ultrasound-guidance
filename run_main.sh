#! /bin/bash

server_ip=141.212.114.250

python3 src/main.py \
  --window_title "Input" \
  --upload_url http://$server_ip:8000/upload

# --save --save_folder /path/to/output/folder