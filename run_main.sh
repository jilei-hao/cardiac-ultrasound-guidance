#! /bin/bash

server_ip=141.212.114.250

source initialize_envs.sh

# python3 src/main.py \
#   --source clarius \
#   --clarius_ip 66.66.66.160 \
#   --clarius_port 5828 \
#   --upload_url http://$server_ip:8000/upload \
#   --guidance_window


python3 src/main.py \
  --window_title "Input" \
  --upload_url http://$server_ip:8000/upload \
  --guidance_window
# --save --save_folder /path/to/output/folder