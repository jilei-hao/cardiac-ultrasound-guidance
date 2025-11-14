#!/bin/bash

data_folder="/Users/jileihao/dev/paradigm/cardiac-ultrasound-guidance/output/tte001/02064993769690"

python src/guidance_ui.py \
--source simulator \
--frames-folder $data_folder \
--window-size 512


# python src/guidance_ui.py \
# --source clarius \
# --clarius-ip 66.66.66.160 \
# --window-size 512