#!/bin/bash

data_folder="/Users/jileihao/dev/paradigm/cardiac-ultrasound-guidance/output/tte001/02064993769690"

python src/guidance_ui.py \
--source simulator \
--frames-folder $data_folder \
--window-size 512