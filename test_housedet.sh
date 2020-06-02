#!/bin/bash
set -e

if [ -z "$1" ]
  then
    echo "Please provide the path to the test image"
fi

# preprocess image to patches
python process_map_image.py --img-path $1

# run centernet detection
python src/test.py housedet --exp_id ct_det_house_res --arch res_101 --keep_res --resume --trainval