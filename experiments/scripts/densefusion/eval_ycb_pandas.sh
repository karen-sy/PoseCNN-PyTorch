#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

# if [ ! -d YCB_Video_toolbox ];then
#     echo 'Downloading the YCB_Video_toolbox...'
#     git clone https://github.com/yuxng/YCB_Video_toolbox.git
#     cd YCB_Video_toolbox
#     unzip results_PoseCNN_RSS2018.zip
#     cd ..
#     cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
# fi

python3 ./tools/densefusion/eval_ycb_pandas.py --dataset_root ./datasets/pandas\
  --model _trained_checkpoints/densefusion/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model _trained_checkpoints/densefusion/ycb/pose_refine_model_69_0.009449292959118935.pth