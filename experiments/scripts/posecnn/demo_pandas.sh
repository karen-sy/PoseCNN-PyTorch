#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

time ./tools/test_images2.py --gpu 0 \
  --datadir datasets/pandas/data/panda_data_pik3_tiny/ \
  --savedir datasets/pandas/data/results_PoseCNN_pandas/ \
  --network posecnn \
  --pretrained _trained_checkpoints/posecnn/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object.yml

