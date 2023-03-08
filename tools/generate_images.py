#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob

from scipy.spatial.transform import Rotation as R

import _init_paths
from fcn.test_imageset import test_image, vis_test
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from sdf.sdf_optimizer import sdf_optimizer



def render_predicted_image(K, h, w, far, near, poses, cls_idxs):


    print("\n\n\n IM IN RENDER IMAEG!")

    cls_indexes = []
    poses_all = []

    for pose, cls_idx in zip(poses, cls_idxs):
        qt = np.zeros((7, ), dtype=np.float32)
        r = R.from_matrix(pose[:3,:3])
        quat = r.as_quat()
        xyz = pose[:3, 3]
        qt[:3] += xyz
        qt[3:] += quat

        poses_all.append(qt.copy())
        cls_indexes.append(cls_idx)

    # from IPython import embed; embed()

    height = h
    width = w 
    intrinsic_matrix = K
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = far
    znear = near
    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    renderer.set_light_pos([0, 0, 0])
    renderer.set_light_color([1, 1, 1])
    renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # pose
    renderer.set_poses(poses_all)
    frame = renderer.render(cls_indexes, image_tensor, seg_tensor)
    image_tensor = image_tensor.flip(0)
    im_render = image_tensor.cpu().numpy()
    im_render = np.clip(im_render, 0, 1)
    im_render = im_render[:, :, :3] * 255
    im_render = im_render.astype(np.uint8)
    im_output = im_render.astype(np.float32) #0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 1.0 * im_render.astype(np.float32)
    im_output = np.clip(im_output, 0, 255)

    ret = im_output.astype(np.uint8) 

    return ret

from PIL import Image
def get_rgb_image(image, max_val=255.0):
    if image.shape[-1] == 3:
        image_type = "RGB"
    else:
        image_type = "RGBA"

    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode=image_type,
    ).convert("RGBA")
    return img

import matplotlib.pyplot as plt
def get_depth_image(image, min=0.0, max=1.0):
    cm = plt.get_cmap('turbo')
    img = Image.fromarray(
        np.rint(cm((np.clip(np.array(image), min, max) - min) / (max - min)) * 255.0).astype(np.int8), mode="RGBA"
    )
    return img

# dataset
dataset = get_dataset("ycb_object_test")

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

print('loading 3D models')
TEST_CLASSES = [i for i in range(15)]
renderer = YCBRenderer(width=w, height=h, gpu_id=0, render_marker=False)
model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in TEST_CLASSES[1:]]
model_texture_paths = [dataset.model_texture_paths[i-1] for i in TEST_CLASSES[1:]]
model_colors = [dataset.model_colors[i-1] for i in TEST_CLASSES[1:]]
renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
# renderer.set_camera_default()
print(dataset.model_mesh_paths)

# load sdfs
if cfg.TEST.POSE_REFINE:
    print('loading SDFs')
    sdf_files = []
    for i in TEST_CLASSES[1:]:
        sdf_files.append(dataset.model_sdf_paths[i-1])
    cfg.sdf_optimizer = sdf_optimizer(TEST_CLASSES[1:], sdf_files)


poses = [np.array([[1,0,0,0],
                [0,-7.0710665e-01, -7.0710677e-01, 2.6442868e-07],
                [0, 7.0710677e-01, -7.0710671e-01, 7.4264050e-01],
                [0,0,0,1]])]
cls_indexes = [11]
near,far = 0.001, 10.0

ret = render_predicted_image(K, h, w, far, near, poses, cls_indexes)

get_rgb_image(ret).save("rgb.png")
get_depth_image(ret, max=far).save("depth.png")
# from IPython import embed; embed()