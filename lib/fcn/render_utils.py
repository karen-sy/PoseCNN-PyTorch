# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import time
import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fcn.config import cfg
from transforms3d.quaternions import quat2mat


def render_image(dataset, im, rois, poses, poses_refine, labels):


    # print("\n\n\n IM IN RENDER IMAEG!")
    # from IPython import embed; embed()
    # label image
    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    num = poses.shape[0]
    classes = dataset._classes_test
    class_colors = dataset._class_colors_test

    cls_indexes = []
    poses_all = []
    poses_refine_all = []
    for i in range(num):
        if cfg.MODE == 'TEST':
            cls_index = int(rois[i, 1]) - 1
        else:
            cls_index = cfg.TRAIN.CLASSES[int(rois[i, 1])] - 1

        if cls_index < 0:
            continue

        cls_indexes.append(cls_index)
        qt = np.zeros((7, ), dtype=np.float32)
        qt[:3] = poses[i, 4:7]  # xyz
        qt[3:] = poses[i, :4]   # pose
        poses_all.append(qt.copy())

        if cfg.TEST.POSE_REFINE and poses_refine is not None:
            qt[:3] = poses_refine[i, 4:7]
            qt[3:] = poses_refine[i, :4]
            poses_refine_all.append(qt.copy())

        cls = int(rois[i, 1])
        print(classes[cls], rois[i, -1], cls_index)
        if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]

            cv2.rectangle(im_label, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[cls], 2)

            # draw center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(im_label, (cx, cy), 2, (255, 255, 0), 10)

    # rendering
    if len(cls_indexes) > 0 and cfg.TRAIN.POSE_REG:

        height = im.shape[0]
        width = im.shape[1]
        intrinsic_matrix = dataset._intrinsic_matrix
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
        znear = 0.01
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # pose
        cfg.renderer.set_poses(poses_all)
        frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output = 0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 1.0 * im_render.astype(np.float32)
        im_output = np.clip(im_output, 0, 255)

        # pose refine
        if cfg.TEST.POSE_REFINE and poses_refine is not None:
             cfg.renderer.set_poses(poses_refine_all)
             frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
             image_tensor = image_tensor.flip(0)
             im_render = image_tensor.cpu().numpy()
             im_render = np.clip(im_render, 0, 1)
             im_render = im_render[:, :, :3] * 255
             im_render = im_render.astype(np.uint8)
             im_output_refine = 0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 1.0 * im_render.astype(np.float32)
             im_output_refine = np.clip(im_output_refine, 0, 255)
             im_output_refine = im_output_refine.astype(np.uint8)
        else:
             im_output_refine = im_output.copy()
    else:
        im_output = 0.4 * im[:,:,(2, 1, 0)]
        im_output_refine = im_output.copy()

    ret =  im_output.astype(np.uint8), im_output_refine.astype(np.uint8), im_label

    # from IPython import embed; embed()

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



def render_predicted_image(dataset, im, rois, poses, cls_indexes):


    # print("\n\n\n IM IN RENDER IMAEG!")
    # from IPython import embed; embed()

    num = poses.shape[0]

    cls_indexes = []
    poses_all = []
    # poses_refine_all = []
    # for i in range(num):
    #     if cfg.MODE == 'TEST':
    #         cls_index = int(rois[i, 1]) - 1
    #     else:
    #         cls_index = cfg.TRAIN.CLASSES[int(rois[i, 1])] - 1

    #     if cls_index < 0:
    #         continue

    #     cls_indexes.append(cls_index)
    


    height = im.shape[0]
    width = im.shape[1]
    intrinsic_matrix = dataset._intrinsic_matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.01
    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # pose
    cfg.renderer.set_poses(poses_all)
    frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
    image_tensor = image_tensor.flip(0)
    im_render = image_tensor.cpu().numpy()
    im_render = np.clip(im_render, 0, 1)
    im_render = im_render[:, :, :3] * 255
    im_render = im_render.astype(np.uint8)
    im_output = im_render.astype(np.float32) #0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 1.0 * im_render.astype(np.float32)
    im_output = np.clip(im_output, 0, 255)

    ret = im_output.astype(np.uint8) 

    return ret
