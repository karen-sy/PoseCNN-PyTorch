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
import easydict
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import scipy.io

import _init_paths
from fcn.test_imageset import test_image, vis_test
from fcn.config import cfg, cfg_from_file, yaml_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from sdf.sdf_optimizer import sdf_optimizer
import pickle

from lib.densefusion.network import PoseNet, PoseRefineNet


def env_setup_posecnn(args, cfg):
    ##################
    # Setup
    ##################

    print('Called with args:')
    print(args)

    if args.meta_file is not None:
        meta = yaml_from_file(args.meta_file)
        # overwrite test classes
        print(meta)
        if 'ycb_ids' in meta:
            cfg.TEST.CLASSES = [0]
            for i in meta.ycb_ids:
                cfg.TEST.CLASSES.append(i)
            print('TEST CLASSES:', cfg.TEST.CLASSES) 

    print('Using config:')
    pprint.pprint(cfg)

    # device
    cfg.gpu_id = args.gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(cfg.gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    cfg.TEST.SYNTHESIZE = False
    dataset = get_dataset(args.dataset_name)

    # index_images = range(len(images_color))
    print(f"Saving results to {args.resdir}")
    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)

    # prepare network
    network_data = torch.load(args.pretrained )
    print("=> using pre-trained network '{}'".format(args.pretrained ))

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()


    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=args.gpu_id, render_marker=False)
        
    if cfg.TEST.SYNTHESIZE:
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
    else:
        model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES[1:]]
        model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES[1:]]
        cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
    cfg.renderer.set_camera_default()
    print(dataset.model_mesh_paths)

    # load sdfs
    if cfg.TEST.POSE_REFINE:
        print('loading SDFs')
        sdf_files = []
        for i in cfg.TEST.CLASSES[1:]:
            sdf_files.append(dataset.model_sdf_paths[i-1])
        cfg.sdf_optimizer = sdf_optimizer(cfg.TEST.CLASSES[1:], sdf_files)

    return dataset, network

def get_image_posecnn(data):
    rgb_data = np.copy(data['rgb'])
    rgb_data[:,:,2] = rgb_data[:,:,0]  # swap b and g  (see get_img_blog in test_pandas.py instead)
    rgb_data[:,:,0] = data['rgb'][:,:,2]
    depth_data = data['depth']
    meta_data = dict({'factor_depth': data['factor_depth'], 'intrinsic_matrix': data['intrinsics']})
    return rgb_data, depth_data, meta_data

def run_posecnn(image_color, image_depth, meta_data, network, dataset, posecnn_cfg):
    ##################
    # TEST EACH IMAGE
    ##################
    # for each image
    # im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
    # depth = pad_im(cv2.imread(images_depth[i], cv2.IMREAD_UNCHANGED), 16)
    # depth = depth.astype('float') / 1000.0
    
    # overwrite intrinsics
    K = meta_data['intrinsic_matrix']
    dataset._intrinsic_matrix = K
    # print(dataset._intrinsic_matrix)

    # get images data
    im = pad_im(image_color, 16)
    depth = pad_im(image_depth, 16) / meta_data['factor_depth']

    print("entering test_image")
    _, _, _, labels, rois, poses, poses_refined = test_image(network, dataset, im, depth)
    # result = test_image(network, dataset, im, depth)

    print("Finished test_image")


    # map the roi index
    for j in range(rois.shape[0]):
        rois[j, 1] = posecnn_cfg.TRAIN.CLASSES.index(posecnn_cfg.TEST.CLASSES[int(rois[j, 1])])

    result = {'labels': labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined, 'intrinsic_matrix': dataset._intrinsic_matrix}
    # filename_head = filename.split(".")[0]
    # head, tail = os.path.split(filename_head)
    # filename = os.path.join(posecnn_args.resdir, tail + '.mat')
    # scipy.io.savemat(filename, result, do_compression=True)

    return result

if __name__ == '__main__':
    posecnn_args = easydict.EasyDict({
        'gpu_id':0, 
        'pretrained':'_trained_checkpoints/posecnn/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth', 
        'cfg_file':'experiments/cfgs/ycb_object.yml', 
        'dataset_name':'ycb_object_test', 
        'depth_name':'*depth.png', 
        'datadir':'datasets/pandas/data/panda_data_pik3_tiny/', #### Change here
        'resdir':'datasets/pandas/data/results_PoseCNN_pandas/', 
        'network_name':'posecnn', 
        'background_name':None,
        'meta_file':None, 
        'pretrained_encoder':None, 
        'codebook':None, 
    })
    if posecnn_args.cfg_file is not None:
        cfg_from_file(posecnn_args.cfg_file)  # creates variable `cfg`
        posecnn_cfg = cfg

    from IPython import embed; embed()
    # test_rgb = 
    # test_depth = 
    # test_intrinsics = 

    dataset_cfg, posecnn_network = env_setup_posecnn(posecnn_args, posecnn_cfg)

    ## load in data
    scene_data_dirs = os.listdir(posecnn_args.datadir)
    scene_data_dirs.sort()
    test_idx = 0
    test_filename_pik = posecnn_args.datadir + scene_data_dirs[test_idx]
    with open(test_filename_pik, 'rb') as file:
        test_data = pickle.load(file)
    
    image_color, image_depth, meta_data = get_image_posecnn(test_data)
    run_posecnn(image_color, image_depth, meta_data, posecnn_network, dataset_cfg, posecnn_cfg)


