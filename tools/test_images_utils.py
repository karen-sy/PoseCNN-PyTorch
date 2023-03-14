import pprint
import os
import os.path as osp
from pathlib import Path
import numpy as np
import copy
import cv2
import numpy as np
from PIL import Image 
import scipy.io as scio
import numpy.ma as ma
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import tools._init_paths

from lib.densefusion.network import PoseNet, PoseRefineNet
from lib.densefusion.transformations import quaternion_matrix, quaternion_from_matrix

from fcn.config import yaml_from_file
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from sdf.sdf_optimizer import sdf_optimizer

from fcn.test_imageset import test_image


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

img_width = 600
img_length = 600


# torch.backends.cudnn.enabled = False

xmap = np.array([[j for i in range(img_length)] for j in range(img_width)])
ymap = np.array([[i for i in range(img_length)] for j in range(img_width)])
num_obj = 21
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1


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

    print("PoseCNN: entering test_image")
    _, _, _, labels, rois, poses, poses_refined = test_image(network, dataset, im, depth)
    # result = test_image(network, dataset, im, depth)

    # from IPython import embed; embed()

    print("PoseCNN: Finished test_image")


    # map the roi index
    for j in range(rois.shape[0]):
        rois[j, 1] = posecnn_cfg.TRAIN.CLASSES.index(posecnn_cfg.TEST.CLASSES[int(rois[j, 1])])

    result = {'labels': labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_refined, 'intrinsic_matrix': dataset._intrinsic_matrix}
    # filename_head = filename.split(".")[0]
    # head, tail = os.path.split(filename_head)
    # filename = os.path.join(posecnn_args.resdir, tail + '.mat')
    # scipy.io.savemat(filename, result, do_compression=True)

    return result




def get_rgb_image(image, max_val=255.0):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGB",
    ).convert("RGBA")
    return img   

def get_image_densefusion(data):
    rgb_data = np.copy(data['rgb'])
    depth_data = data['depth']
    meta_data = dict({'factor_depth': data['factor_depth'], 'intrinsic_matrix': data['intrinsics']})
    return rgb_data, depth_data, meta_data

def get_blended_image(img, anno_img, max_val=255.0, alpha=0.5):
    og_img_viz = get_rgb_image(img)
    anno_img_viz = get_rgb_image(anno_img)

    return Image.blend(og_img_viz, anno_img_viz, alpha=0.5)

def get_bbox(posecnn_rois, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def env_setup_densefusion(densefusion_args):
    estimator = PoseNet(num_points = num_points, num_obj = num_obj)
    estimator.cuda()
    estimator.load_state_dict(torch.load(densefusion_args.model))
    estimator.eval()

    refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
    refiner.cuda()
    refiner.load_state_dict(torch.load(densefusion_args.refine_model))
    refiner.eval()

    class_file = open('{0}/classes.txt'.format(densefusion_args.dataset_config_dir))  # same as YCB
    class_id = 1 
    cld = {}
    class_names = ['0']
    while 1:
        class_input = class_file.readline()
        if not class_input:
            break
        class_input = class_input[:-1]
        class_names.append(class_input)
        this_dir = osp.dirname(__file__)
        dir_path = osp.join(this_dir, '..','data')
        input_file = open('{0}/models/{1}/points.xyz'.format(dir_path, class_input))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1]
            input_line = input_line.split(' ')
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        cld[class_id] = np.array(cld[class_id])
        class_id += 1

    return estimator, refiner, class_names, cld

def run_DenseFusion(image_color, image_depth, meta_data, estimator, refiner, class_names, cld, densefusion_args, scene_frame_name='out', posecnn_meta=None):
    # img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    # depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    # posecnn_meta = scio.loadmat('{0}/results_PoseCNN_pandas/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    # label = np.array(posecnn_meta['labels'])  # segmentation label 480x640  (unique: [labels,in,img])
    # posecnn_rois = np.array(posecnn_meta['rois'])
    # with open(scene_frame_pik, 'rb') as file:
    #     data = pickle.load(file)


    print(f"****************Processing {scene_frame_name}*******************")

    # image_color = data['rgb']  # color
    # im = pad_im(image_color, 16)
    image_depth = pad_im(image_depth, 16) / meta_data['factor_depth']

    # image_depth = image_depth / meta_data['factor_depth']
    K = meta_data['intrinsic_matrix']
    # depth_factor = data['factor_depth']

    cam_cx = K[0][-1]   
    cam_cy = K[1][-1]
    cam_fx = K[0][0]
    cam_fy = K[1][1]
    cam_scale = 1 #10000.0
    img_width, img_length = image_depth.shape

    if posecnn_meta is None:
        posecnn_meta = scio.loadmat(f'{densefusion_args.dataset_root}/data/results_PoseCNN_pandas/{scene_frame_name}.mat')
    else:
        print("******Loading posecnn results from current run******")
    label = np.array(posecnn_meta['labels'])  # segmentation label 480x640  (unique: [labels,in,img])
    posecnn_rois = np.array(posecnn_meta['rois'])


    lst = posecnn_rois[:, 1:2].flatten()
    my_result_wo_refine = []
    my_result = []
    img_with_pts_refined = None  # initialize result images
    img_with_pts_unrefined = None
    for idx in range(len(lst)):
        itemid = lst[idx]
        print(class_names[int(itemid)])
        try:
            if int(itemid) == 0:
                raise ValueError("Bad PoseCNN Result")
    
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(image_depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = image_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            
            # from IPython import embed; embed()

            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(image_color)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)  # ***************************************** 
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())


            ### viz unrefined results
            mat_r= quaternion_matrix(my_r)[0:3,0:3]
            imgpts, jac = cv2.projectPoints(cld[itemid], mat_r, my_t, K, np.array([[0., 0.0,  0.0, 0.0, 0.0]]))

            if img_with_pts_unrefined is None:
                img_with_pts_unrefined = np.array(image_color)
            for pt_coord in np.int32(imgpts):
                x,y = pt_coord[0]
                img_with_pts_unrefined = cv2.circle(img_with_pts_unrefined, (x,y), 1, (0,255,255))         

            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

            # Here 'my_pred' is the final pose estimation result after refinement (class_name: {'class_id':int, 'rot_q': quaternion, 'tr': translation})
            my_result.append({class_names[int(itemid)]: {'class_id': int(itemid), 'rot_q':my_r, 'tr':my_t}})


            ##################
            #  VIZ REFINED RESULTS (https://github.com/j96w/DenseFusion/issues/27)
            ##################
            mat_r= quaternion_matrix(my_r)[0:3,0:3]
            imgpts, jac = cv2.projectPoints(cld[itemid], mat_r, my_t, K, np.array([[0., 0.0,  0.0, 0.0, 0.0]]))

            if img_with_pts_refined is None:
                img_with_pts_refined = np.array(image_color)
            for pt_coord in np.int32(imgpts):
                x,y = pt_coord[0]
                img_with_pts_refined = cv2.circle(img_with_pts_refined, (x,y), 1, (0,255,255))  
            
        except (ZeroDivisionError, ValueError, IndexError):
            print("PoseCNN Detector Lost {0} at Keyframe".format(itemid))
            # from IPython import embed; embed()
            # my_result_wo_refine.append([0.0 for i in range(7)])
            # my_result.append([0.0 for i in range(7)])

    ## SAVE VIZ
    if not os.path.exists(densefusion_args.result_refine_dir):
        Path(densefusion_args.result_refine_dir).mkdir(parents=True)
        Path(densefusion_args.result_refine_dir + '/pred_txt').mkdir(parents=True)
    if not os.path.exists(densefusion_args.result_wo_refine_dir):
        Path(densefusion_args.result_wo_refine_dir).mkdir(parents=True)
        Path(densefusion_args.result_wo_refine_dir + '/pred_txt').mkdir(parents=True)
    if img_with_pts_refined is not None:
        get_blended_image(image_color, img_with_pts_refined).save(f"{densefusion_args.result_refine_dir}/{scene_frame_name}.png")
    else:  # save blank img
        get_blended_image(image_color, image_color).save(f"{densefusion_args.result_refine_dir}/{scene_frame_name}.png")

    if img_with_pts_unrefined is not None:
        get_blended_image(image_color, img_with_pts_unrefined).save(f"{densefusion_args.result_wo_refine_dir}/{scene_frame_name}.png")
    else:
        get_blended_image(image_color, image_color).save(f"{densefusion_args.result_wo_refine_dir}/{scene_frame_name}.png")

    with open(f"{densefusion_args.result_refine_dir}/pred_txt/{scene_frame_name}.txt", 'w') as f:
        f.write("model preds:\n")
        for itemid in lst:
            f.write(class_names[int(itemid)])
            f.write('\n')

    print(f"\n saved results to {densefusion_args.result_refine_dir}\n")

    # scio.savemat(f"{result_wo_refine_dir}/{scene_frame_name}.mat", {'poses':my_result_wo_refine})
    # scio.savemat(f"{result_refine_dir}/{scene_frame_name}.mat", {'poses':my_result})
    i = 0
    print(f"Finish {scene_frame_name} keyframe")

    return my_result 


