import _init_paths
import argparse
import os
import copy
import cv2
import random
import numpy as np
from PIL import Image 
import scipy.io as scio
import numpy.ma as ma
import pickle
from scipy.spatial.transform import Rotation as sciR
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
# from datasets.ycb.dataset import PoseDataset
from lib.densefusion.network import PoseNet, PoseRefineNet
from lib.densefusion.transformations import quaternion_matrix, quaternion_from_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'datasets/pandas/', help='dataset root dir')
parser.add_argument('--model', type=str, default = '_trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '_trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

img_width = 480
img_length = 848


torch.backends.cudnn.enabled = False

xmap = np.array([[j for i in range(img_length)] for j in range(img_width)])
ymap = np.array([[i for i in range(img_length)] for j in range(img_width)])
num_obj = 21
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/pandas/dataset_config'
ycb_toolbox_dir = '_YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/pandas/Densefusion_wo_refine_result' #'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/pandas/Densefusion_iterative_result' #'experiments/eval_result/ycb/Densefusion_iterative_result'



def get_rgb_image(image, max_val=255.0):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGB",
    ).convert("RGBA")
    return img   

def get_blended_image(img, anno_img, max_val=255.0, alpha=0.5):
    og_img_viz = get_rgb_image(img)
    anno_img_viz = get_rgb_image(anno_img)

    return Image.blend(og_img_viz, anno_img_viz, alpha=0.5)

def get_bbox(posecnn_rois):
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

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()


# construct list of frames to test
testlist = []
# input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))   # ***************************
# while 1:
#     input_line = input_file.readline()
#     if not input_line:
#         break
#     if input_line[-1:] == '\n':
#         input_line = input_line[:-1]
#     testlist.append(input_line)
# input_file.close()
# print(len(testlist))
DATAPATH = os.path.join(opt.dataset_root, 'data/panda_data_pik3_tiny')
for scene_frame_pik in os.listdir(DATAPATH):  # 000001-000001.pik
    # scene_frame_name = scene_frame_pik.split("/")[-1].split(".")[0]
    scene_frame_name = scene_frame_pik
    testlist.append(scene_frame_name)
testlist.sort()
print(len(testlist))


class_file = open('{0}/classes.txt'.format(dataset_config_dir))  # same as YCB
class_id = 1 
cld = {}
class_names = ['0']
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]
    class_names.append(class_input)
    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
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

for i, scene_frame_pik in enumerate(testlist):
    
    # img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    # depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    # posecnn_meta = scio.loadmat('{0}/results_PoseCNN_pandas/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    # label = np.array(posecnn_meta['labels'])  # segmentation label 480x640  (unique: [labels,in,img])
    # posecnn_rois = np.array(posecnn_meta['rois'])
    scene_frame_pik_dir = os.path.join(DATAPATH, scene_frame_pik)  # full path to .pik file
    with open(scene_frame_pik_dir, 'rb') as file:
        data = pickle.load(file)
    scene_frame_name = scene_frame_pik.split(".")[0]  # 000001-000001

    if ('5' in scene_frame_name and int(scene_frame_name[-1]) == 7):
        from IPython import embed; embed()

    print(f"****************Processing {scene_frame_name}*******************")

    img = data['rgb']  # color
    depth = data['depth'] / data['factor_depth']
    K = data['intrinsics']
    # depth_factor = data['factor_depth']

    cam_cx = K[0][-1]   
    cam_cy = K[1][-1]
    cam_fx = K[0][0]
    cam_fy = K[1][1]
    cam_scale = 1 #10000.0
    img_width, img_length = depth.shape

    posecnn_meta = scio.loadmat(f'{opt.dataset_root}/data/results_PoseCNN_pandas/{scene_frame_name}.mat')
    label = np.array(posecnn_meta['labels'])  # segmentation label 480x640  (unique: [labels,in,img])
    posecnn_rois = np.array(posecnn_meta['rois'])


    # from IPython import embed; embed() #**********************************************

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
    
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
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

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            
            # from IPython import embed; embed()

            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
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
                img_with_pts_unrefined = np.array(img)
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

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

            my_result.append(my_pred.tolist())


            ##################
            #  VIZ REFINED RESULTS (https://github.com/j96w/DenseFusion/issues/27)
            ##################
            mat_r= quaternion_matrix(my_r)[0:3,0:3]
            imgpts, jac = cv2.projectPoints(cld[itemid], mat_r, my_t, K, np.array([[0., 0.0,  0.0, 0.0, 0.0]]))

            if img_with_pts_refined is None:
                img_with_pts_refined = np.array(img)
            for pt_coord in np.int32(imgpts):
                x,y = pt_coord[0]
                img_with_pts_refined = cv2.circle(img_with_pts_refined, (x,y), 1, (0,255,255))   

        except (ZeroDivisionError, ValueError, IndexError):
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, i))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])


    ## SAVE VIZ
    if img_with_pts_refined is not None:
        get_blended_image(img, img_with_pts_refined).save(f"{result_refine_dir}/{scene_frame_name}.png")
    else:  # save blank img
        get_blended_image(img, img).save(f"{result_refine_dir}/{scene_frame_name}.png")

    if img_with_pts_unrefined is not None:
        get_blended_image(img, img_with_pts_unrefined).save(f"{result_wo_refine_dir}/{scene_frame_name}.png")
    else:
        get_blended_image(img, img).save(f"{result_wo_refine_dir}/{scene_frame_name}.png")

    with open(f"{result_refine_dir}/pred_txt/{scene_frame_name}.txt", 'w') as f:
        f.write("model preds:\n")
        for itemid in lst:
            f.write(class_names[int(itemid)])
            f.write('\n')


    # scio.savemat(f"{result_wo_refine_dir}/{scene_frame_name}.mat", {'poses':my_result_wo_refine})
    # scio.savemat(f"{result_refine_dir}/{scene_frame_name}.mat", {'poses':my_result})
    print(f"Finish No.{i+1}/{len(testlist)} keyframe")

    # from IPython import embed; embed()
from IPython import embed; embed()