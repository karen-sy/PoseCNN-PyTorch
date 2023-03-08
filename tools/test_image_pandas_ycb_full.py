import os
import numpy as np
import pickle

from test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_rgb_image, get_image_densefusion, get_blended_image, get_bbox, env_setup_densefusion, run_DenseFusion
import easydict
from fcn.config import cfg, cfg_from_file


def main(image_color_rgb, image_color_bgr, image_depth, meta_data):
    #############
    # Run Test on Image(s)
    #############

    posecnn_meta = run_posecnn(image_color_bgr, image_depth, meta_data, posecnn_network, dataset_cfg, posecnn_cfg)    
    return run_DenseFusion(image_color_rgb, image_depth, meta_data,
                    df_estimator, df_refiner, 
                    scene_frame_name=scene_data_dirs[test_idx].split(".")[0], 
                    class_names=class_names, 
                    cld=cld,
                    densefusion_args=densefusion_args,
                    posecnn_meta=posecnn_meta)


if __name__ == '__main__':
    #############
    # Setup PoseCNN
    #############
    posecnn_args = easydict.EasyDict({
        'gpu_id':0, 
        'pretrained':'trained_checkpoints/posecnn/checkpoints/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth', 
        'cfg_file':'experiments/cfgs/ycb_object.yml', 
        'dataset_name':'ycb_object_test', 
        'depth_name':None, # will process results from pik instead 
        'datadir':'datasets/pandas/data/new_panda/', # provide a dir containing piks
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

    dataset_cfg, posecnn_network = env_setup_posecnn(posecnn_args, posecnn_cfg)

    #############
    # Setup DenseFusion
    #############
    densefusion_args = easydict.EasyDict({
        'dataset_root': 'datasets/pandas',
        'model': 'trained_checkpoints/densefusion/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth',
        'refine_model': 'trained_checkpoints/densefusion/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth',
        'dataset_config_dir':'datasets/pandas/dataset_config',
        'ycb_toolbox_dir':None, #'YCB_Video_toolbox',
        'result_wo_refine_dir':'experiments/eval_result/pandas/Densefusion_wo_refine_result', #'experiments/eval_result/ycb/Densefusion_wo_refine_result'
        'result_refine_dir':'experiments/eval_result/pandas/Densefusion_iterative_result' #'experiments/eval_result/ycb/Densefusion_iterative_result'
    })

    df_estimator, df_refiner, class_names, cld = env_setup_densefusion(densefusion_args)

    #############
    # Load Test Image
    #############
    scene_data_dirs = os.listdir(posecnn_args.datadir)
    scene_data_dirs.sort()
    print("all scene data:", scene_data_dirs)



    # test_idx = np.random.randint(len(scene_data_dirs))

    for test_idx in range(len(scene_data_dirs)):
        test_filename_pik = posecnn_args.datadir + scene_data_dirs[test_idx]

        # if 'scene' not in test_filename_pik:
        #     continue

        with open(test_filename_pik, 'rb') as file:
            test_data = pickle.load(file)
        image_color_bgr, image_depth, meta_data = get_image_posecnn(test_data)  # BGR
        image_color_bgr = image_color_bgr[:,:,:3]
        image_color_rgb, _, _ = get_image_densefusion(test_data)
        image_color_rgb = image_color_rgb[:,:,:3]
        # from IPython import embed; embed()
    ###########
    # Run PoseCNN + DenseFusion
    ###########
        print(f"\n Running models on {test_filename_pik}..")
        result = main(image_color_rgb, image_color_bgr, image_depth, meta_data)

    from IPython import embed; embed()