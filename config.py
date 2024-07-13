import os
import os.path as osp
import sys
import numpy as np
import datetime
import yaml
import shutil
import glob
from easydict import EasyDict as edict
import torch.distributed as dist

class Config:
    save_folder = '/path/to/your/save/folder'
    resume_ckpt = 'checkpoints/3dpw_best_ckpt.pth.tar'
    with_contrastive = False
    ## dataset
    # MuCo, Human36M, MSCOCO, PW3D, FreiHAND
    trainset_3d = ['Human36M','MuCo']  # 'Human36M', 'MuCo'
    trainset_2d = ['CrowdPose','MSCOCO']  # 'MSCOCO', 'MPII', 'CrowdPose'
    testset = 'PW3D'  # 'MuPoTs' 'MSCOCO' Human36M, MSCOCO, 'PW3D'

    ## render
    renderimg = 'testvideo'

    ## model setting
    resnet_type = 50  # 50, 101, 152
    frozen_bn = False
    distributed = False
    upsample_net = False
    use_cls_token = False # if True use cls token else mean pooling
    num_layers = 6
    enc_layers = 3
    dec_layers = 3
    local_rank = 0
    max_norm = 10
    weight_decay = 0
    is_local = False
    update_bbox = False
    img_scale_factor = 1.
    ## input, output
    input_img_shape = (256, 256)  #(256, 192)
    input_hm_shape = (32,32)
    output_hm_shape = (64, 64, 64)
    bbox_3d_size = 2 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 0.3
    sigma = 2.5
    focal = None # virtual focal lengths
    princpt = (input_img_shape[1] / 2, input_img_shape[0] / 2)  # virtual principal point position

    J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
    H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

    ## training config
    lr_dec_epoch = [15] if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else [17,21]
    end_epoch = 20 #13 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 25
    lr = 1e-4
    lr_backbone = 1e-4
    lr_dec_factor = 10
    train_batch_size = 4
    use_gt_info = True

    ## testing config
    test_batch_size = 8
    crowd = False
    vis = False
    render = False

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    bbox_3d_size = 2
    camera_3d_size = 2.5
    
    ## loss weight
    joints_3d_weight = 16.
    joints_2d_weight = 0.125
    vertices_weight = 5.
    smpl_weight = 10.
    shape_weight = 1.

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir)
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')


    output_dir = osp.join(output_dir, save_folder)
    print('output dir: ', output_dir)

    model_dir = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    vqtokens = 24
    vqclasses = 2048

    def set_args(self, continue_train=False, is_test=False, resume_ckpt=''):
        if not is_test:
            self.continue_train = continue_train
            if self.continue_train:
                if resume_ckpt and osp.exists(resume_ckpt):
                    shutil.copy(resume_ckpt, osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))

                # else:
                #     shutil.copy(osp.join(cfg.root_dir, 'data', 'snapshot_0.pth.tar'), osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))
                
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

    def update(self, args, folder_dict=None):
        for arg in vars(args):
            setattr(cfg, arg, getattr(args, arg))

        with open(args.cfg) as f:
            exp_config = edict(yaml.safe_load(f))
            for k, v in exp_config.items():
                if hasattr(args, k):
                    v = getattr(args, k)
                setattr(cfg, k, v)
               
        if folder_dict is not None:
            for k, v in folder_dict.items():
                setattr(cfg, k, v)
        return exp_config


cfg = Config()
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
# dataset_list = ['CrowdPose', 'Human36M', 'MSCOCO', 'MuCo', 'PW3D']
dataset_list = ['CrowdPose', 'Human36M', 'My_MuCo', 'PW3D']
for i in range(len(dataset_list)):
    add_pypath(osp.join(cfg.data_dir, dataset_list[i]))
