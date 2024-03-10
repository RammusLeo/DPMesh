# import torch
import torch.nn as nn
import torch.nn.modules.transformer
import torch.nn.functional as F
from loguru import logger
import numpy as np
import os
from os import listdir, mkdir, path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from common.utils.transforms import batch_convert_to_rotmat,rotmat_to_rot6d
from common.vqvae.vae_utils.pctvqvae import PCTVQVAE

class PCT_VQVAE_6D(PCTVQVAE):
    def __init__(self,cfg,initpath=None):
        self.cfg = cfg
        self.initpath = initpath
        super().__init__(
            cfg=cfg,
            num_joints=23,
            input_dim=6
        )
        if initpath:
            self.init_weight()

    def forward(self,mode, x, target=None):
        
        gt_pose = target["gt_pose"][:,3:]
        B,C = gt_pose.shape
        # prex = gt_pose.permute(0,2,1).contiguous().float()
        # prex = F.avg_pool1d(prex, kernel_size=T)

        gt_pose_mat = batch_convert_to_rotmat(gt_pose.squeeze(-1),rep='aa').reshape(B,-1,9)      #B, 24, 9
        gt_pose_6d = rotmat_to_rot6d(gt_pose_mat).reshape(B,-1,6)

        recoverd_pose, encoding_indices, e_latent_loss =super().forward(gt_pose_6d)

        outputs = {"pred":recoverd_pose,"gt":target["gt_pose"]}
        return outputs
        
    def init_weight(self):
        if self.initpath:
            weight = torch.load(self.initpath, map_location="cpu")["model"]
            for k in list(weight.keys()):
                if k.startswith("smpl"):
                    del weight[k]
            a,b = self.load_state_dict(weight,strict=False)
            print("vq missing",a)
            print("vq unexpected", b)
            print("vqvae model has been loaded from checkpoint!")