import torch
import torch.nn as nn
import torch.nn.modules.transformer
import torch.nn.functional as F
from loguru import logger
import numpy as np
import os
from os import listdir, mkdir, path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from common.utils.transforms import batch_convert_to_rotmat
from common.vqvae.vae_utils.pctvqvae_mask import PCTVQVAE_MASK

class PCT_VQVAE_MASK(PCTVQVAE_MASK):
    def __init__(self,cfg,initpath=None):
        self.cfg = cfg
        self.initpath = initpath
        super().__init__(
            cfg=cfg,
            num_joints=23
        )

        if initpath:
            self.init_weight()

    def mpjpe(self, gt_Jtr, pred_Jtr):
        # B, T, 22, 3
        gt_pelvis = (gt_Jtr[:,[2],:] + gt_Jtr[:,[3],:]) / 2.0
        pred_pelvis = (pred_Jtr[:,[2],:] + pred_Jtr[:,[3],:]) / 2.0

        gt_Jtr-=gt_pelvis
        pred_Jtr-=pred_pelvis
        error = torch.sqrt(((gt_Jtr - pred_Jtr) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy()*1000
        error = np.mean(error)
        return error

    def alignedjntloss(self, gt_keypoints_3d, pred_keypoints_3d):
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return self.loss_smooth(gt_keypoints_3d,pred_keypoints_3d)
    
    def params_group(self):
        return [
            {
                'params': self.parameters(),
                'lr': self.cfg.SCHED.LR
            }
        ]
    

    def forward(self,mode, x, target=None):
        
        gt_pose = target["gt_pose"]
        B,C = gt_pose.shape
        # prex = gt_pose.permute(0,2,1).contiguous().float()
        # prex = F.avg_pool1d(prex, kernel_size=T)

        gt_pose_mat = batch_convert_to_rotmat(gt_pose.squeeze(-1),rep='aa').reshape(B,-1,9)      #B, 24, 9

        recoverd_pose, encoding_indices, e_latent_loss =super().forward(gt_pose_mat)

        loss_reconstruct = self.loss_smooth(recoverd_pose,gt_pose_mat)
        gtfusion = gt_pose_mat.reshape(-1,24,3,3)
        # loss_rottrans = self.loss_smooth(torch.matmul(rotmat_hat, torch.transpose(rotmat_hat, 1,2)), torch.eye(3,3).to(rotmat_hat))
        # loss_det = self.loss_smooth(torch.linalg.det(rotmat_hat),torch.tensor([1]).to(rotmat_hat))

        outputs = {"pred":recoverd_pose,"gt":target["gt_pose"]}
        return outputs
    


    def get_latent_code(self,gt_pose):
        B,T,C = gt_pose.shape
        prex = gt_pose.permute(0,2,1).contiguous().float()
        prex = F.avg_pool1d(prex, kernel_size=T)

        gt_pose_mat = batch_convert_to_rotmat(prex.squeeze(-1),rep='aa').reshape(B,-1,9)
        return super().get_latent_code(gt_pose_mat), gt_pose_mat
    
    def get_batch_latent_code(self,gt_pose):
        B,C = gt_pose.shape
        gt_pose_first = batch_convert_to_rotmat(gt_pose,rep='aa').reshape(B,-1,9)
        return super().get_latent_code(gt_pose_first)

    def get_batch_masked_latent_code(self,gt_pose,maskidx):
        B,C = gt_pose.shape
        gt_pose_mat = batch_convert_to_rotmat(gt_pose,rep='aa').reshape(B,-1,9)
        return super().get_masked_latent_code(gt_pose_mat,maskidx)
    
    def get_latent_feat(self,gt_pose):
        B,C = gt_pose.shape
        gt_pose_first = batch_convert_to_rotmat(gt_pose,rep='aa').reshape(B,-1,9)
        return super().get_latent_feat(gt_pose_first).reshape(B,self.cfg["STAGE_I"]["CODEBOOK"]["token_num"],-1)

    def get_decode_pose(self, cls_logits):
        return super().get_decode_pose(cls_logits)
    
    def get_masked_decode_pose(self, cls_logits,maskindices):
        return super().get_masked_decode_pose(cls_logits,maskindices)
        
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