import os
import torch
# import horovod.torch as hvd
import torch.nn as nn
from torch.nn import functional as F
import sys
import math
import yaml
from einops import rearrange, repeat
import numpy as np
import torch.distributed.nn as distnn
sys.path.append('./')
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, Vposer, PositionalEncoding1D
from nets.loss import CoordLoss, ParamLoss, VQLoss
from utils.smpl import SMPL
from nets.transformer import bulid_transformer_encoder, PositionEmbeddingSine, bulid_transformer
from nets.infonce import AllGather, intra_info_nce_loss, inter_info_nce_loss
from utils.transforms import rot6d_to_axis_angle,rotation_matrix_to_angle_axis
from vpd.vpdencoder_useattn import VPDEncoder

allgather = AllGather.apply

class VPDTeacherModel(nn.Module):
    def __init__(self,cfg):
        super(VPDTeacherModel, self).__init__()
        self.backbone = VPDEncoder()
        self.cfg = cfg
        self.kp_lifting = nn.Linear(2,768)
        self.down_linear = nn.Conv2d(2048 , 256, 1, 1)
        self.conv2d_to_3d = nn.Conv2d(2048, 256*8, 1, 1)
        self.exponent = nn.Parameter(torch.tensor(3, dtype=torch.float32))
        self.pos_embed_1d = PositionalEncoding1D()
        self.conv_3d_coord = nn.Sequential(
            nn.Conv3d(256 + 3, 256, 1, 1),
            # BasicBlock_3D(256, 256)
        ) 
        self.refine_3d = bulid_transformer_encoder(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            num_encoder_layers=cfg.enc_layers,
            normalize_before=False,
        )
        
    
    def get_camera_trans(self, cam_param, meta_info, is_render,focal_length):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(focal_length*focal_length*self.cfg.camera_3d_size*self.cfg.camera_3d_size/(self.cfg.input_img_shape[0]*self.cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(self.cfg.input_img_shape[0]*self.cfg.input_img_shape[1]) / (bbox[:, 2]*bbox[:, 3]).sqrt().cuda()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(self.cfg.input_hm_shape[1])
        y = torch.arange(self.cfg.input_hm_shape[0])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float()
        yy = yy[None, None, :, :].cuda().float()

        x = joint_coord_img[:, :, 0, None, None]
        y = joint_coord_img[:, :, 1, None, None]
        heatmap = torch.exp(
            -(((xx - x) / self.cfg.sigma) ** 2) / 2 - (((yy - y) / self.cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans,focal_length):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans) # B, 6890, 3
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam) # B, 30, 3
        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * focal_length + self.cfg.princpt[0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * focal_length + self.cfg.princpt[1]
        x = x / self.cfg.input_img_shape[1] * self.cfg.output_hm_shape[2]
        y = y / self.cfg.input_img_shape[0] * self.cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def cascade_fc(self, hs, net_list):
        assert len(hs) == len(net_list)
        for i in range(len(hs)):
            if i == 0:
                out = net_list[i](hs[i])
            else:
                offset = net_list[i](torch.cat([hs[i],out],dim=-1))
                out = out + offset
        return out

    def get_relative_depth_anchour(self, k , map_size=8):
        range_arr = torch.arange(map_size, dtype=torch.float32, device=k.device) / map_size # (0, 1)
        Y_map = range_arr.reshape(1,1,1,map_size,1).repeat(1,1,map_size,1,map_size) 
        X_map = range_arr.reshape(1,1,1,1,map_size).repeat(1,1,map_size,map_size,1) 
        Z_map = torch.pow(range_arr, k)
        Z_map = Z_map.reshape(1,1,map_size,1,1).repeat(1,1,1,map_size,map_size) 
        return torch.cat([Z_map, Y_map, X_map], dim=1) # 1, 3, 8, 8, 8

    def forward(self, inputs):
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(inputs['joints'].detach())
            # remove blob centered at (0,0) == invalid ones
            input_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
        # img = self.backbone(inputs['img'], skip_early=False)
        # img = self.pose2feat(img, input_heatmap)
        # img = self.backbone(img, skip_early=True)
        context = self.kp_lifting(inputs['joints'][...,:2]/self.cfg.input_hm_shape[0])
        img = self.backbone(x = inputs['img'], control=input_heatmap,context=context)             #B, 2048, 8, 8
        dense_feat = self.conv2d_to_3d(img)
        dense_feat = rearrange(dense_feat, 'b (c d) h w -> b c d h w', c=256, d=8)
        exponent = torch.clamp(self.exponent, 1, 20)
        relative_depth_anchour = self.get_relative_depth_anchour(exponent)
        cam_anchour_maps = repeat(relative_depth_anchour, 'n c d h w -> (b n) c d h w', b=dense_feat.size(0))
        dense_feat = torch.cat([dense_feat, cam_anchour_maps], dim=1)
        dense_feat = self.conv_3d_coord(dense_feat)
        dense_feat = rearrange(dense_feat, 'b c d h w -> (d h w) b c', c=256, d=8).contiguous()
        pos_3d = repeat(self.pos_embed_1d.pos_table, 'n c -> n b c', b=inputs['img'].size(0))
        dense_feat = self.refine_3d(dense_feat, pos=pos_3d)
        dense_feat = rearrange(dense_feat, '(d h w) b c -> b c d h w', d=8, h=8, w=8).contiguous()
        img = self.down_linear(img)
        # img = rearrange(img, 'b c h w -> (h w) b c')
        
        outputs = {"feat_3d": dense_feat,
                   "feat_2d": img}

        return outputs

def mlp(in_feat, out_feat, layers):
    net = []
    for _ in range(layers):
        net.append(nn.Linear(in_feat, in_feat))
        net.append(nn.ReLU())
    net.append(nn.Linear(in_feat, out_feat))
    return nn.Sequential(*net)

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(joint_num, mode, cfg):

    backbone = ResNetBackbone(cfg.resnet_type)
    pose2feat = Pose2Feat(joint_num,feat_dim=64)
    # vposer = Vposer()
    vqcfgpath = os.path.join("common/vqvae/config/pct_vqvae.yaml")
    with open(vqcfgpath, 'r', encoding='utf-8') as f:
        vqcfgfile = f.read()
    vqconfig = yaml.load(vqcfgfile,Loader=yaml.FullLoader)
    vqvae = PCT_VQVAE_MODEL(vqconfig,"common/vqvae/models/256x2048x48.pth")
    for param in vqvae.parameters():
        param.requires_grad = False

    if mode == 'train':
        pose2feat.apply(init_weights)
   
    model = VQ_Model(backbone, pose2feat, vqvae, cfg=cfg)
    return model

