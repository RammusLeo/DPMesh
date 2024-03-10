
import torch
import torch.nn as nn
import numpy as np
import math
import os.path as osp
from collections import OrderedDict
from utils.transforms import rot6d_to_axis_angle,rot6d_to_rotmat,rotation_matrix_to_angle_axis

class Regressor(nn.Module):
    def __init__(self, smpl_mean_params, img_feat_num=2048):
        super(Regressor, self).__init__()
        npose = 24 * 6
        nshape = 10
        ncam = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + npose + nshape + ncam
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, xf, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = xf.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            # print(xf.shape, pred_pose.shape, pred_shape.shape, pred_cam.shape)
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)

            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 3, 3)
        pred_rotmat = rotation_matrix_to_angle_axis(pred_rotmat).reshape(batch_size,-1)

        return pred_rotmat, pred_shape, pred_cam