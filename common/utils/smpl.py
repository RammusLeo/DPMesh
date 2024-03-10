import numpy as np
import torch
import os.path as osp
import json
from config import cfg

import sys
sys.path.insert(0, cfg.smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from utils.transforms import  build_adj, normalize_adj, transform_joint_to_other_db,batch_rodrigues


class SMPL(object):
    def __init__(self):
        self.neutral = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
        self.male = SMPL_Layer(gender='male', model_root=cfg.smpl_path + '/smplpytorch/native/models')
        self.female = SMPL_Layer(gender='female', model_root=cfg.smpl_path + '/smplpytorch/native/models')
        self.layer = {'neutral': self.get_layer(), 'male': self.get_layer('male'), 'female': self.get_layer('female')}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].th_faces.numpy()
        self.joint_regressor = self.layer['neutral'].th_J_regressor.numpy()
        self.shape_param_dim = 10
        self.vposer_code_dim = 32

        # add nose, L/R eye, L/R ear,
        self.face_kps_vertex = (331, 2802, 6262, 3489, 3990) # mesh vertex idx
        nose_onehot = np.array([1 if i == 331 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        left_eye_onehot = np.array([1 if i == 2802 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        right_eye_onehot = np.array([1 if i == 6262 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        left_ear_onehot = np.array([1 if i == 3489 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        right_ear_onehot = np.array([1 if i == 3990 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor = np.concatenate((self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))
        # add head top
        self.joint_regressor_extra = np.load(osp.join(cfg.root_dir, 'data', 'J_regressor_extra.npy'))
        self.joint_regressor = np.concatenate((self.joint_regressor, self.joint_regressor_extra[3:4, :])).astype(np.float32)
        self.J_regressor_h36m_correct = torch.from_numpy(np.load(osp.join(cfg.root_dir, 'data', 'Human36M', 'J_regressor_h36m_correct.npy')))

        self.orig_joint_num = 24
        self.joint_num = 30 # original: 24. manually add nose, L/R eye, L/R ear, head top
        self.six_locate_center = ('Pelvis', 'L_Hip', 'R_Hip', 'L_Shoulder', 'R_Shoulder', 'Neck')
        self.orig_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax',
                            'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax',
                            'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
        self.joints_meaning = ('Pelvis', 'Left_Hip', 'Right_Hip', 'Torso', 'Left_Knee', 'Right_Knee', 'Spine', 
                            'Left_Ankle', 'Right_Ankle', 'Chest', 'Left_Toe', 'Right_Toe', 'Neck', 
                            'Left_Thorax', 'Right_Thorax', 'Head', 'Left_Shoulder', 'Right_Shoulder', 
                            'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist', 'Left_Hand', 'Right_Hand', 
                            'Nose', 'Left_Eye', 'Right_Eye', 'Left_Ear', 'Right_Ear', 'Head_top')
        self.flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )
        # self.joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        # 'L_Elbow','L_Wrist','Neck','Head_top','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        # self.flip_pairs = ((0,5),(1,4),(2,3),(6,11),(7,10),(8,9),(20,21),(22,23))
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.skeleton = ( (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28), (24,29) )
         

        # joint set for PositionNet prediction
        self.graph_joint_num = 15
        self.graph_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'Head_top', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist')
        self.graph_flip_pairs = ((1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14))
        self.graph_skeleton = ((0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (4, 6), (0, 7), (7, 8), (7, 9), (9, 11), (11, 13), (7, 10), (10, 12), (12, 14))
        
        self.common_joints_name = ('R_Ankle','R_Knee','R_Hip','L_Hip', 'L_Knee',  'L_Ankle','R_Wrist','R_Elbow', 'R_Shoulder','L_Shoulder', 'L_Elbow', 'L_Wrist',  'Neck', 'Head_top')
        # construct graph adj
        self.graph_adj = self.get_graph_adj()

        self.idx_list_15 = []
        for name in self.graph_joints_name:
            idx = self.joints_name.index(name)
            self.idx_list_15.append(idx)
        self.idx_list_6 = []
        for name in self.six_locate_center:
            idx = self.joints_name.index(name)
            self.idx_list_6.append(idx)

    def reduce_joint_set(self, joint):
        # new_joint = []
        # for name in self.graph_joints_name:
        #     idx = self.joints_name.index(name)
        #     new_joint.append(joint[:,idx,:])
        # new_joint = torch.stack(new_joint,1)
        # return new_joint
        return joint[:,self.idx_list_15,:].contiguous()

    def get_graph_adj(self):
        adj_mat = build_adj(self.graph_joint_num, self.graph_skeleton, self.graph_flip_pairs)
        normalized_adj = normalize_adj(adj_mat)
        return normalized_adj

    def get_layer(self, gender='neutral'):
        if gender == 'neutral':
            return self.neutral
        elif gender == 'male':
            return self.male
        elif gender == 'female':
            return self.female
        else:
            raise ValueError('Gender invalid input:' + gender)

    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3) # (batch_size * 24, 1, 3)
            R = batch_rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:,1:,:] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890, 3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i-1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1,0,2,3).contiguous().view(24,-1)).view(6890, batch_size, 4, 4).transpose(0,1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        joints_extra = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra])
        joints = torch.cat((joints, joints_extra), dim=1)
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input