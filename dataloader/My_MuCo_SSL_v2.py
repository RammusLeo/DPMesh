import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import pickle
import transforms3d
from pycocotools.coco import COCO
from config import cfg
import lmdb
from common.utils.tsv_file import TSVFile, CompositeTSVFile
from common.utils.image_ops import img_from_base64
from common.utils.posefix import replace_joint_img
from common.utils.tsv_file_ops import load_linelist_file
from common.utils.smpl import SMPL
from torchvision import transforms
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, compute_iou, load_img_from_lmdb,addocc
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
# from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, vis_bbox
import transforms3d


class My_MuCo_SSL_v2(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        print('='*20, 'MuCo', '='*20)
        self.transform_hard = transform
        self.transform_basic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.data_split = data_split
        self.img_dir = osp.join(cfg.root_dir, 'data', 'MuCo')
        self.annot_path = osp.join(cfg.root_dir, 'data', 'MuCo', 'MuCo-3DHP.json')
        self.annot_path_new = osp.join('data', 'annotations', 'muco3.pkl')
        self.smpl_param_path = osp.join(cfg.root_dir, 'data', 'MuCo', 'smpl_param.json')
        self.img_file = osp.join(cfg.metro_dir, 'muco', 'train.img.tsv')
        self.hw_file = osp.join(cfg.metro_dir, 'muco', 'train.hw.tsv')
        self.img_tsv = self.get_tsv_file(self.img_file)
        self.hw_tsv = self.get_tsv_file(self.hw_file)
        self.fitting_thr = 25 # milimeter

        self.linelist_file = osp.join(cfg.metro_dir, 'muco', 'train.linelist.tsv')
        self.line_list = load_linelist_file(self.linelist_file)

        # COCO joint set
        self.coco_joint_num = 17  # original: 17
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')

        # MuCo joint set
        self.muco_joint_num = 21
        self.muco_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.muco_flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.muco_skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.muco_root_joint_idx = self.muco_joints_name.index('Pelvis')
        self.muco_coco_common_jidx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

        # H36M joint set
        self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'Human36M', 'J_regressor_h36m_correct.npy')) # use h36m joint regrssor (only use subset from original muco joint set)
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.datalist = self.load_data()
        print("muco data len: ", len(self.datalist))

        self.image_keys = self.prepare_image_key_to_index()
        for k in list(self.image_keys.keys())[:5]:
            print(k)

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def prepare_image_key_to_index(self):
        tsv = self.hw_tsv
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def get_tsv_file(self, tsv_path):
        # if tsv_file:
        #     if self.is_composite:
        #         return CompositeTSVFile(tsv_file, self.linelist_file,
        #                 root=self.root)
        #     tsv_path = find_file_path_in_yaml(tsv_file, self.root)
        return TSVFile(tsv_path)

    def get_image(self, idx): 
        line_no = self.get_line_no(idx)
        row = self.img_tsv[line_no]
        # use -1 to support old format with multiple columns.
        cv2_im = img_from_base64(row[-1])
        # if self.cv2_output:
        #     return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        return cv2_im

    def load_data(self):
        with open(self.annot_path_new,'rb') as f:
            datalist = pickle.load(f)
        return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1, -1);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # flip smpl pose parameter (axis-angle)
        if do_flip:
            smpl_pose = smpl_pose.view(-1, 3)
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):  # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
            smpl_pose[:, 1:3] *= -1;  # multiply -1 to y and z axis of axis-angle
            smpl_pose = smpl_pose.view(1, -1)

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.
        return smpl_pose[0].numpy(), smpl_shape[0].numpy()

    def get_fitting_error(self, muco_joint, smpl_mesh, do_flip):
        muco_joint = muco_joint.copy()
        muco_joint = muco_joint - muco_joint[self.muco_root_joint_idx,None,:] # root-relative
        if do_flip:
            muco_joint[:,0] = -muco_joint[:,0]
            for pair in self.muco_flip_pairs:
                muco_joint[pair[0],:] , muco_joint[pair[1],:] = muco_joint[pair[1],:].copy(), muco_joint[pair[0],:].copy()
        muco_joint_valid = np.ones((self.muco_joint_num,3), dtype=np.float32)
      
        # transform to h36m joint set
        h36m_joint = transform_joint_to_other_db(muco_joint, self.muco_joints_name, self.h36m_joints_name)
        h36m_joint_valid = transform_joint_to_other_db(muco_joint_valid, self.muco_joints_name, self.h36m_joints_name)
        h36m_joint = h36m_joint[h36m_joint_valid==1].reshape(-1,3)

        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl[h36m_joint_valid==1].reshape(-1,3)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:] # translation alignment
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # if not hasattr(self, 'aug_lmdb'):
        #     aug_db_path = osp.join(cfg.root_dir, 'data', 'MuCo', 'data', 'augmented_set_lmdb', 'augmented_set_lmdb.lmdb')
        #     aug_env = lmdb.open(aug_db_path,
        #                     subdir=os.path.isdir(aug_db_path),
        #                     readonly=True, lock=False,
        #                     readahead=False, meminit=False)
        #     self.aug_lmdb = aug_env.begin(write=False)
        #     unaug_db_path = osp.join(cfg.root_dir, 'data', 'MuCo', 'data', 'unaugmented_set_lmdb', 'unaugmented_set_lmdb.lmdb')
        #     unaug_env = lmdb.open(unaug_db_path,
        #                     subdir=os.path.isdir(unaug_db_path),
        #                     readonly=True, lock=False,
        #                     readahead=False, meminit=False)
        #     self.unaug_lmdb = unaug_env.begin(write=False)
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
        if cfg.update_bbox:
            height, width = img_shape
            bbox = process_bbox(data['tight_bbox'], width, height)
        # img
        # is_aug = img_path.split('/')[-3] 
        # if is_aug == 'augmented_set':
        #     img = load_img_from_lmdb(img_path, self.aug_lmdb)
        # elif is_aug == 'unaugmented_set':
        #     img = load_img_from_lmdb(img_path, self.unaug_lmdb)
        # else:
        #     raise NotImplementedError('Unknown dataset: {}'.format(is_aug))
        # img = load_img(img_path)
        try:
            img = self.get_image(self.image_keys[img_path])
        except KeyError:
            return None
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split,occ=False)
        img_teacher = self.transform_basic(img.astype(np.float32))
        img_student = addocc(img,bbox)
        img_student = self.transform_hard(img_student.astype(np.uint8))
        
        # muco gt
        muco_joint_img = data['joint_img']
        muco_joint_cam = data['joint_cam']
        muco_joint_cam = muco_joint_cam - muco_joint_cam[self.muco_root_joint_idx,None,:] # root-relative
        muco_joint_valid = data['joint_valid']
        if do_flip:
            muco_joint_img[:,0] = img_shape[1] - 1 - muco_joint_img[:,0]
            muco_joint_cam[:,0] = -muco_joint_cam[:,0]
            for pair in self.muco_flip_pairs:
                muco_joint_img[pair[0],:], muco_joint_img[pair[1],:] = muco_joint_img[pair[1],:].copy(), muco_joint_img[pair[0],:].copy()
                muco_joint_cam[pair[0],:], muco_joint_cam[pair[1],:] = muco_joint_cam[pair[1],:].copy(), muco_joint_cam[pair[0],:].copy()
                muco_joint_valid[pair[0],:], muco_joint_valid[pair[1],:] = muco_joint_valid[pair[1],:].copy(), muco_joint_valid[pair[0],:].copy()

        muco_joint_img_xy1 = np.concatenate((muco_joint_img[:,:2], np.ones_like(muco_joint_img[:,:1])),1)
        muco_joint_img[:,:2] = np.dot(img2bb_trans, muco_joint_img_xy1.transpose(1,0)).transpose(1,0)
        # for swap
        if len(data['near_joints']) > 0:
            near_joint_list = []
            for nj in data['near_joints']:
                near_joint = np.ones((self.coco_joint_num, 3), dtype=np.float32)
                nj_xy1 = np.concatenate((nj[:, :2], np.ones_like(nj[:, :1])), axis=1)
                near_joint[:, :2] = np.dot(img2bb_trans, nj_xy1.transpose(1,0)).transpose(1,0)
                near_joint_list.append(near_joint)
            near_joints = np.asarray(near_joint_list, dtype=np.float32)
        else:
            near_joints = np.zeros((1, self.coco_joint_num, 3), dtype=np.float32)

        input_muco_joint_img = muco_joint_img.copy()
        muco_joint_img[:,0] = muco_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        muco_joint_img[:,1] = muco_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        muco_joint_img[:,2] = muco_joint_img[:,2] - muco_joint_img[self.muco_root_joint_idx][2] # root-relative
        muco_joint_img[:,2] = (muco_joint_img[:,2] / (cfg.bbox_3d_size * 1000 / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter

        # check truncation
        muco_joint_trunc = muco_joint_valid * ((muco_joint_img[:,0] >= 0) * (muco_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                    (muco_joint_img[:,1] >= 0) * (muco_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                    (muco_joint_img[:,2] >= 0) * (muco_joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

        # transform muco joints to target db joints
        muco_joint_img = transform_joint_to_other_db(muco_joint_img, self.muco_joints_name, self.joints_name)
        muco_joint_cam = transform_joint_to_other_db(muco_joint_cam, self.muco_joints_name, self.joints_name)
        muco_joint_valid = transform_joint_to_other_db(muco_joint_valid, self.muco_joints_name, self.joints_name)
        muco_joint_trunc = transform_joint_to_other_db(muco_joint_trunc, self.muco_joints_name, self.joints_name)

        # apply PoseFix
        input_muco_joint_img[:, 2] = 1 # joint valid
        tmp_joint_img = transform_joint_to_other_db(input_muco_joint_img, self.muco_joints_name, self.coco_joints_name)
        tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], near_joints, data['num_overlap'], img2bb_trans)
        tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.muco_joints_name)
        input_muco_joint_img[self.muco_coco_common_jidx, :2] = tmp_joint_img[self.muco_coco_common_jidx, :2]
        """
        # debug PoseFix result
        newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), input_muco_joint_img.T, self.muco_skeleton)
        cv2.imshow(f'{img_path}', newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        """
        input_muco_joint_img[:, 0] = input_muco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        input_muco_joint_img[:, 1] = input_muco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        input_muco_joint_img = transform_joint_to_other_db(input_muco_joint_img, self.muco_joints_name, self.joints_name)

        if smpl_param is not None:
            # smpl coordinates
            smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
            # smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
            # smpl_coord_img = cam2pixel(smpl_coord_cam, cam_param['focal'], cam_param['princpt'])
            if do_flip:
                smpl_coord_img = data['smpl_coord_img_raw']
                smpl_coord_cam = data['smpl_coord_cam_raw']
            else:
                smpl_coord_img = data['smpl_coord_img_flip']
                smpl_coord_cam = data['smpl_coord_cam_flip']
            # x,y affine transform, root-relative depth
            smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:, :2], np.ones_like(smpl_coord_img[:, 0:1])), 1)
            smpl_coord_img[:, :2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            smpl_joint_cam = smpl_coord_cam
            smpl_coord_img[:, 2] = smpl_coord_img[:, 2] - smpl_coord_cam[self.root_joint_idx][2]
            smpl_coord_img[:, 0] = smpl_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            smpl_coord_img[:, 1] = smpl_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            smpl_coord_img[:, 2] = (smpl_coord_img[:, 2] / (cfg.bbox_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]

            # check truncation
            smpl_trunc = (
                        (smpl_coord_img[:, 0] >= 0) * (smpl_coord_img[:, 0] < cfg.output_hm_shape[2]) * (smpl_coord_img[:, 1] >= 0) * (smpl_coord_img[:, 1] < cfg.output_hm_shape[1]) * (smpl_coord_img[:, 2] >= 0) * (
                            smpl_coord_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(np.float32)

            # split mesh and joint coordinates
            # smpl_mesh_img = smpl_coord_img[:self.vertex_num];
            smpl_joint_img = smpl_coord_img
            # smpl_mesh_trunc = smpl_trunc[:self.vertex_num];
            smpl_joint_trunc = smpl_trunc

            # already checked in load_data()
            # is_valid_fit = True
            
        else:
            smpl_joint_img = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            smpl_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            smpl_mesh_img = np.zeros((self.vertex_num,3), dtype=np.float32) # dummy
            smpl_pose = np.zeros((72), dtype=np.float32) # dummy
            smpl_shape = np.zeros((10), dtype=np.float32) # dummy
            smpl_joint_trunc = np.zeros((self.joint_num,1), dtype=np.float32) # dummy
            smpl_mesh_trunc = np.zeros((self.vertex_num,1), dtype=np.float32) # dummy
            # is_valid_fit = False
        is_valid_fit = data['is_valid_fit']
        # 3D data rotation augmentation
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
        [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
        [0, 0, 1]], dtype=np.float32)
        # muco coordinate
        muco_joint_cam = np.dot(rot_aug_mat, muco_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter
        # parameter
        smpl_pose = smpl_pose.reshape(-1,3)
        root_pose = smpl_pose[self.root_joint_idx,:]
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
        smpl_pose = smpl_pose.reshape(-1)
        # smpl coordinate
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[self.root_joint_idx,None] # root-relative
        smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter

        # SMPL pose parameter validity
        smpl_param_valid = np.ones((self.smpl.orig_joint_num, 3), dtype=np.float32)
        for name in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
            smpl_param_valid[self.joints_name.index(name)] = 0
        smpl_param_valid = smpl_param_valid.reshape(-1)
        cliff_focal_length = np.sqrt(img_shape[0]*img_shape[0]+img_shape[1]*img_shape[1])

        mask_prop = np.random.rand(muco_joint_trunc.shape[0],muco_joint_trunc.shape[1])  # B, N, 1
        mask = mask_prop < 0.2 # 10% of joints are masked
        joint_mask_student = muco_joint_trunc * (1 - mask.astype(np.float32))


        # images = {'img_teacher': img_teacher, 'img_student': img_student}
        # masks = {'mask_teacher': muco_joint_trunc, 'mask_student': joint_mask_student}
        # inputs = {'joints': input_muco_joint_img[:, :2]}
        teacher_inputs = {'img': img_teacher, 'joints_mask': muco_joint_trunc, 'joints': muco_joint_img}
        student_inputs = {'img': img_student, 'joints_mask': joint_mask_student, 'joints': input_muco_joint_img}
        targets = {'orig_joint_img': muco_joint_img, 'fit_joint_img': smpl_joint_img, 'orig_joint_cam': muco_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
        meta_info = {'cliff_focal_length': cliff_focal_length,'orig_joint_valid': muco_joint_valid, 'orig_joint_trunc': muco_joint_trunc, 'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc, 'is_valid_fit': float(is_valid_fit), 'is_3D': float(True)}

        return student_inputs, teacher_inputs,  targets, meta_info

