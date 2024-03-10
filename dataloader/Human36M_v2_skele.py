import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
import lmdb
import pickle
from common.utils.posefix import replace_joint_img
from common.utils.smpl import SMPL
from torchvision import transforms
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, load_img_from_lmdb, addocc
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from common.utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton,vis_keypoints_with_skeleton_fortrain
# from utils.vis import vis_mesh, save_obj
from common.utils.openpose_utils import OPENJOINTSNAME,OPENSKELETON,OPENCOLOR


class Human36M_v2_skele(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        print('='*20, 'Human36M', '='*20)
        # self.transform_hard = transform
        # self.transform_basic = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.root_dir, 'data', 'Human36M', 'images')
        self.annot_path = osp.join(cfg.root_dir, 'data', 'Human36M', 'annotations')
        self.annot_path_new = osp.join('data', 'annotations', 'human36m3.pkl')
        self.human_bbox_root_dir = osp.join(cfg.root_dir, 'data', 'Human36M', 'rootnet_output', 'bbox_root_human36m_output.json')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.fitting_thr = 25 # milimeter

        # COCO joint set
        self.coco_joint_num = 17  # original: 17
        self.coco_joints_name = (
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')

        # H36M joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.h36m_skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))
        self.h36m_coco_common_jidx = (1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16)  # for posefix, exclude pelvis

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

        self.openpose_joints_name = OPENJOINTSNAME
        self.openpose_skeleton = OPENSKELETON
        self.openpose_color = OPENCOLOR

        self.datalist = self.load_data()
        print("h36m data len: ", len(self.datalist))

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            # subject = [1]
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
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

    def get_fitting_error(self, h36m_joint, smpl_mesh, do_flip):
        h36m_joint = h36m_joint - h36m_joint[self.h36m_root_joint_idx,None,:] # root-relative
        if do_flip:
            h36m_joint[:,0] = -h36m_joint[:,0]
            for pair in self.h36m_flip_pairs:
                h36m_joint[pair[0],:] , h36m_joint[pair[1],:] = h36m_joint[pair[1],:].copy(), h36m_joint[pair[0],:].copy()

        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:] # translation alignment

        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
        if cfg.update_bbox:
            height, width = img_shape
            bbox = process_bbox(data['tight_bbox'], width, height)
        # img_path = os.path.join(cfg.root_dir, "/".join(img_path.split("/")[-5:]))
        img_path = os.path.join(cfg.root_dir, img_path)
        # img
        if not os.path.exists(img_path):
            return None
        try:
            img = load_img(img_path)
        except:
            return None
        # img = load_img_from_lmdb(img_path, self.lmdb)
        # print("h36m has img")
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split,occ=False)
        img = self.transform(img.astype(np.float32))/255.
        # img_teacher = self.transform_basic(img.astype(np.float32))
        # img_student = addocc(img,bbox)
        # img_student = self.transform_hard(img_student.astype(np.uint8))
        
        if self.data_split == 'train':
            # h36m gt
            h36m_joint_img = data['joint_img']
            h36m_joint_cam = data['joint_cam']
            h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.h36m_root_joint_idx,None,:] # root-relative
            h36m_joint_valid = data['joint_valid']
            if do_flip:
                h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
                h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
                for pair in self.h36m_flip_pairs:
                    h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(), h36m_joint_img[pair[0],:].copy()
                    h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(), h36m_joint_cam[pair[0],:].copy()
                    h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] = h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()

            h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
            h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
            input_h36m_joint_img = h36m_joint_img.copy()
            h36m_joint_img[:,0] = h36m_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            h36m_joint_img[:,1] = h36m_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[self.h36m_root_joint_idx][2] # root-relative
            h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter

            # check truncation
            h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                        (h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                        (h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

            """
            print(f'{img_path} trunc:\n', h36m_joint_trunc.nonzero())
            tmp_coord = h36m_joint_img[:, :2] * np.array([[cfg.input_img_shape[1] / cfg.output_hm_shape[2], cfg.input_img_shape[0]/ cfg.output_hm_shape[1]]])
            newimg = vis_keypoints(img.numpy().transpose(1,2,0), tmp_coord)
            cv2.imshow(f'{img_path}', newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            """

            # transform h36m joints to target db joints
            h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
            h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
            h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
            h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)

            # apply PoseFix
            input_h36m_joint_img[:, 2] = 1  # joint valid
            tmp_joint_img = transform_joint_to_other_db(input_h36m_joint_img, self.h36m_joints_name, self.coco_joints_name)
            tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], data['near_joints'], data['num_overlap'], img2bb_trans)
            tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.h36m_joints_name)
            input_h36m_joint_img[self.h36m_coco_common_jidx, :2] = tmp_joint_img[self.h36m_coco_common_jidx, :2]
            """
            # debug PoseFix result
            newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), input_h36m_joint_img.T, self.h36m_skeleton)
            cv2.imshow(f'{img_path}', newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            import pdb; pdb.set_trace()
            """
            input_h36m_joint_img[:, 0] = input_h36m_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
            input_h36m_joint_img[:, 1] = input_h36m_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]
            # input_h36m_joint_img = transform_joint_to_other_db(input_h36m_joint_img, self.h36m_joints_name, self.joints_name)
            joint_mask = h36m_joint_trunc
            input_kp_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.joints_name, self.openpose_joints_name)
            input_kp_img = transform_joint_to_other_db(input_h36m_joint_img, self.h36m_joints_name, self.openpose_joints_name)
            input_skeleton_img = vis_keypoints_with_skeleton_fortrain(input_kp_img.T,
                                                                      self.openpose_skeleton,
                                                                      input_kp_trunc,
                                                                      (cfg.input_hm_shape[0],cfg.input_hm_shape[1],3),
                                                                      self.openpose_color)
            input_skeleton_img = self.transform(input_skeleton_img.astype(np.float32))/255.
            if smpl_param is not None:
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

                # if fitted mesh is too far from h36m gt, discard it
                # is_valid_fit = True
                # error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, do_flip)
                # if error > self.fitting_thr:
                #     is_valid_fit = False

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
            # h36m coordinate
            h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter
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

            # heavy_occluded = "A person"
            # if torch.sum(joint_mask)<10:
            #     heavy_occluded = "A haevy occluded person"
            # mask_prop = np.random.rand(joint_mask.shape[0],joint_mask.shape[1])  # B, N, 1
            # mask = mask_prop < 0.2 # 10% of joints are masked
            # joint_mask_student = joint_mask * (1 - mask.astype(np.float32))

            # images = {'img_teacher': img_teacher, 'img_student': img_student}
            # masks = {'mask_teacher': joint_mask, 'mask_student': joint_mask_student}
            # inputs = {'joints': input_h36m_joint_img[:, :2]}
            inputs = {'img': img, 
                    #   'joints_mask': joint_mask, 
                    #   'joints': input_h36m_joint_img[:, :2], 
                      'input_skeleton_img':input_skeleton_img}
            targets = {'orig_joint_img': h36m_joint_img, 'fit_joint_img': smpl_joint_img, 'orig_joint_cam': h36m_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
            meta_info = {'cliff_focal_length': cliff_focal_length,'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc, 
                         'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc, 'is_valid_fit': float(is_valid_fit), 'is_3D': float(True),
                        #  "heavy_occluded": heavy_occluded
                         }
            return inputs , targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_lixel': [], 'pa_mpjpe_lixel': [], 'mpjpe_param': [], 'pa_mpjpe_param': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # mesh from lixel
            # x,y: resize to input image space and perform bbox to image affine transform
            mesh_out_img = out['mesh_coord_img']
            mesh_out_img[:,0] = mesh_out_img[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_out_img[:,1] = mesh_out_img[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_out_img_xy1 = np.concatenate((mesh_out_img[:,:2], np.ones_like(mesh_out_img[:,:1])),1)
            mesh_out_img[:,:2] = np.dot(out['bb2img_trans'], mesh_out_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # z: devoxelize and translate to absolute depth
            root_joint_depth = annot['root_joint_depth']
            mesh_out_img[:,2] = (mesh_out_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size * 1000 / 2)
            mesh_out_img[:,2] = mesh_out_img[:,2] + root_joint_depth
            # camera back-projection
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mesh_out_cam = pixel2cam(mesh_out_img, focal, princpt)

            # h36m joint from gt mesh
            pose_coord_gt_h36m = annot['joint_cam'] 
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx,None] # root-relative 
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint,:] 
            
            # h36m joint from lixel mesh
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx,None] # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint,:]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
            eval_result['mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1)).mean())
            eval_result['pa_mpjpe_lixel'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1)).mean())
    
            vis = False
            # if vis:
            #     filename = annot['img_path'].split('/')[-1][:-4]

            #     img = load_img(annot['img_path'])[:,:,::-1]
            #     img = vis_mesh(img, mesh_out_img, 0.5)
            #     cv2.imwrite(filename + '.jpg', img)

            #     save_obj(mesh_out_cam, self.smpl.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE from lixel mesh: %.2f mm' % np.mean(eval_result['mpjpe_lixel']))
        print('PA MPJPE from lixel mesh: %.2f mm' % np.mean(eval_result['pa_mpjpe_lixel']))
        
        print('MPJPE from param mesh: %.2f mm' % np.mean(eval_result['mpjpe_param']))
        print('PA MPJPE from param mesh: %.2f mm' % np.mean(eval_result['pa_mpjpe_param']))