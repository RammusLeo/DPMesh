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
# from utils.renderer import Renderer
from tqdm import tqdm
import lmdb
from common.utils.smpl import SMPL
from common.utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, load_img_from_lmdb
from common.utils.transforms import cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, denorm_joints, convert_crop_cam_to_orig_img
from common.utils.vis import save_obj, vis_keypoints_with_skeleton, vis_bbox, render_mesh,make_heatmap
import pickle
from common.utils.openpose_utils import OPENJOINTSNAME,OPENSKELETON,OPENCOLOR

### ONLY FOR TEST
def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord

class VideoMulti(torch.utils.data.Dataset):
    def __init__(self,transform,img_name = None):
        self.transform = transform
        self.img_name = img_name

        self.conf_thr = 0.05
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex
        self.openpose_joints_name = OPENJOINTSNAME
        self.openpose_skeleton = OPENSKELETON
        self.openpose_color = OPENCOLOR
        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Pelvis')
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))
        self.datalist = self.load_data()
        if self.datalist is not None:
            print("people len: ", len(self.datalist))

    
    def load_data(self):
        annots_path = osp.join(cfg.root_dir, 'demo',  cfg.renderimg, 'annotations', self.img_name+'.pkl') # you may get the pred 2D pose in PKL format
        with open(annots_path,'rb') as f:
            datalist = pickle.load(f)
        return datalist

    def make_2d_gaussian_heatmap(self, joint_coord_img, imgshape):
        x = np.arange(imgshape[1])
        y = np.arange(imgshape[0])
        yy, xx = np.meshgrid(y, x)
        xx = xx[ None, :, :].astype(np.float32)
        yy = yy[ None, :, :].astype(np.float32)

        x = joint_coord_img[:,  0, None, None]
        y = joint_coord_img[:,  1, None, None]
        heatmap = np.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        if self.datalist is None:
            return None
        annot = self.datalist[idx]
        img_path = osp.join(cfg.root_dir, 'demo',  cfg.renderimg, 'images', self.img_name+'.jpg')
        img = cv2.imread(img_path)
        height, weight, channel = img.shape
        
        img_shape = img.shape
        
        coco_joint_img = np.array(annot['keypoints']).reshape(17, 3)
        joint_coord_img = add_pelvis(coco_joint_img, self.coco_joints_name)
        joint_coord_img = add_neck(joint_coord_img, self.coco_joints_name)
        joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
        joint_valid = np.ones_like(joint_coord_img[:, :1], dtype=np.float32)
        
        joint_valid[joint_coord_img[:, 2] <= self.conf_thr] = 0
        if coco_joint_img.sum() == 0:
            bbox = np.array(annot['bbox']).astype(np.float32).reshape(-1) # xyxy
            
        else:
            bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
        bbox = process_bbox(bbox.copy(), weight, height, is_3dpw_test=True)

        img, img2bb_trans, bb2img_trans, _, _ = augmentation(img, bbox, 'test')
        img = self.transform(img.astype(np.float32))/255.


        joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
        joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

        joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.input_hm_shape[1]
        joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.input_hm_shape[0]
        joint_trunc = joint_valid * (
                    (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.input_hm_shape[1]) * \
                    (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.input_hm_shape[0])).reshape(-1, 1).astype(np.float32)

        cliff_focal_length = np.sqrt(img_shape[0]*img_shape[0]+img_shape[1]*img_shape[1])
        inputs = {'img': img, 
                  'joints': joint_coord_img, 
                  'joints_mask': joint_trunc}
        targets = {}
        meta_info = {'cliff_focal_length': cliff_focal_length,
                     'bb2img_trans': bb2img_trans, 
                     'img2bb_trans': img2bb_trans, 
                     'bbox': bbox, 
                     'img_path': img_path, }
                    #  "pid": annot['pid']}
       
        return inputs, targets, meta_info
    
    def evaluate(self, outs):
        # order = cfg.order   #change manually
        img_path = osp.join(cfg.root_dir, 'demo', cfg.renderimg, 'images', self.img_name+".jpg")
        rawimg = cv2.imread(img_path)
        orig_img = rawimg.copy()
        h,w,c = rawimg.shape
        random.seed(299)
        colorlist = []
        for i in range(100):
            colorlist.append((0.4+0.6*random.random(),0.6+0.4*random.random(),0.8+0.2*random.random(),1.0))
        '''
        red = (0.8,0.8,1.0,1.0)
        green = (0.8,1.0,0.8,1.0)
        blue = (1.0,0.8,0.8,1.0)        
        '''    
        for idx,out in tqdm(enumerate(outs[::-1]),total=len(outs)):
            mesh_cam_render = out['mesh_cam_render']
            bbox = out['bbox']
            # pid = out['pid']
            princpt = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
            try:
                rawimg = render_mesh(rawimg, mesh_cam_render, self.face, {'focal': cfg.focal, 'princpt': princpt},color=colorlist[idx])
            except:
                continue
        outputimg = np.concatenate((orig_img, rawimg), axis=1)[:-180,...]
        cv2.imwrite(os.path.join(cfg.root_dir, 'demo', cfg.renderimg, "renderimgs",self.img_name+".jpg"), outputimg)
        
