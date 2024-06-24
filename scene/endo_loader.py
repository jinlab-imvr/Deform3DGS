import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.cameras import Camera
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import glob
from torchvision import transforms as T
import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F
from utils.general_utils import inpaint_depth, inpaint_rgb

def generate_se3_matrix(translation, rotation_rad):


    # Create rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
                   [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]])

    Ry = np.array([[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]])

    Rz = np.array([[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
                   [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create S(3) matrix
    se3_matrix = np.eye(4)


    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = translation

    return se3_matrix

def update_extr(c2w, rotation_deg, radii_mm):
        rotation_rad = np.radians(rotation_deg)
        translation = np.array([-radii_mm * np.sin(rotation_rad) , 0, radii_mm * (1 - np.cos(rotation_rad))])
        # translation = np.array([0, 0, 10])
        se3_matrix = generate_se3_matrix(translation, (0,rotation_rad,0)) # transform_C_C'
        extr = np.linalg.inv(se3_matrix) @ np.linalg.inv(c2w) # transform_C'_W = transform_C'_C @ (transform_W_C)^-1
        
        return np.linalg.inv(extr) # c2w
    
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    Zfar: float
    Znear: float

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)







class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        
        self.maxtime = 1.0
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        
        # coordinate transformation 
        if 'stereo_' in self.root_dir:
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            try:
                poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            except: 
                # No far and near
                poses = poses_arr.reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # StereoMIS has well calibrated intrinsics
            cy, cx, focal =  poses[0, :, -1]
            cy = 512//2
            cx = 640//2
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , cx],
                                        [0, focal, cy],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        else: 
            # load poses
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            H, W, focal = poses[0, :, -1]
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , W//2],
                                        [0, focal, H//2],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
        
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            
            # # ======================Generate the novel view for infer (StereoMIS)==========================
            # thetas = np.linspace(0, 30, poses.shape[0], endpoint=False)
            # c2w = update_extr(c2w, rotation_deg=thetas[idx], radii_mm=30)
            # # =================================================================================
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1] #w2c
            R = np.transpose(R) #c2w
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.image_paths = agg_fn("images")
        self.depth_paths = agg_fn("depth")
        self.masks_paths = agg_fn("masks")

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"
        
    def format_infos(self, split):
        cameras = []
        
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else:
            idxs = self.video_idxs
        
        for idx in tqdm(idxs):
            # mask / depth
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)
            # StereoMIS 
            if 'stereo_' in self.root_dir:
                mask = np.array(mask)
                if len(mask.shape) > 2:
                    mask = (mask[..., 0]>0).astype(np.uint8)
            else:
                mask = 1 - np.array(mask) / 255.0
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth) 
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                          image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"), time=time,
                          Znear=None, Zfar=None, K=self.K, h=self.img_wh[1], w=self.img_wh[0]))
        return cameras
    
    def filling_pts_colors(self, filling_mask, ref_depth, ref_image):
         # bool
        refined_depth = inpaint_depth(ref_depth, filling_mask)
        refined_rgb = inpaint_rgb(ref_image, filling_mask)
        return refined_rgb, refined_depth

    
    def get_sparse_pts(self, sample=True):
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth = np.percentile(depth[depth!=0], 0.1)
        inf_depth = np.percentile(depth[depth!=0], 99.9)
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0
        if 'stereo_' in self.root_dir:
            mask = np.array(Image.open(self.masks_paths[0]))
            if len(mask.shape) > 2:
                mask = (mask[..., 0]>0).astype(np.uint8) 
        else:
            mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0
        mask = np.logical_and(depth_mask, mask)   
        color = np.array(Image.open(self.image_paths[0]))/255.0
        # color_uint8 = np.array(Image.open(self.image_paths[0]), dtype=np.uint8)
        # filling_mask = np.logical_not(mask)
        # color, depth = self.filling_pts_colors(ref_image=color_uint8, ref_depth=depth, filling_mask=filling_mask)
        # color = color/255.0
        
        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)
        
        pts, colors = self.search_pts_colors_with_motion(pts, colors, mask, c2w)
        
        normals = np.zeros((pts.shape[0], 3))

        if sample:
            num_sample = int(0.1 * pts.shape[0])
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]
        
        return pts, colors, normals

    def calculate_motion_masks(self, ):
        images = []
        for j in range(0, len(self.image_poses)):
            color = np.array(Image.open(self.image_paths[j]))/255.0
            images.append(color)
        images = np.asarray(images).mean(axis=-1)
        diff_map = np.abs(images - images.mean(axis=0))
        diff_thrshold = np.percentile(diff_map[diff_map!=0], 95)
        return diff_map > diff_thrshold
        
    def search_pts_colors_with_motion(self, ref_pts, ref_color, ref_mask, ref_c2w):
        # calculating the motion mask
        motion_mask = self.calculate_motion_masks()
        interval = 1
        if len(self.image_poses) > 150: # in case long sequence
            interval = 2
        for j in range(1,  len(self.image_poses), interval):
            ref_mask_not = np.logical_not(ref_mask)
            ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])
            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0
            # mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0   
            if 'stereo_' in self.root_dir:
                mask = np.array(Image.open(self.masks_paths[j]))
                if len(mask.shape) > 2:
                    mask = (mask[..., 0]>0).astype(np.uint8)
                    
            else:
                mask = 1 - np.array(Image.open(self.masks_paths[j]))/255.0       
            depth_mask = np.ones(depth.shape).astype(np.float32)
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
            # Out of the visibility
            out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                    (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
            out_vis_pt_idx = np.where(out_vis_mask)[0]
            visible_mask = (1 - out_vis_mask)>0
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1
            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
            if compl_idxs.shape[0] <= 50:
                continue
            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]
            sel_idxs = np.random.choice(compl_pts.shape[0], int(0.1*compl_pts.shape[0]), replace=True)
            ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)

        
        if ref_pts.shape[0] > 600000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True)  
            ref_pts = ref_pts[sel_idxs]         
            ref_color = ref_color[sel_idxs] 
        return ref_pts, ref_color
    
         
    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = self.K[0,-1]
        cy = self.K[1,-1]
        X_Z = (i-cx) / self.focal[0]
        Y_Z = (j-cy) / self.focal[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
    
    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz
