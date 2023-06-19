import numpy as np

import torch
import os
import cv2 as cv
from torch.utils.data import Dataset
from scipy.spatial import distance
from numpy import loadtxt
import glob
import os.path as osp


class PatchDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, root_path, patch_size,Descriptor ):
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.patch_size = patch_size
        self.image_name = []
        self.Descriptor=Descriptor

        self.image_name += [f for f in sorted(os.listdir(self.root_path  + "frames/"))]



    def __len__(self):
        return len(self.image_name )


    def enhance(self, img):
        crop_img = img[70 : int(img.shape[0]) - 70, 50 : int(img.shape[1]) - 40]
        gray2 = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        # image_enhanced = cv.equalizeHist(gray2)
        return image_enhanced

    def __getitem__(self, idx):

        frame_src_rgb = cv.imread(self.root_path +"frames/"+self.image_name[idx], 3)
        Next_frame_rgb = cv.imread(self.root_path + "frames/"+self.image_name[idx+1], 3)

        frame_src = self.enhance(frame_src_rgb )
        Next_frame = self.enhance(Next_frame_rgb )

        h, w = frame_src.shape
        # --------------------------Descriptor-------------------------------------------------------

        if self.Descriptor == 'SIFT':
            sift = cv.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(frame_src, None)
            kp2, des2 = sift.detectAndCompute(Next_frame, None)
        if self.Descriptor == 'SURF':
            surf = cv.xfeatures2d.SURF_create()
            kp1, des1 = surf.detectAndCompute(frame_src, None)
            kp2, des2 = surf.detectAndCompute(Next_frame, None)
        if self.Descriptor == 'AKAZE':
            akaze = cv.AKAZE_create(threshold=0.001)
            kp1, des01 = akaze.detectAndCompute(frame_src, None)
            kp2, des02 = akaze.detectAndCompute(Next_frame, None)
        if self.Descriptor == 'KAZE':
            kaze = cv.KAZE_create()
            kp1, des1 = kaze.detectAndCompute(frame_src, None)
            kp2, des2 = kaze.detectAndCompute(Next_frame, None)



        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        src_pts=[]
        dst_pts = []
        for keypoint in kp1:
            src_pts.append(keypoint.pt)
        list_keypoints_src = np.float32(src_pts).reshape(-1, 1, 2)

        patches_src = []
        patches_next_frame = []
        key_src=[]
        key_next_frame=[]
        score_src=[]
        score_next_frame=[]
        # ---------------------------------------------------Generate_data-----------------------------------------------------------

        for b in range(0, len(list_keypoints_src)):

            xa = int(list_keypoints_src[b][0][0])
            ya = int(list_keypoints_src[b][0][1])
            if (((ya - self.patch_size) > 0) & ((xa - self.patch_size) > 0) & ((ya + self.patch_size) < h) & ((xa + self.patch_size) < w)):
               crop_patches_src = frame_src[
                    ya - self.patch_size : ya + self.patch_size, xa - self.patch_size : xa + self.patch_size]

               patches_src.append(crop_patches_src)
               key_src.append(list_keypoints_src[b][0])
               score_src.append(kp1[b].response )
        # ---------------------------------------------------Generate_data-----------------------------------------------------------
        for keypoint in kp2:
            dst_pts.append(keypoint.pt)
        list_keypoints_dst = np.float32(dst_pts).reshape(-1, 1, 2)

        for b in range(0, len(list_keypoints_dst)):
                xp = int(list_keypoints_dst[b][0][0])
                yp = int(list_keypoints_dst[b][0][1])

                if (((yp - self.patch_size) > 0) & ((xp - self.patch_size) > 0) & ((yp + self.patch_size) < h) & (
                        (xp + self.patch_size) < w)):
                   crop_patches_next_frame = Next_frame[
                                              yp - self.patch_size: yp + self.patch_size,
                                              xp - self.patch_size: xp + self.patch_size]

                   patches_next_frame.append(crop_patches_next_frame)
                   key_next_frame.append(list_keypoints_dst[b][0])
                   score_next_frame.append(kp2[b].response)

        key_src= torch.tensor(np.array(key_src), dtype=torch.float32)
        key_next_frame = torch.tensor(np.array(key_next_frame), dtype=torch.float32)
        patches_src= torch.tensor(np.array(patches_src), dtype=torch.float32)
        patches_next_frame = torch.tensor(np.array(patches_next_frame), dtype=torch.float32)
        score_src=torch.tensor([np.array(score_src)], dtype=torch.float32)
        score_next_frame = torch.tensor([np.array(score_next_frame)], dtype=torch.float32)

        return {
            "image_src_path": self.root_path +"frames/"+self.image_name[idx],
            "image_dst_path": self.root_path +"frames/"+self.image_name[idx+1],
            "image_src": frame_src ,
            "image_dst": Next_frame ,
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            "keypoint_dst": key_next_frame,
            "keypoint_src": key_src,
            "score_src":score_src,
            "score_dst": score_next_frame,

        }