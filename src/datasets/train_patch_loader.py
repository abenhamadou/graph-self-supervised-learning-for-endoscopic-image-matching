import torch
import json
from torch.utils.data import Dataset
import os.path as osp
import glob

from src.utils.image_keypoints_extractors import*
import random

class PatchDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, root_path, patch_size,cfg):
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.cfg=cfg
        self.patch_size = patch_size
        self.all_data_frames = self.get_file_list()


    def get_file_list(self):
        with open(osp.join(self.root_path,"data.json"), "r") as f:
            self.data = json.load(f)
        self.images_list= self.data["frame_name"]
        return self.images_list



    def __len__(self):
        return len(self.all_data_frames)


    def __getitem__(self, idx):
        #get frame path
        sequence_name=self.data["sequence_name"][idx]
        frame_path=osp.join(self.root_path,f"{sequence_name}/frames/{self.all_data_frames[idx]}")
        image = cv2.imread(frame_path, 3)
        enhanced_image = enhance_image(image)
        # load patches
        patch_list=sorted(glob.glob(osp.join(self.root_path,f"patches/{sequence_name}/{osp.splitext(self.all_data_frames[idx])[0]}"+ "/*")))
        patch_src=[cv2.imread(patch,cv2.IMREAD_GRAYSCALE) for patch in patch_list]
        # load data
        source_keypoints_coords = np.float32(self.data["key_points"][idx]).reshape(-1, 1, 2)
        scores1 = (self.data["scores"][idx]).copy()
        nb_keypoints = len(source_keypoints_coords)
        image_height, image_width = enhanced_image.shape
        (center_x, center_y) = (image_width // 2, image_height // 2)

        # select random transformation between predefined transformation list
        transformation = random.choice(self.cfg.params.transformation_list)
        z = self.cfg.params.patch_size
        # ---------------------------------------------------Data-preparation-----------------------------------------------------------
        if transformation == "rotation":
            rotation_angle = random.choice(self.cfg.params.predefined_angle_degrees)
            transformation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            warped_image = cv2.warpAffine(enhanced_image, transformation_matrix, (image_width, image_height))
        elif transformation == 'translation':
            transformation_matrix = np.float32([[1, 0, random.choice(self.cfg.params.predefined_translation_pixels)], [0, 1, random.choice(self.cfg.params.predefined_translation_pixels)]])
            warped_image = cv2.warpAffine(enhanced_image, transformation_matrix, (image_width, image_height))
        elif transformation == 'zoom':
            zf = random.choice(self.cfg.params.predefined_zoom_factors)
            warped_image = clipped_zoom(enhanced_image, zf)
        patch_dst=[]
        pts_src= []
        pts_dst=[]
        to_be_removed_ids=[]
        for b in range(0, nb_keypoints):
            if transformation in ["rotation",'translation']:
                rotated_point = transformation_matrix.dot(
                    np.array((int(source_keypoints_coords[b][0][0]), int(source_keypoints_coords[b][0][1])) + (1,)))
                xp = int(rotated_point[0])
                yp = int(rotated_point[1])
            elif transformation == 'zoom':
                xp, yp = zoom_coordinates(enhanced_image, source_keypoints_coords[b][0][0], source_keypoints_coords[b][0][1], zf)
                # check if the patch is inside the image canvas
            if (
                ((yp - z) > 0)
                & ((xp - z) > 0)
                & ((yp + z) < image_height)
                & ((xp + z) < image_width)
                ):
                    # do crop patch from the  warped image
                    crop_img_p = warped_image[yp - z: yp + z, xp - z: xp + z]
                    patch_dst.append(crop_img_p)
                    pts_src.append((source_keypoints_coords[b][0][0],source_keypoints_coords[b][0][1]))
                    pts_dst.append((xp,yp))
            else:
                to_be_removed_ids.append(b)
        for el_idx in sorted(to_be_removed_ids, reverse=True):
            del patch_src[el_idx]
            del scores1[el_idx]

        pts_src = torch.tensor(np.array(pts_src), dtype=torch.float32)
        pts_dst = torch.tensor(np.array(pts_dst), dtype=torch.float32)
        patches_src= torch.tensor(np.array(patch_src), dtype=torch.float)
        patches_dst = torch.tensor(np.array(patch_dst), dtype=torch.float)
        score = torch.tensor([np.array(scores1)], dtype=torch.float32)

        return {
            "image_src": enhanced_image,
            "image_dst": warped_image,
            "patch_src": patches_src,
            "patch_dst": patches_dst,
            "keypoint_dst": pts_dst,
            "keypoint_src": pts_src,
            "score_keypoints": score
        }
