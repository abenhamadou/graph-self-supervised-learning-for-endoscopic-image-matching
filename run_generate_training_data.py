import glob
import os
import logging
from _version import __version__
import hydra
from omegaconf import OmegaConf
from src.utils.image_keypoints_extractors import*
import cv2
import numpy as np
import json
import io

from src.utils.path import get_cwd


logger = logging.getLogger("Dataset-Generator")
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_generate_training_data")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")

    logger.info("Start Processing")
    frame_name=[]
    sequence_name=[]
    scores=[]
    point_coordinate=[]
    # process sequence folder one by one
    #-----------------------------------------------------------------------------------------
    input_image_lists = []
    for input_folder in cfg.paths.raw_data_dirs:
        input_image_lists.append(sorted(glob.glob(input_folder + "/*.png")))

    # process sequence folder one by one
    for input_image_list, folder_name in zip(input_image_lists, cfg.paths.raw_data_dirs):

        assert input_image_list

        # get only the folder name and create the output folder
        export_folder = cfg.paths.export_dir

        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        # go through the input image list
        for idx, image_path in enumerate(input_image_list):

            frame_name.append(os.path.basename(image_path))
            sequence_name.append(os.path.split(os.path.split(folder_name)[0])[1])
            # read and enhance current image frame
            image = cv2.imread(image_path, 3)
            enhanced_image = enhance_image(image)
            keypoints, _ = extract_image_keypoints(enhanced_image, cfg.params.keypoint_extractor)
            nb_keypoints = len(keypoints)
            # convert keypoints to numpy fp32
            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            # keep sparse keypoints by removing closest points under threshold in a greedy way
            to_be_removed_ids = []
            for i in range(nb_keypoints):
                for j in range(nb_keypoints):
                    if i != j and j not in to_be_removed_ids:
                        dist = distance.euclidean(np.squeeze(source_keypoints_coords[i]), np.squeeze(source_keypoints_coords[j]))
                        if dist < cfg.params.keypoint_dist_threshold:
                            to_be_removed_ids.append(j)

            keypoints = list(keypoints)

            for el_idx in sorted(to_be_removed_ids, reverse=True):
                del keypoints[el_idx]

            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            nb_keypoints = len(source_keypoints_coords)
            image_height, image_width = enhanced_image.shape

            z= cfg.params.patch_size
            pts_src = []
            key_points = []
            patch_counter = 1
            for b in range(0, nb_keypoints - 1):
                xa = int(source_keypoints_coords[b][0][0])
                ya = int(source_keypoints_coords[b][0][1])
                    # check if the patch is inside the image canvas
                if (
                        ((ya - z) > 0)
                        & ((xa - z) > 0)
                        & ((ya + z) < image_height)
                        & ((xa + z) < image_width)
                ):
                    # do crop patch from the  warped image
                    crop_img_a = enhanced_image[ya - z: ya + z, xa - z: xa + z]
                    pts_src.append((int(source_keypoints_coords[b][0][0]), int(source_keypoints_coords[b][0][1])))

                    key_points.append(keypoints[b])
                    # construct output filenames for patches
                    curr_output_folder = os.path.join(export_folder, f"patches/{os.path.split(os.path.split(folder_name)[0])[1]}/{os.path.splitext(os.path.basename(image_path))[0]}/")

                    if not os.path.exists(curr_output_folder):
                       os.makedirs(curr_output_folder)
                    patch_name= curr_output_folder + f"{str(patch_counter).zfill(3)}.png"
                    patch_counter += 1
                    cv2.imwrite(patch_name, crop_img_a)
            scores.append([kp.response for kp in key_points])
            point_coordinate.append(pts_src)

    # Define data

    train_data = {"sequence_name":sequence_name ,"frame_name": frame_name, "scores": scores, "key_points": point_coordinate}
    # Write JSON file
    json_export_file = os.path.join(export_folder, "data.json")
    with io.open(json_export_file , 'w', encoding='utf8') as outfile:
                 json.dump(train_data ,outfile,indent=3, sort_keys=True,separators=(',', ': '), ensure_ascii=False)


if __name__ == "__main__":
    main()