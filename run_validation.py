import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
import tqdm
import torch
from src.model.model import Attentinal_GNN

logger = logging.getLogger("Validation")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import statistics
from src.utils.matcher import feature_match, feature_extraction, evaluate_matches
from src.datasets.test_patch_loader import PatchDataset


@hydra.main(version_base=None, config_path="config", config_name="config_validation")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    root_folder = os.path.abspath(os.path.split(__file__)[0] + "/")
    logger.info("Working dir: " + root_folder)
    logger.info("Loading parameters from config file")
    validation_data_root = cfg.paths.validation_data
    model_name = cfg.params.model_name
    model_weights_path = cfg.params.weights_path
    patch_size = cfg.params.patch_size
    distance_matching_threshold = cfg.params.distance_matching_threshold
    matching_threshold = cfg.params.matching_threshold
    batch_size = cfg.params.batch_size

    # load the model to be evaluated
    model = Attentinal_GNN()
    model.load_state_dict(torch.load(model_weights_path)['state_dict'])


    # generate testing data
    test_dataset = PatchDataset(validation_data_root, patch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset , batch_size=batch_size, shuffle=False, pin_memory=True)
    # back metrics
    precision = []
    matching_score = []

    # go through the patches, frame by frame
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        image_src=data["image_src"]
        image_dst=data["image_dst"]
        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        gt_keypoint_src = data["keypoint_src"]
        gt_keypoint_dst = data["keypoint_dst"]
        key_point_score_src=data["score_src"]
        key_point_score_dst = data["score_dst"]

        # extract feature vector for all patches
        list_desc_src = feature_extraction(patch_src, gt_keypoint_src,key_point_score_src,model, image_src)
        list_desc_dst = feature_extraction(patch_dst,gt_keypoint_dst,key_point_score_dst, model, image_dst)

        # do matching
        matches, distance = feature_match(list_desc_src, list_desc_dst, matching_threshold)

        # compute evaluation metrics
        nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
            gt_keypoint_src, gt_keypoint_dst, matches, distance_matching_threshold,distance,matching_threshold
        )
        precision.append(nb_true_matches / (nb_false_matching + nb_true_matches))
        matching_score.append(nb_true_matches / (nb_false_matching + nb_true_matches + nb_rejected_matches))

    logger.info(f"Precision= {statistics.mean(precision):0.4f}")
    logger.info(f"Matching_score= {statistics.mean(matching_score):0.4f}")


if __name__ == "__main__":
    main()
