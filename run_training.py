import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf
import warnings
import statistics
import torch
from src.model.model import Attentinal_GNN
from src.datasets.train_patch_loader import PatchDataset
from src.loss.contrastive_loss_layers import ContrastiveLoss
from src.utils.path import get_cwd

logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="config", config_name="config_train")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")
    data_dir_list = cfg.paths.train_data
    nb_epoch = cfg.params.nb_epoch
    batch_size = cfg.params.batch_size
    image_size = cfg.params.image_size
    initial_lr = cfg.params.lr
    minibatch_size= cfg.params.minibatch_size
    temperature_parameter= cfg.params.temperature_parameter
    # gathers all epoch losses
    loss_list = []

    # creates dataset and datalaoder
    dataset = PatchDataset(data_dir_list, image_size, cfg)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = Attentinal_GNN()

    criterion = ContrastiveLoss( minibatch_size=minibatch_size, temperature_parameter=temperature_parameter).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

    logger.info("Start Epochs ...")

    for epoch in range(nb_epoch):
        loss_epoch = []
        for (idx, data) in enumerate(train_loader):
            optimizer.zero_grad()
            z1 = model(data["patch_src"],data["keypoint_src"], data["score_keypoints"],data["image_src"]).to(device)
            z2 = model(data["patch_dst"],data["keypoint_dst"],data["score_keypoints"],data["image_dst"]).to(device)
            z1 = z1.reshape(z1.shape[1], z1.shape[2]).transpose(0, 1)
            z2 = z2.reshape(z2.shape[1], z2.shape[2]).transpose(0, 1)
            #z1= torch.Tensor.cpu(z1).detach()
            #z2 = torch.Tensor.cpu(z2).detach()
            loss = criterion.loss(z1, z2, batch_size=minibatch_size)
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_epoch.append(loss.item())

        logger.info(f"Epoch= {epoch:04d} Loss= {statistics.mean(loss_epoch):0.4f}")

    checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss_epoch}
    checkpoint_export_path = os.path.join(output_dir, f"{cfg.params.model}.pth")
    torch.save(checkpoint, checkpoint_export_path)
    logger.info(f"Checkpoint savec to: {checkpoint_export_path}")



if __name__ == "__main__":
    main()
