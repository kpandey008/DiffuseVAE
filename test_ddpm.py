import click
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.diffusion import UNetModel, DDPM, DDPMWrapper

from util import get_dataset, save_as_images, configure_device


@click.command()
@click.argument("chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
def sample(
    chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    save_path=os.getcwd(),
):
    # TODO: Update this method to work for cpus
    dev, _ = configure_device(device)

    # Model
    unet = UNetModel(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
    )
    online_network = DDPM(unet)
    target_network = copy.deepcopy(online_network)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    samples_list = []
    for idx in range(num_samples):
        with torch.no_grad():
            x_t = torch.randn(1, 3, image_size, image_size).to(dev)
            samples_list.append(ddpm_wrapper(x_t).cpu())

    cat_preds = torch.cat(samples_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    save_as_images(cat_preds)


if __name__ == "__main__":
    sample()
