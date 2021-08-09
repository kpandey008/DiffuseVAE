import click
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.diffusion.ddpm import DDPM
from models.diffusion.unet import UNet

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
    dev, _ = configure_device(device)

    # Model
    decoder = UNet(64).to(dev)
    ddpm = DDPM.load_from_checkpoint(chkpt_path, decoder=decoder).to(dev)
    ddpm.eval()

    samples_list = []
    for idx in range(num_samples):
        with torch.no_grad():
            x_t = torch.randn(1, 3, image_size, image_size).to(dev)
            samples_list.append(ddpm(x_t).cpu())

    cat_preds = torch.cat(samples_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    save_as_images(cat_preds)


if __name__ == "__main__":
    sample()
