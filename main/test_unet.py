import click
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.unet import UNet
from util import get_dataset, save_as_images, configure_device


def compare_imgs(img, recons, pred, save_path=None, figsize=(10, 4)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(recons.permute(1, 2, 0))
    ax[1].set_title("VAE Reconstruction")
    ax[1].axis("off")

    ax[2].imshow(pred.permute(1, 2, 0))
    ax[2].set_title("Refinement")
    ax[2].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
    # plt.show()


# TODO: Check how to perform Multi-GPU inference
@click.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=16)
@click.option("--save-path", default=os.getcwd())
def reconstruct(
    chkpt_path,
    root,
    device="gpu:1",
    num_samples=-1,
    save_path=os.getcwd(),
):
    dev = configure_device(device)
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take value > 0")

    # Dataset
    dataset = get_dataset("recons", root)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    unet = UNet.load_from_checkpoint(chkpt_path).to(dev)
    unet.eval()

    recons_list = []
    img_list = []
    preds_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        recons, img = batch
        recons = recons.to(dev)
        with torch.no_grad():
            preds = unet(recons)

        if count + recons.size(0) >= num_samples:
            img_list.append(img[:num_samples, :, :, :].cpu())
            recons_list.append(recons[:num_samples, :, :, :].cpu())
            preds_list.append(preds[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        recons_list.append(recons.cpu())
        img_list.append(img.cpu())
        preds_list.append(preds.cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0)
    cat_recons = torch.cat(recons_list, dim=0)
    cat_preds = torch.cat(preds_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    for idx, (img, recons, pred) in enumerate(zip(cat_img, cat_recons, cat_preds)):
        compare_imgs(
            img, recons, pred, save_path=os.path.join(save_path, f"compare_{idx}.png")
        )


if __name__ == "__main__":
    reconstruct()
