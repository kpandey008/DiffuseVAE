import click
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.vae import VAE
from models.refiner.unet import UNet
from util import get_dataset, save_as_images, configure_device


@click.group()
def cli():
    pass


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("refine-chkpt-path")
@click.option("--n-steps", default=10)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
def interpolate(
    vae_chkpt_path,
    refine_chkpt_path,
    n_steps=10,
    z_dim=1024,
    save_path=os.getcwd(),
):
    vae = VAE.load_from_checkpoint(vae_chkpt_path)
    vae.eval()

    unet = UNet.load_from_checkpoint(refine_chkpt_path)
    unet.eval()

    # Sample z
    z_1 = torch.randn(1, z_dim, 1, 1)
    z_2 = torch.randn(1, z_dim, 1, 1)

    # interpolate
    lam = np.linspace(0, 1, num=n_steps)
    vae_interpolations = []
    combined_interpolations = []
    for l in lam:
        z_inter = z_1 * (l) + z_2 * (1 - l)

        # Forward pass through VAE
        with torch.no_grad():
            x_inter_vae = vae(z_inter)
            x_inter_combined = unet(x_inter_vae)
        vae_interpolations.append(x_inter_vae.squeeze())
        combined_interpolations.append(x_inter_combined.squeeze())

    os.makedirs(save_path, exist_ok=True)
    compare_interpolations(
        vae_interpolations,
        combined_interpolations,
        save_path=os.path.join(save_path, "combined_inter.png"),
    )


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("refine-chkpt-path")
@click.option("--num-samples", default=16)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
@click.option("--compare", default=True)
def sample_combined(
    vae_chkpt_path,
    refine_chkpt_path,
    num_samples=16,
    z_dim=1024,
    save_path=os.getcwd(),
    compare=True,
):
    vae = VAE.load_from_checkpoint(vae_chkpt_path)
    vae.eval()

    unet = UNet.load_from_checkpoint(refine_chkpt_path)
    unet.eval()

    # Sample z
    N = min(num_samples, 16)
    num_iters = num_samples // N  # For very large samples
    combined_sample_list = []
    vae_sample_list = []
    for _ in range(num_iters):

        z = torch.randn(num_samples, z_dim, 1, 1)
        with torch.no_grad():
            vae_recons = vae(z)
            recons = unet(vae_recons)
        vae_sample_list.append(vae_recons)
        combined_sample_list.append(recons)

    cat_vae_output = torch.cat(vae_sample_list, dim=0)
    cat_combined_output = torch.cat(combined_sample_list, dim=0)
    output_dir = os.path.splitext(save_path)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Save Samples
    save_as_images(cat_vae_output, os.path.join(output_dir, "vae"))
    save_as_images(cat_combined_output, os.path.join(output_dir, "combined"))

    if compare:
        save_dir = os.path.join(output_dir, "compare")
        os.makedirs(save_dir, exist_ok=True)
        # Save a comparison of all images
        for idx, (gen, refined) in enumerate(zip(cat_vae_output, cat_combined_output)):
            compare_samples(
                gen,
                refined,
                save_path=os.path.join(save_dir, f"compare_{idx}.png"),
            )


# TODO: Check how to perform Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--dataset", default="celebamaskhq")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def reconstruct(
    chkpt_path,
    root,
    device="gpu:1",
    dataset="celeba-hq",
    image_size=128,
    num_samples=-1,
    save_path=os.getcwd(),
    write_mode="image",
):
    dev, _ = configure_device(device)
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")

    # Dataset
    dataset = get_dataset(dataset, root, image_size, norm=False, flip=False)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward_recons(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :].cpu())
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch.cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0)
    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "orig"),
            denorm=False,
        )
        save_as_images(
            cat_img,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == "__main__":
    cli()
