import click
import copy
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T

from models.diffusion import UNetModel, DDPM, DDPMWrapper, SuperResModel
from models.vae import VAE

from util import save_as_images, configure_device


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Sample")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


@click.group()
def cli():
    pass


@cli.command()
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


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
def sample_cond(
    vae_chkpt_path,
    ddpm_chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    z_dim=1024,
    save_path=os.getcwd(),
):
    # TODO: Update this method to work for cpus
    dev, _ = configure_device(device)

    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Model
    unet = SuperResModel(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    ).to(dev)
    online_network = DDPM(unet).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    samples_list = []
    for idx in range(num_samples):
        with torch.no_grad():
            # Sample from VAE
            z = torch.randn(1, z_dim, 1, 1, device=dev)
            recons_ = vae(z)

            # Sample from DDPM
            x_t = torch.randn(1, 3, image_size, image_size).to(dev)
            recons = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(recons_)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons).cpu()
            samples_list.append(ddpm_sample)

            compare_samples(
                recons_.squeeze().cpu(),
                ddpm_sample.squeeze(),
                save_path=f"/content/compare_{idx}.png",
            )

    cat_preds = torch.cat(samples_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    save_as_images(cat_preds)


if __name__ == "__main__":
    cli()
