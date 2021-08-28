import click
import copy
import math
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T

from pytorch_lightning.utilities.seed import seed_everything
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
        plt.savefig(save_path, dpi=100, pad_inches=0)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
def sample(
    chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    n_steps=1000,
    save_path=os.getcwd(),
):
    seed_everything(0)
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
        dropout=0,
        num_heads=1,
    )
    online_network = DDPM(unet)
    target_network = copy.deepcopy(online_network)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    batch_size = min(16, num_samples)

    ddpm_samples_list = []
    for idx in range(math.ceil(num_samples / batch_size)):
        with torch.no_grad():
            # Sample from DDPM
            x_t = torch.randn(batch_size, 3, image_size, image_size).to(dev)
            ddpm_sample = ddpm_wrapper(x_t, n_steps=n_steps).cpu()
            ddpm_samples_list.append(ddpm_sample)

    ddpm_cat_preds = torch.cat(ddpm_samples_list[:num_samples], dim=0)

    # Save the image and reconstructions as numpy arrays
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output"))


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--compare", default=True)
def sample_cond(
    vae_chkpt_path,
    ddpm_chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    z_dim=1024,
    n_steps=1000,
    save_path=os.getcwd(),
    compare=True,
):
    seed_everything(0)
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

    batch_size = min(16, num_samples)

    ddpm_samples_list = []
    vae_samples_list = []
    for idx in range(math.ceil(num_samples / batch_size)):
        with torch.no_grad():
            # Sample from VAE
            z = torch.randn(batch_size, z_dim, 1, 1, device=dev)
            recons_ = vae(z)
            vae_samples_list.append(recons_.cpu())

            # Sample from DDPM
            x_t = torch.randn(batch_size, 3, image_size, image_size).to(dev)
            recons = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(recons_)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons, n_steps=n_steps).cpu()
            ddpm_samples_list.append(ddpm_sample)

    ddpm_cat_preds = torch.cat(ddpm_samples_list[:num_samples], dim=0)
    vae_cat_preds = torch.cat(vae_samples_list[:num_samples], dim=0)

    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)

    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output"))

    # Save a comparison of all images
    if compare:
        for idx, (ddpm_pred, vae_pred) in enumerate(zip(ddpm_cat_preds, vae_cat_preds)):
            compare_samples(
                vae_pred,
                ddpm_pred * 0.5 + 0.5,
                save_path=os.path.join(save_path, "compare", f"compare_{idx}.png"),
            )


if __name__ == "__main__":
    cli()
