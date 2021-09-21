import copy
import math
import os

import click
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets.latent import LatentDataset, ZipDataset
from datasets.recons import ReconstructionDataset
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMWrapper, SuperResModel, UNetModel
from models.vae import VAE
from util import configure_device, normalize, save_as_images


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@click.group()
def cli():
    pass


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
@click.option("--num-samples", default=1)
@click.option("--n-steps", default=1000)
@click.option("--n-workers", default=8)
@click.option("--image-size", default=128)
@click.option("--batch-size", default=8)
@click.option("--compare", default=True, type=bool)
@click.option("--temp", default=1.0, type=float)
@click.option("--seed", default=0)
def generate_recons(vae_chkpt_path, ddpm_chkpt_path, root, **kwargs):
    seed_everything(kwargs.get("seed"))

    transforms = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    batch_size = kwargs.get("batch_size")
    n_steps = kwargs.get("n_steps")
    n_samples = kwargs.get("num_samples")
    image_size = kwargs.get("image_size")

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(vae_chkpt_path)
    vae.eval()

    # Load pretrained wrapper
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
    )
    online_network = DDPM(unet)
    online_network.eval()
    target_network = copy.deepcopy(online_network)
    target_network.eval()

    # NOTE: Using strict=False since the VAE model is not included
    # in the pretrained DDPM state_dict
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
        vae=vae,
        strict=False,
        pred_steps=n_steps,
        eval_mode="recons",
    )

    # Create predict dataset of reconstructions
    recons_dataset = ReconstructionDataset(
        root, transform=transforms, subsample_size=n_samples
    )
    ddpm_latent_dataset = TensorDataset(
        torch.randn(n_samples, 3, image_size, image_size)
    )
    pred_dataset = ZipDataset(recons_dataset, ddpm_latent_dataset)

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = kwargs.get("device")
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=kwargs.get("n_workers"),
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        kwargs.get("save_path"),
        "batch",
        compare=kwargs.get("compare"),
        n_steps=n_steps,
        eval_mode="recons",
    )
    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = kwargs.get("save_path")
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


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
@click.option("--n-workers", default=8)
@click.option("--batch-size", default=8)
@click.option("--temp", default=1.0, type=float)
@click.option("--seed", default=0)
@click.option("--checkpoints", default="")
def sample_cond(vae_chkpt_path, ddpm_chkpt_path, **kwargs):
    seed_everything(kwargs.get("seed"))

    batch_size = kwargs.get("batch_size")
    z_dim = kwargs.get("z_dim")
    n_steps = kwargs.get("n_steps")
    image_size = kwargs.get("image_size")
    n_samples = kwargs.get("num_samples")
    checkpoints = __parse_str(kwargs.get("checkpoints"))

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(vae_chkpt_path)
    vae.eval()

    # Load pretrained wrapper
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
    )
    online_network = DDPM(unet)
    online_network.eval()
    target_network = copy.deepcopy(online_network)
    target_network.eval()

    # NOTE: Using strict=False since the VAE model is not included
    # in the pretrained DDPM state_dict
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
        vae=vae,
        strict=False,
        pred_steps=n_steps,
        eval_mode="sample",
        pred_checkpoints=checkpoints,
    )

    # Create predict dataset of latents
    z_dataset = LatentDataset(
        (n_samples, z_dim, 1, 1),
        (n_samples, 3, image_size, image_size),
        n_steps=n_steps,
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = kwargs.get("device")
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=kwargs.get("n_workers"),
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        kwargs.get("save_path"),
        "batch",
        compare=kwargs.get("compare"),
        n_steps=n_steps,
        eval_mode="sample",
    )
    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = kwargs.get("save_path")
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


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


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.squeeze().permute(1, 2, 0))
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
        ax[0, i].imshow(inter_1.squeeze().permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.squeeze().permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--n-interpolate", default=10)
@click.option("--use-concat", default=True, type=bool)
@click.option("--temp", default=1.0, type=float)
@click.option("--seed", default=0)
def interpolate_vae(vae_chkpt_path, ddpm_chkpt_path, **kwargs):
    seed_everything(kwargs.get("seed"))
    dev, _ = configure_device(kwargs.get("device"))
    image_size = kwargs.get("image_size")
    z_dim = kwargs.get("z_dim")
    n_steps = kwargs.get("n_steps")

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=kwargs.get("n_interpolate"), device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Superres Model
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
    unet.eval()

    online_network = DDPM(
        unet,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    target_network.eval()
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []

    with torch.no_grad():
        # Interpolate in the VAE latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev)
        z_2 = torch.randn(1, z_dim, 1, 1, device=dev)

        for idx, l in tqdm(enumerate(lam)):
            # Sample from VAE
            z_inter = z_1 * l + z_2 * (1 - l)
            recons_inter = vae(z_inter)

            vae_samples_list.append(recons_inter.cpu())

            # Sample from DDPM
            recons = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(recons_inter)
            x_t = (kwargs.get("temp") * torch.randn_like(recons)).to(dev)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons, n_steps=n_steps)[
                str(n_steps)
            ].cpu()
            ddpm_samples_list.append(normalize(ddpm_sample))

    cat_ddpm_samples = torch.cat(ddpm_samples_list, dim=0)
    cat_vae_samples = torch.cat(vae_samples_list, dim=0)

    # Save DDPM and VAE samples
    save_path = kwargs.get("save_path")
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    save_as_images(cat_ddpm_samples, file_name=os.path.join(save_path, "inter_ddpm"))
    save_as_images(cat_vae_samples, file_name=os.path.join(save_path, "inter_vae"))

    # Compare
    save_path = kwargs.get("save_path")
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    compare_interpolations(
        ddpm_samples_list,
        vae_samples_list,
        save_path=os.path.join(save_path, "inter_compare.png"),
    )


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--truncation", default=1.0, type=float)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--n-interpolate", default=10)
@click.option("--reuse-epsilon", default=False, type=bool)
@click.option("--temp", default=1.0, type=float)
@click.option("--seed", default=0)
def interpolate_ddpm(vae_chkpt_path, ddpm_chkpt_path, **kwargs):
    seed_everything(kwargs.get("seed"))
    dev, _ = configure_device(kwargs.get("device"))
    image_size = kwargs.get("image_size")
    z_dim = kwargs.get("z_dim")
    n_steps = kwargs.get("n_steps")

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=kwargs.get("n_interpolate"), device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Superres Model
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
    unet.eval()

    online_network = DDPM(
        unet,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    target_network.eval()
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []

    with torch.no_grad():
        # Interpolate in the DDPM latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev)
        recons_inter = vae(z_1)
        recons = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(recons_inter)

        x_t1 = kwargs.get("temp") * torch.randn(
            1, 3, image_size, image_size, device=dev
        )
        x_t2 = kwargs.get("temp") * torch.randn(
            1, 3, image_size, image_size, device=dev
        )

        for idx, l in tqdm(enumerate(lam)):
            # Sample from DDPM
            x_t_inter = x_t1 * l + x_t2 * (1 - l)
            ddpm_sample = ddpm_wrapper(x_t_inter, cond=recons, n_steps=n_steps)[
                str(n_steps)
            ].cpu()
            ddpm_samples_list.append(normalize(ddpm_sample))
            vae_samples_list.append(recons.cpu())

    cat_ddpm_samples = torch.cat(ddpm_samples_list, dim=0)
    cat_vae_samples = torch.cat(vae_samples_list, dim=0)

    # Save DDPM and VAE samples
    save_path = kwargs.get("save_path")
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    save_as_images(cat_ddpm_samples, file_name=os.path.join(save_path, "inter_ddpm"))
    save_as_images(cat_vae_samples, file_name=os.path.join(save_path, "inter_vae"))

    # Compare
    save_path = kwargs.get("save_path")
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    plot_interpolations(
        ddpm_samples_list, save_path=os.path.join(save_path, "inter_plot.png")
    )


if __name__ == "__main__":
    cli()
