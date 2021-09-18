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

from datasets.latent import LatentDataset, ZipDataset
from datasets.recons import ReconstructionDataset
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMWrapper, SuperResModel, UNetModel
from models.vae import VAE
from util import (
    configure_device,
    save_as_images,
)


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


if __name__ == "__main__":
    cli()
