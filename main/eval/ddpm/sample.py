# Helper script to sample from an unconditional DDPM model
# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import copy

import hydra
import pytorch_lightning as pl
from datasets.latent import UncondLatentDataset
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMWrapper, UNetModel
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path=os.path.join(p, "configs"))
def sample(config):
    # Seed and setup
    config = config.dataset.ddpm
    seed_everything(config.evaluation.seed, workers=True)

    # Ensure unconditional DDPM mode
    assert config.evaluation.type == "uncond"

    batch_size = config.evaluation.batch_size
    n_steps = config.evaluation.n_steps
    n_samples = config.evaluation.n_samples
    image_size = config.data.image_size

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    decoder = UNetModel(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        num_heads=config.model.n_heads,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    online_ddpm = DDPM(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )
    target_ddpm = DDPM(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=None,
        conditional=False,
        pred_steps=n_steps,
        eval_mode="sample",
        resample_strategy=config.evaluation.resample_strategy,
        skip_strategy=config.evaluation.skip_strategy,
        sample_method=config.evaluation.sample_method,
        sample_from=config.evaluation.sample_from,
        data_norm=config.data.norm,
        strict=False,
    )

    # Create predict dataset of latents
    z_dataset = UncondLatentDataset(
        (n_samples, 3, image_size, image_size),
    )

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config.evaluation.device
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
        num_workers=config.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    write_callback = ImageWriter(
        config.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=False,
        sample_prefix=config.evaluation.sample_prefix,
        save_mode=config.evaluation.save_mode,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


if __name__ == "__main__":
    sample()
