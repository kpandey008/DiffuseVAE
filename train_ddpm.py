# Uses the openai Unet port
import copy
import logging
import os

import click
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMWrapper, SuperResModel, UNetModel
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@click.command()
@click.argument("root")
@click.option("--dim", default=64)
@click.option("--attn-resolutions", default="16,")
@click.option("--n-residual", default=1)
@click.option("--dim-mults", default="1,2,4,8")
@click.option("--dropout", default=0, type=float)
@click.option("--n-heads", default=1)
@click.option("--beta1", default=1e-4, type=float)
@click.option("--beta2", default=0.02, type=float)
@click.option("--n-timesteps", default=1000)
@click.option("--fp16", default=False, type=bool)
@click.option("--seed", default=0)
@click.option("--use-ema", default=True, type=bool)
@click.option("--ema-decay", default=0.9999, type=float)
@click.option("--batch-size", default=32)
@click.option("--epochs", default=1000)
@click.option("--log-step", default=1)
@click.option("--device", default="gpu:0")
@click.option("--chkpt-interval", default=1)
@click.option("--optimizer", default="Adam")
@click.option("--lr", default=2e-5, type=float)
@click.option("--restore-path", default=None)
@click.option("--results-dir", default=os.getcwd())
@click.option("--dataset", default="celeba-hq")
@click.option("--flip", default=False, type=bool)
@click.option("--image-size", default=128)
@click.option("--workers", default=4)
@click.option("--use-cond", default=True, type=bool)
@click.option("--loss", default="l1", type=click.Choice(["l1", "l2"]))
def train(root, **kwargs):
    # Set seed
    seed_everything(kwargs.get("seed"), workers=True)

    # Dataset and transforms
    d_type = kwargs.get("dataset")
    image_size = kwargs.get("image_size")
    dataset = get_dataset(d_type, root, image_size, flip=kwargs.get("flip"))
    N = len(dataset)
    batch_size = kwargs.get("batch_size")
    batch_size = min(N, batch_size)

    # Model
    lr = kwargs.get("lr")
    attn_resolutions = __parse_str(kwargs.get("attn_resolutions"))
    dim_mults = __parse_str(kwargs.get("dim_mults"))

    # Use the superres model for conditional training
    decoder_cls = UNetModel if not kwargs.get("use_cond") else SuperResModel
    decoder = decoder_cls(
        in_channels=3,
        model_channels=kwargs.get("dim"),
        out_channels=3,
        num_res_blocks=kwargs.get("n_residual"),
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=kwargs.get("dropout"),
        num_heads=kwargs.get("n_heads"),
    )

    ddpm = DDPM(
        decoder,
        beta_1=kwargs.get("beta1"),
        beta_2=kwargs.get("beta2"),
        T=kwargs.get("n_timesteps"),
    )
    ddpm_wrapper = DDPMWrapper(
        ddpm,
        copy.deepcopy(ddpm),
        lr=lr,
        loss=kwargs.get("loss"),
        conditional=kwargs.get("use_cond"),
    )

    # Trainer
    train_kwargs = {}
    restore_path = kwargs.get("restore_path")
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = kwargs.get("results_dir")
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="ddpmv2-{epoch:02d}-{loss:.4f}",
        every_n_epochs=kwargs.get("chkpt_interval", 1),
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = kwargs.get("epochs")
    train_kwargs["log_every_n_steps"] = kwargs.get("log_step")
    train_kwargs["callbacks"] = [chkpt_callback]

    if kwargs.get("use_ema"):
        ema_callback = EMAWeightUpdate(tau=kwargs.get("ema_decay"))
        train_kwargs["callbacks"].append(ema_callback)

    device = kwargs.get("device")
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=True)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if kwargs.get("fp16"):
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=kwargs.get("workers"),
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # # Gradient Clipping (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = 1.0

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    print(train_kwargs)
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloader=loader)


if __name__ == "__main__":
    train()
