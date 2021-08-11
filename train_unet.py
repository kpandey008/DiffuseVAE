import click
import logging
import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from models.unet import UNet
from models.unet_big import UNet as BigUNet
from util import get_dataset, configure_device


logger = logging.getLogger(__name__)


@click.command()
@click.argument("root")
@click.option("--batch-size", default=64)
@click.option("--epochs", default=1000)
@click.option("--workers", default=4)
@click.option("--lr", default=3e-5, type=float)
@click.option("--log-step", default=1)
@click.option("--device", default="gpu:1")
@click.option("--chkpt-interval", default=1)
@click.option("--optimizer", default="Adam")
@click.option("--sample-interval", default=100)  # Integrate this!
@click.option("--restore-path", default=None)
@click.option("--results-dir", default=os.getcwd())
@click.option("--model", default="big")
def train(root, **kwargs):
    # Dataset
    dataset = get_dataset("recons", root)
    N = len(dataset)
    batch_size = kwargs.get("batch_size")
    batch_size = min(N, batch_size)

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=kwargs.get("workers"),
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    # Model
    lr = kwargs.get("lr")
    if kwargs.get("model") == "big":
        unet = BigUNet(lr=lr)
    else:
        unet = UNet(lr=lr)

    # Trainer
    train_kwargs = {}
    restore_path = kwargs.get("restore_path")
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = kwargs.get("results_dir")
    chkpt_callback = ModelCheckpoint(
        # monitor="Recons Loss",
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="unet-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=kwargs.get("chkpt_interval", 10),
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = kwargs.get("epochs")
    train_kwargs["log_every_n_steps"] = kwargs.get("log_step")
    train_kwargs["callbacks"] = [chkpt_callback]

    device = kwargs.get("device")
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(unet, train_dataloader=loader)


if __name__ == "__main__":
    train()
