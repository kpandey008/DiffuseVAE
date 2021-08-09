import click
import logging
import os
import pytorch_lightning as pl
import torchvision.transforms as T

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from models.diffusion.ddpm import DDPM
from models.diffusion.unet import UNet
from util import configure_device, get_dataset


logger = logging.getLogger(__name__)


@click.command()
@click.argument("root")
@click.option("--t-embed", default=64)
@click.option("--beta1", default=1e-4, type=float)
@click.option("--beta2", default=0.02, type=float)
@click.option("--n-steps", default=1000)
@click.option("--batch-size", default=32)
@click.option("--epochs", default=1000)
@click.option("--image-size", default=128)
@click.option("--workers", default=4)
@click.option("--lr", default=2e-5)
@click.option("--log-step", default=1)
@click.option("--device", default="gpu:0")
@click.option("--dataset", default="celeba-hq")
@click.option("--subsample-size", default=None)  # Integrate this!
@click.option("--chkpt-interval", default=1)
@click.option("--optimizer", default="Adam")
@click.option("--sample-interval", default=100)  # Integrate this!
@click.option("--restore-path", default=None)
@click.option("--results-dir", default=os.getcwd())
def train(root, **kwargs):
    # Transforms
    image_size = kwargs.get("image_size")
    assert image_size in [128, 256, 512]
    transforms = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Dataset
    d_type = kwargs.get("dataset")
    dataset = get_dataset(d_type, root, transform=transforms)
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
        persistent_workers=True,
    )

    # Model
    lr = kwargs.get("lr")
    decoder = UNet(t_embed_dim=kwargs.get("t_embed"))
    ddpm = DDPM(
        decoder,
        beta_1=kwargs.get("beta1"),
        beta_2=kwargs.get("beta2"),
        T=kwargs.get("n_steps"),
        lr=lr,
    )

    # Trainer
    train_kwargs = {}
    restore_path = kwargs.get("restore_path")
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = kwargs.get("results_dir")
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="ddpm-{epoch:02d}-{loss:.2f}",
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
    trainer.fit(ddpm, train_dataloader=loader)


if __name__ == "__main__":
    train()
