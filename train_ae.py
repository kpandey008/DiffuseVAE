import click
import logging
import os
import pytorch_lightning as pl
import torchvision.transforms as T

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from models.vae import VAE
from util import get_dataset


logger = logging.getLogger(__name__)


@click.command()
@click.argument("root")
@click.option("--code-size", default=1024, type=int)
@click.option("--batch-size", default=16)
@click.option("--epochs", default=1000)
@click.option("--image-size", default=128)
@click.option("--workers", default=2)
@click.option("--lr", default=1e-4)
@click.option("--log-step", default=1)
@click.option("--device", default="gpu", type=click.Choice(["cpu", "gpu", "tpu"]))
@click.option("--dataset", default="celeba-hq")
@click.option("--subsample-size", default=None)
@click.option("--chkpt-interval", default=1)
@click.option("--optimizer", default="Adam")
@click.option("--sample-interval", default=100)
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
    )

    # Model
    lr = kwargs.get("lr")
    enc_kwargs = {"code_size": kwargs.get("code_size")}
    vae = VAE(lr=lr, enc_kwargs=enc_kwargs)

    # Trainer
    train_kwargs = {}
    restore_path = kwargs.get("restore_path")
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = kwargs.get("results_dir")
    chkpt_callback = ModelCheckpoint(
        monitor="Total Loss",
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="vae-{epoch:02d}-{train_loss:.2f}",
        every_n_epochs=kwargs.get("chkpt_interval", 10),
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = kwargs.get("epochs")
    train_kwargs["log_every_n_steps"] = kwargs.get("log_step")
    train_kwargs["callbacks"] = [chkpt_callback]

    device = kwargs.get("device")
    if device == "gpu":
        train_kwargs["gpus"] = [1]
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloader=loader)


if __name__ == "__main__":
    train()
