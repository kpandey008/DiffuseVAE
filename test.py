import click
import numpy as np
import os
import torch
import torchvision.transforms as T

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.vae import VAE
from util import get_dataset, save_as_images, configure_device


@click.group()
def cli():
    pass


@cli.command()
@click.argument("chkpt-path")
@click.option("--num-samples", default=16)
@click.option("--z-dim", default=1024)
@click.option("--save-path", default=os.getcwd())
def sample(chkpt_path, num_samples=16, z_dim=1024, save_path=os.getcwd()):
    vae = VAE.load_from_checkpoint(chkpt_path)
    vae.eval()

    # Sample z
    N = min(num_samples, 16)
    num_iters = num_samples // N  # For very large samples
    sample_list = []
    for _ in range(num_iters):

        z = torch.randn(num_samples, z_dim, 1, 1)
        with torch.no_grad():
            recons = vae(z).squeeze()
        sample_list.append(recons)

    cat_output = torch.cat(sample_list, dim=0)
    output_dir = os.path.splitext(save_path)[0]
    os.makedirs(output_dir)
    save_as_images(cat_output, output_dir)


# TODO: Check how to perform Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--dataset", default="celeba-hq")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
def reconstruct(
    chkpt_path,
    root,
    device="gpu:1",
    dataset="celeba-hq",
    image_size=128,
    num_samples=-1,
    save_path=os.getcwd(),
):
    dev = configure_device(device)
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")

    # Transforms
    assert image_size in [128, 256, 512]
    transforms = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )

    # Dataset
    dataset = get_dataset(dataset, root, transform=transforms)

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path).to(dev)
    vae.eval()

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward_recons(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :])
            sample_list.append(recons[:num_samples, :, :, :])
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch.cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0).numpy()
    cat_sample = torch.cat(sample_list, dim=0).numpy()

    print(cat_img.shape)
    print(cat_sample.shape)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "images.npy"), cat_img)
    np.save(os.path.join(save_path, "recons.npy"), cat_sample)


if __name__ == "__main__":
    cli()
