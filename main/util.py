import logging
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import models.backbone.resnet as backbone_models
from datasets import (
    CelebADataset,
    CelebAMaskHQDataset,
    AFHQDataset,
    ReconstructionDataset,
    CIFAR10Dataset,
    ReconstructionDatasetv2,
)

logger = logging.getLogger(__name__)


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = "-1"
        return f"cuda:{gpu_id}", gpu_id
    elif device == "tpu":
        if _xla_available:
            return xm.xla_device()
        raise Exception("Install PyTorch XLA to use TPUs")
    elif device == "cpu":
        return "cpu"
    else:
        raise NotImplementedError(f"The device type `{device}` is not supported")


# TODO: As more backbones are added register them with the modelstore
def get_resnet_models(backbone_name, pretrained=False, **kwargs):
    """Returns a backbone and its associated inplanes
    Args:
        backbone_name (str): Requested backbone
        pretrained (bool, optional): If the backbone must be pretrained on ImageNet. Defaults to False.
    Raises:
        NotImplementedError: If the backbone_name is not supported
    """
    backbone = None
    supported_backbones = ["resnet18", "resnet34", "resnet50", "resnet101"]
    if backbone_name not in supported_backbones:
        raise NotImplementedError(f"The backbone {backbone_name} is not supported")

    backbone = getattr(backbone_models, backbone_name)(pretrained=pretrained, **kwargs)
    return backbone


def get_dataset(name, root, image_size, norm=True, flip=False, **kwargs):
    assert isinstance(norm, bool)
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip() if flip else T.Lambda(lambda t: t),
        ]
    )
    if name == "celeba":
        dataset = CelebADataset(root, norm=norm, transform=transform, **kwargs)
    elif name == "celebamaskhq":
        dataset = CelebAMaskHQDataset(root, norm=norm, transform=transform, **kwargs)
    elif name == "afhq":
        dataset = AFHQDataset(root, norm=norm, transform=transform, **kwargs)
    elif name == "recons":
        dataset = ReconstructionDataset(root, norm=norm, transform=transform, **kwargs)
    elif name == "reconsv2":
        dataset = ReconstructionDatasetv2(
            root, norm=norm, transform=transform, **kwargs
        )
    elif name == "cifar10":
        assert image_size == 32
        transform = T.Compose(
            [
                T.RandomHorizontalFlip() if flip else T.Lambda(lambda t: t),
            ]
        )
        dataset = CIFAR10Dataset(root, transform=transform, norm=norm, **kwargs)
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset


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


def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(3):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (
            channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0]
        )
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj


def save_as_images(obj, file_name="output", denorm=True):
    # Saves predictions as png images
    if denorm:
        # obj = normalize(obj)
        obj = obj * 0.5 + 0.5
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        out = (out * 255).clip(0, 255).astype(np.uint8)
        img_out = Image.fromarray(out)
        current_file_name = file_name + "_%d.png" % i
        img_out.save(current_file_name, "png")


def save_as_np(obj, file_name="output", denorm=True):
    # Saves predictions directly as numpy arrays
    if denorm:
        # obj = normalize(obj)
        obj = obj * 0.5 + 0.5
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        current_file_name = file_name + "_%d.npy" % i
        np.save(current_file_name, out)


def compare_samples(samples, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    ncols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for idx, (caption, img) in enumerate(samples.items()):
        ax[idx].imshow(img.permute(1, 2, 0))
        ax[idx].set_title(caption)
        ax[idx].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)

    plt.close()
