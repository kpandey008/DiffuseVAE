import torch
import torch.nn as nn
import models.backbone.resnet as backbone_models

from datasets import CelebADataset, CelebAMaskHQDataset


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


def get_dataset(name, root, **kwargs):
    if name == "celeba":
        dataset = CelebADataset(root, **kwargs)
    elif name == "celeba-hq":
        dataset = CelebAMaskHQDataset(root, **kwargs)
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset
