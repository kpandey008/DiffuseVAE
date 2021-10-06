import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, root, norm=True, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.norm = norm
        self.dataset = CIFAR10(self.root, train=True, download=True, **kwargs)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        return torch.tensor(img).float()

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    root = "/data/kushagrap20/datasets/"
    dataset = CIFAR10Dataset(root)
    print(dataset[0].shape)
