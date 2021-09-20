import os

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, root, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.dataset = CIFAR10(self.root, train=True, download=True, **kwargs)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    root = "/data/kushagrap20/datasets/"
    dataset = CIFAR10Dataset(root)
    print(len(dataset))
