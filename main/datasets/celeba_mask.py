import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# A very simplistic implementation of the CelebMaskHQ dataset supporting only images and no annotations
# TODO: Add functionality to download CelebA-MaskHQ and setup the dataset automatically
class CelebAMaskHQDataset(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = []

        img_path = os.path.join(self.root, "CelebA-HQ-img")
        for img in tqdm(os.listdir(img_path)):
            self.images.append(os.path.join(img_path, img))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root = "/data/kushagrap20/datasets/CelebAMask-HQ"
    dataset = CelebAMaskHQDataset(root, subsample_size=None)
    print(len(dataset))
