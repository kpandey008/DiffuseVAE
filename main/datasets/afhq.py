import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class AFHQv2Dataset(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        # We only train on the AFHQ train set
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = []

        subfolder_list = ["dog"]
        base_path = os.path.join(self.root, "train")
        for subfolder in subfolder_list:
            sub_path = os.path.join(base_path, subfolder)

            for img in tqdm(os.listdir(sub_path)):
                self.images.append(os.path.join(sub_path, img))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            # To enable deterministic samples set a random seed at
            # a global level
            self.images = np.random.choice(
                self.images, size=subsample_size, replace=False
            )

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
    root = "/data1/kushagrap20/datasets/afhq_v2/"
    dataset = AFHQv2Dataset(root)
    print(len(dataset))
