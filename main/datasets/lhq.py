import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torchvision


class LHQ256Dataset(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.norm = norm
        self.transform = transform

        self.images = []

        for img in tqdm(sorted(os.listdir(self.root))):
            self.images.append(os.path.join(self.root, img))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root = "/data1/kushagrap20/datasets/lhq_256/lhq_256/"
    dataset = LHQ256Dataset(root, norm=False)
    print(len(dataset))
    img = dataset[0]
    torchvision.utils.save_image(img, "sample.png")
