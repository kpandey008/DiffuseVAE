import numpy as np
import os
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# A very simplistic implementation of the CelebA dataset supporting only images and no annotations
# TODO: Add functionality to download CelebA and setup the dataset automatically
class CelebADataset(Dataset):
    def __init__(
        self,
        root,
        subsample_size=None,
        transform=None,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform

        self.images = []

        for img in tqdm(os.listdir(root)):
            self.images.append(os.path.join(self.root, img))

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

        return img

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root = "/data/kushagrap20/datasets/img_align_celeba"
    dataset = CelebADataset(root, subsample_size=10000)
    print(len(dataset))
