import numpy as np
import os

from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(self, root, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform

        self.images = np.load(os.path.join(self.root, "images.npy"))
        self.recons = np.load(os.path.join(self.root, "recons.npy"))

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            # To enable deterministic samples set a random seed at
            # a global level
            inds = np.random.choice(np.arange(self.images.shape[0]))
            self.images = self.images[:inds]
            self.recons = self.recons[:inds]

        assert self.images.shape[0] == self.recons.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        recons = self.recons[idx]
        if self.transform is not None:
            img = self.transform(img)
            recons = self.transform(recons)

        return recons, img

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root = "/data/kushagrap20/vaedm/reconstructions"
    dataset = ReconstructionDataset(root)
    print(len(dataset))
