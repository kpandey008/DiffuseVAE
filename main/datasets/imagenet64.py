import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ImageNet64Dataset(Dataset):
    NUM_BATCHES = 10
    BATCH_SIZE = 128116
    NUM_CLASSES = 1000

    def __init__(
        self, root, norm=True, subsample_size=None, transform=None, return_label=True
    ):
        # The root directory must contain 10 data batch objects
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        assert len(os.listdir(root)) == self.NUM_BATCHES
        self.root = root
        self.transform = transform
        self.norm = norm
        self.return_label = return_label
        self.subsample_size = subsample_size

        self.data_objs = []
        self.labels = set()

        for b in tqdm(os.listdir(self.root)):
            with open(os.path.join(self.root, b), "rb") as fp:
                obj = pickle.load(fp)

            self.data_objs.append(obj)
            self.labels.update(obj["labels"])

        assert len(self.labels) == self.NUM_CLASSES

    def __getitem__(self, idx):
        # Select the batch from the idx
        batch_idx = idx // self.BATCH_SIZE
        truncated_idx = idx % self.BATCH_SIZE

        batch = self.data_objs[batch_idx]
        img = torch.from_numpy(batch["data"][truncated_idx].reshape(3, 64, 64))
        label = torch.tensor(batch["labels"][truncated_idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (img.float() / 127.5) - 1.0
        else:
            img = img.float() / 255.0

        if not self.return_label:
            return img
        return img, label

    def __len__(self):
        return (
            self.BATCH_SIZE * self.NUM_BATCHES
            if self.subsample_size is None
            else self.subsample_size
        )


if __name__ == "__main__":
    root = "/data1/kushagrap20/datasets/imagenet64/"
    dataset = ImageNet64Dataset(root, norm=False)
    img, label = dataset[128116 * 8 + 13]
    assert img.shape == (3, 64, 64)
    assert label.shape == ()
