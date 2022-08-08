import lmdb
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO


class FFHQLmdbDataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        split="train",
        transform=None,
        image_size=256,
        original_resolution=256,
    ):
        self.transform = transform
        self.env = lmdb.open(
            root,
            readonly=True,
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.norm = norm
        self.original_resolution = original_resolution
        self.image_size = image_size

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        if split is None:
            self.offset = 0
        elif split == "train":
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == "test":
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        index = index + self.offset

        with self.env.begin(write=False) as txn:
            key = f"{self.original_resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return self.length


class FFHQDataset(Dataset):
    def __init__(
        self,
        root,
        norm=True,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = [
            os.path.join(root, img)
            for img in os.listdir(self.root)
            if img.endswith(".png")
        ]

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return len(self.images)
