import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CMHQLmdbDataset(Dataset):
    def __init__(self, root, norm=True, transform=None, original_resolution=1024):
        self.transform = transform
        self.original_resolution = original_resolution
        self.env = lmdb.open(
            root,
            readonly=True,
            max_readers=32,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.norm = norm

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode("utf-8")
            img_bytes = txn.get(key)

        img = Image.frombytes(
            "RGB", (self.original_resolution, self.original_resolution), img_bytes
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __len__(self):
        return self.length
