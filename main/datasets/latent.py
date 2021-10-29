import torch

from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, z_vae_size, z_ddpm_size, **kwargs):
        # NOTE: The batch index must be included in the latent code size input
        self.z_vae = torch.randn(z_vae_size)
        self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):
        return self.z_ddpm[idx], self.z_vae[idx]

    def __len__(self):
        return int(self.z_ddpm.size(0))


class UncondLatentDataset(Dataset):
    def __init__(self, z_ddpm_size, **kwargs):
        # NOTE: The batch index must be included in the latent code size input
        self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):
        return self.z_ddpm[idx]

    def __len__(self):
        return int(self.z_ddpm.size(0))


class ZipDataset(Dataset):
    def __init__(self, recons_dataset, latent_dataset, **kwargs):
        # NOTE: The batch index must be included in the latent code size input
        assert len(recons_dataset) == len(latent_dataset)
        self.recons_dataset = recons_dataset
        self.latent_dataset = latent_dataset

    def __getitem__(self, idx):
        return self.recons_dataset[idx], self.latent_dataset[idx]

    def __len__(self):
        return len(self.recons_dataset)
