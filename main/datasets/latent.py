import torch
from torch.utils.data import Dataset
from joblib import load


class LatentDataset(Dataset):
    def __init__(
        self,
        z_vae_size,
        z_ddpm_size,
        share_ddpm_latent=False,
        expde_model_path=None,
        **kwargs
    ):
        # NOTE: The batch index must be included in the latent code size input
        n_samples, *dims = z_ddpm_size

        self.z_vae = torch.randn(z_vae_size)
        self.share_ddpm_latent = share_ddpm_latent

        # Load the Ex-PDE model and sample z_vae from it instead!
        if expde_model_path is not None and expde_model_path != "":
            print("Found an Ex-PDE model. Will sample z_vae from it instead!")
            gmm = load(expde_model_path)
            gmm.set_params(random_state=kwargs.get("seed", 0))
            self.z_vae = (
                torch.from_numpy(gmm.sample(n_samples)[0]).view(z_vae_size).float()
            )
            assert self.z_vae.size() == z_vae_size

        if self.share_ddpm_latent:
            self.z_ddpm = torch.randn(dims)
        else:
            self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):
        if self.share_ddpm_latent:
            return self.z_ddpm, self.z_vae[idx]
        return self.z_ddpm[idx], self.z_vae[idx]

    def __len__(self):
        return int(self.z_vae.size(0))


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
