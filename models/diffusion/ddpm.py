import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .unet import UNet

from tqdm import tqdm


class DDPM(pl.LightningModule):
    def __init__(self, decoder, beta_1=1e-4, beta_2=0.02, T=1000, lr=1e-4):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.criterion = nn.L1Loss()
        self.lr = lr

        # Precompute alpha_t for all t values
        betas = np.linspace(self.beta_1, self.beta_2, self.T)
        alphas = 1 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar_shifted = np.append(1.0, alpha_bar[:-1])

        sigma_t = betas * (1.0 - alpha_bar_shifted) / (1.0 - alpha_bar)

        self.betas = torch.from_numpy(betas)
        self.alphas = torch.from_numpy(alphas)
        self.alpha_bar = torch.from_numpy(alpha_bar)
        self.sigma_t = torch.from_numpy(sigma_t)

        self.const_dev_set = False

    def compute_noisy_input(self, x, eps, t):
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        return (
            x * torch.sqrt(self.alpha_bar[t])[:, None, None, None]
            + eps * torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        )

    def forward(self, x_t):
        # The sampling process goes here!
        x = x_t

        # Set device
        dev = x_t.device
        self.betas.to(dev)
        self.alphas.to(dev)
        self.alpha_bar.to(dev)
        self.sigma_t.to(dev)

        for t in tqdm(range(self.T - 1, -1, -1)):
            z = torch.randn_like(x_t)
            pred_mean = (
                x
                - (self.betas[t] / (torch.sqrt(1 - self.alpha_bar[t])))
                * self.decoder(x_t, torch.tensor([t], device=dev))
            ) / torch.sqrt(self.alphas[t])
            x = pred_mean + torch.sqrt(self.sigma_t[t]) * z
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        dev = x.device
        if not self.const_dev_set:
            self.betas = self.betas.to(dev)
            self.alphas = self.alphas.to(dev)
            self.alpha_bar = self.alpha_bar.to(dev)
            self.sigma_t = self.sigma_t.to(dev)

            self.const_dev_set = True

        # Sample timepoints
        t = torch.randint(0, self.T, size=(x.size(0),)).to(dev)

        # Sample noise
        eps = torch.randn_like(x).to(dev)

        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t)
        eps_pred = self.decoder(x_t.float(), t)

        # Compute loss
        loss = self.criterion(eps, eps_pred)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    decoder = UNet(t_embed_dim=64)
    ddpm = DDPM(decoder)
    sample = torch.randn(4, 3, 128, 128)
    loss = ddpm.training_step(sample, 1)
    print(loss)

    # Test sampling
    x_t = torch.randn(4, 3, 128, 128)
    samples = ddpm(x_t)
    print(samples.shape)
