import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from unet import Unet

from tqdm import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(pl.LightningModule):
    def __init__(self, decoder, beta_1=1e-4, beta_2=0.02, T=1000, lr=1e-4):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.criterion = nn.L1Loss()
        self.lr = lr

        # Flag to keep track of device settings
        self.setup_consts = False

    def setup_precomputed_const(self, dev):
        # Main
        self.betas = torch.linspace(self.beta_1, self.beta_2, self.T, device=dev)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_shifted = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])

        # Posterior covariance of the forward process
        self.post_variance = (
            self.betas * (1.0 - self.alpha_bar_shifted) / (1.0 - self.alpha_bar)
        )

        # Auxillary consts
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.minus_sqrt_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.post_coeff_1 = (
            torch.sqrt(self.alpha_bar_shifted) * self.betas / (1 - self.alpha_bar)
        )
        self.post_coeff_2 = (
            torch.sqrt(self.alphas)
            * (1 - self.alpha_bar_shifted)
            / (1 - self.alpha_bar)
        )

    def get_posterior_mean_covariance(self, x_t, t, clip_denoised=True):
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        # Generate the reconstruction from x_t
        x_recons = (
            x_t
            - self.decoder(x_t, t_) * extract(self.minus_sqrt_alpha_bar, t_, x_t.shape)
        ) / extract(self.sqrt_alpha_bar, t_, x_t.shape)

        # Clip
        if clip_denoised:
            x_recons.clamp_(-1.0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
        )
        post_variance = extract(self.post_variance, t_, x_t.shape)
        return post_mean, post_variance

    def forward(self, x_t):
        # The sampling process goes here!
        x = x_t

        # Set device
        dev = x_t.device
        if not self.setup_consts:
            self.setup_precomputed_const(dev)
            self.setup_consts = True

        for t in tqdm(reversed(range(0, self.T))):
            z = torch.randn_like(x_t)
            post_mean, post_variance = self.get_posterior_mean_covariance(x, t)
            # Langevin step!
            x = post_mean + torch.sqrt(post_variance) * z
        return x

    def compute_noisy_input(self, x_start, eps, t):
        assert eps.shape == x_start.shape
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        return x_start * extract(self.sqrt_alpha_bar, t, x_start.shape) + eps * extract(
            self.minus_sqrt_alpha_bar, t, x_start.shape
        )

    def training_step(self, batch, batch_idx):
        x = batch
        dev = x.device
        if not self.setup_consts:
            self.setup_precomputed_const(dev)
            self.setup_consts = True

        # Sample timepoints
        t = torch.randint(0, self.T, size=(x.size(0),), device=dev)

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t)
        eps_pred = self.decoder(x_t, t)

        # Compute loss
        loss = self.criterion(eps, eps_pred)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    decoder = Unet(64)
    ddpm = DDPM(decoder)
    sample = torch.randn(4, 3, 128, 128)
    loss = ddpm.training_step(sample, 1)
    print(loss)

    # Test sampling
    x_t = torch.randn(4, 3, 128, 128)
    samples = ddpm(x_t)
    print(samples.shape)
