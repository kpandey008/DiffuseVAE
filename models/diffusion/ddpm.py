import pytorch_lightning as pl
import torch
import torch.nn as nn

from unet import UNet

from tqdm import tqdm


class DDPM(pl.LightningModule):
    def __init__(self, decoder, beta_1=1e-4, beta_2=0.02, T=1000, lr=1e-4):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.criterion = nn.MSELoss()
        self.lr = lr

        # Precompute alpha_t for all t values
        self.betas = torch.linspace(self.beta_1, self.beta_2, self.T)
        self.alpha_bar = torch.zeros_like(self.betas)
        alpha = 1
        for t in range(self.T):
            alpha = alpha * (1 - self.betas[t])
            self.alpha_bar[t] = alpha

        # Precompute sigma_t for sampling
        self.sigma_t = torch.zeros_like(self.betas - 1)
        for t in range(self.T - 1):
            self.sigma_t[t] = (
                self.betas[t + 1]
                * (1 - self.alpha_bar[t])
                / (1 - self.alpha_bar[t + 1])
            )

    def compute_noisy_input(self, x, eps, t):
        return (
            x * torch.sqrt(self.alpha_bar[t])[:, None, None, None]
            + eps * torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        )

    def forward(self, x_t):
        # The sampling process goes here!
        x = x_t

        for t in tqdm(range(self.T - 1, 1, -1)):
            z = torch.randn_like(x_t)
            x = (
                x
                - (self.betas[t] / (torch.sqrt(1 - self.alpha_bar[t])))
                * self.decoder(x_t, torch.tensor([t]))
            ) / torch.sqrt(1 - self.betas[t]) + self.sigma_t[t] * z

        # Output only the mean for the last step
        x = x - torch.sqrt(self.betas[0]) * self.decoder(x, torch.tensor([0]))
        return x

    def training_step(self, batch, batch_idx):
        x = batch

        # Sample timepoints
        t = torch.randint(0, self.T, size=(x.size(0),))

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t)
        eps_pred = self.decoder(x_t, t)

        # Compute loss
        loss = self.criterion(eps, eps_pred)
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
