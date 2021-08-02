import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, mid_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResBlockv2(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=True,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = nn.Conv2d(in_width, middle_width, 1, bias=False)
        self.c2 = nn.Conv2d(middle_width, middle_width, 3, bias=False)
        self.c3 = nn.Conv2d(middle_width, middle_width, 3, bias=False)
        self.c4 = nn.Conv2d(middle_width, out_width, 1, bias=False)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class Encoder(nn.Module):
    def __init__(self, res_block_conf=[2, 2, 2], code_size=1024):
        super().__init__()
        assert len(res_block_conf) == 3
        self.code_size = code_size

        blocks = [
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.Sequential(*[ResBlock(64, 32) for _ in range(res_block_conf[0])]),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.Sequential(*[ResBlock(128, 64) for _ in range(res_block_conf[1])]),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.Sequential(*[ResBlock(256, 128) for _ in range(res_block_conf[2])]),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.fc = nn.Linear(512, self.code_size)

        self.mod = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.mod(input)
        x = self.fc(x.view(-1, 1, 1, 512))
        return x


class Decoder(nn.Module):
    def __init__(self, inplanes, res_block_conf=[2, 2]):
        super().__init__()
        assert len(res_block_conf) == 2

        blocks = [
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv2d(inplanes, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv2d(256, 64, 3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(64, 64) for _ in range(res_block_conf[1])]),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.Sequential(*[ResBlock(32, 32) for _ in range(res_block_conf[1])]),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
        ]

        self.mod = nn.Sequential(*blocks)

    def forward(self, input):
        return self.mod(input)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):
    def __init__(self, alpha=1.0, lr=1e-4, enc_kwargs={}, dec_kwargs={}):
        super().__init__()
        self.save_hyperparameters()
        self.alpha = alpha
        self.lr = lr

        # Encoder architecture
        self.enc = Encoder(**enc_kwargs)

        # Latents
        self.relu = nn.ReLU()
        code_size = self.enc.code_size
        self.fc_mu = nn.Linear(code_size, code_size)
        self.fc_sigma = nn.Linear(code_size, code_size)

        # Decoder Architecture
        self.dec = Decoder(code_size, **dec_kwargs)

    def encode(self, x):
        x = self.enc(x)
        return self.relu(self.fc_mu(x)), self.relu(self.fc_sigma(x))

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x = batch

        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z.view(-1, self.enc.code_size, 1, 1))

        # Compute loss
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(decoder_out, x)
        kl_loss = self.compute_kl(mu, logvar)
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)

        total_loss = recons_loss + self.alpha * kl_loss
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
