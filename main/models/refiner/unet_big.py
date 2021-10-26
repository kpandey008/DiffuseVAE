import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # A simple conv-bn-relu unit. Does not change the spatial size
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.block1 = Block(in_channels, mid_channels)
        self.block2 = Block(mid_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x

        # Block 1
        x = self.block1(x)

        # Block 2
        x = self.block2(x) + self.residual_conv(identity)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.block1 = ResidualBlock(
            in_channels, out_channels, mid_channels=in_channels // 2
        )
        self.block2 = ResidualBlock(
            out_channels, out_channels, mid_channels=out_channels // 2
        )

    def forward(self, x):
        x = self.down(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block1 = ResidualBlock(
            in_channels, out_channels, mid_channels=in_channels // 2
        )
        self.block2 = ResidualBlock(
            out_channels, out_channels, mid_channels=out_channels // 2
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(pl.LightningModule):
    def __init__(self, n_heads=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Initial conv
        self.inc = nn.Conv2d(3, 64, 3, padding=1)

        # Down blocks
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)

        self.down_attn = nn.MultiheadAttention(
            512, n_heads, dropout=0.1, batch_first=True
        )

        # Up blocks
        self.up1 = Up(1024, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.up5 = Up(128, 64)

        self.up_attn = nn.MultiheadAttention(
            256, n_heads, dropout=0.1, batch_first=True
        )

        # Final conv
        self.outc = OutConv(64, 3)

        self.lr = lr

    def forward(self, x):
        x1 = self.inc(x)

        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Attention here!
        B, C, H, W = x4.shape
        q = x4.view(B, H * W, C)
        x4, _ = self.down_attn(q, q, q)
        x4 = x4.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Resume downsampling
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # Upsampling with concat
        x = self.up1(x6, x5)
        x = self.up2(x, x4)

        # Attention here!
        B, C, H, W = x.shape
        q = x.view(B, H * W, C)
        x, _ = self.up_attn(q, q, q)
        x = x.view(B, C, H, W)

        # Resume upsampling
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return torch.sigmoid(self.outc(x))

    def training_step(self, batch, batch_idx):
        recons, img = batch
        preds = self(recons)

        # Compute loss
        l1_loss = nn.L1Loss(reduction="mean")
        recons_loss = l1_loss(preds, img)
        self.log("Recons Loss", recons_loss, prog_bar=True)

        return recons_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    sample = torch.randn(4, 3, 128, 128)
    unet = UNet(64)
    out = unet(sample)
    print(out.shape)
