import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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
    def __init__(self, in_channels, out_channels, time_embed_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.block1 = Block(in_channels, mid_channels)
        self.block2 = Block(mid_channels, out_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, mid_channels), nn.ReLU()
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t_embed):
        identity = x

        # Block 1
        embed = self.time_mlp(t_embed)
        x = self.block1(x)
        x = embed[:, :, None, None] + x

        # Block 2
        x = self.block2(x) + self.residual_conv(identity)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, t_embed_dim):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.r_block = ResidualBlock(in_channels, out_channels, t_embed_dim)

    def forward(self, x, t_embed):
        x = self.down(x)
        x = self.r_block(x, t_embed)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, t_embed_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r_block = ResidualBlock(
            in_channels, out_channels, t_embed_dim, mid_channels=in_channels // 2
        )

    def forward(self, x1, x2, t_embed):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.r_block(x, t_embed)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, t_embed_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim * 4),
            nn.ReLU(),
            nn.Linear(t_embed_dim * 4, t_embed_dim),
        )

        # Initial conv
        self.inc = nn.Conv2d(3, 64, 3, padding=1)

        # Down blocks
        self.down1 = Down(64, 128, t_embed_dim)
        self.down2 = Down(128, 256, t_embed_dim)
        self.down3 = Down(256, 512, t_embed_dim)
        self.down4 = Down(512, 512, t_embed_dim)

        # Up blocks
        self.up1 = Up(1024, 256, t_embed_dim)
        self.up2 = Up(512, 128, t_embed_dim)
        self.up3 = Up(256, 64, t_embed_dim)
        self.up4 = Up(128, 64, t_embed_dim)

        # Final conv
        self.outc = OutConv(64, 3)

    def forward(self, x, t):
        t_embed = self.time_mlp(t)

        x1 = self.inc(x)

        # Downsampling
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x4 = self.down3(x3, t_embed)
        x5 = self.down4(x4, t_embed)

        # Upsampling with concat
        x = self.up1(x5, x4, t_embed)
        x = self.up2(x, x3, t_embed)
        x = self.up3(x, x2, t_embed)
        x = self.up4(x, x1, t_embed)
        return self.outc(x)


if __name__ == "__main__":
    sample = torch.randn(1, 3, 128, 128)
    unet = UNet(64)
    out = unet(sample, t=torch.tensor([1]))
    print(out.shape)
