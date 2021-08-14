import math
import torch
import torch.nn as nn


class Upsample(nn.Module):
    """Upsampling module using the interpolation and a conv module"""

    def __init__(self, dim, mode="nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class Downsample(nn.Module):
    """Downsampling block using strided convolutions"""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    """Residual block implementation as in original DDPM implementation"""

    def __init__(self, in_dim, out_dim, t_embed_dim, dropout=0, num_groups=32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block1 = Block(in_dim, out_dim, dropout=dropout, groups=num_groups)
        self.block2 = Block(out_dim, out_dim, dropout=dropout, groups=num_groups)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, out_dim),
        )
        self.res_conv = (
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)

        # Timestep embedding
        h += self.mlp(t_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoid embedding input to the DDPM denoiser"""

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


class ResDown(nn.Module):
    def __init__(self, in_ch, out_ch, t_embed, down=True):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, t_embed, dropout=0, num_groups=32)
        self.res2 = ResnetBlock(out_ch, out_ch, t_embed, dropout=0, num_groups=32)
        self.down = Downsample(out_ch) if down else None

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)

        if self.down is not None:
            x = self.down(x)
        return x


class ResUp(nn.Module):
    def __init__(self, in_ch, out_ch, t_embed, up=True):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, t_embed, dropout=0, num_groups=32)
        self.res2 = ResnetBlock(out_ch, out_ch, t_embed, dropout=0, num_groups=32)
        self.up = Upsample(out_ch) if up else None

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)

        if self.up is not None:
            x = self.up(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        groups=8,
        channels=3,
        dropout=0,
        n_heads=1,
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.out_dim = 3 if out_dim is None else out_dim
        self.groups = groups
        self.dropout = dropout
        self.n_heads = n_heads

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        # Downsampling
        self.in_conv = nn.Conv2d(self.channels, self.dim, 3, padding=1, stride=1)
        self.down_1 = ResDown(self.dim, self.dim, self.dim)
        self.down_2 = ResDown(self.dim, self.dim, self.dim)
        self.down_3 = ResDown(self.dim, 2 * self.dim, self.dim)
        self.down_4 = ResDown(2 * self.dim, 2 * self.dim, self.dim)
        self.down_5 = ResDown(2 * self.dim, 4 * self.dim, self.dim)
        self.down_6 = ResDown(4 * self.dim, 4 * self.dim, self.dim, down=False)

        self.down_attn = nn.MultiheadAttention(2 * self.dim, self.n_heads)

        # Middle
        mid_in_dim = 4 * self.dim
        self.mid_block1 = ResnetBlock(
            mid_in_dim, mid_in_dim, self.dim, num_groups=groups, dropout=self.dropout
        )
        self.mid_attn = nn.MultiheadAttention(mid_in_dim, self.n_heads)
        self.mid_block2 = ResnetBlock(
            mid_in_dim, mid_in_dim, self.dim, num_groups=groups, dropout=self.dropout
        )

        # Upsampling
        self.up_1 = ResUp(8 * self.dim, 4 * self.dim, self.dim, up=False)
        self.up_2 = ResUp(8 * self.dim, 2 * self.dim, self.dim)
        self.up_3 = ResUp(4 * self.dim, 2 * self.dim, self.dim)
        self.up_4 = ResUp(4 * self.dim, self.dim, self.dim)
        self.up_5 = ResUp(2 * self.dim, self.dim, self.dim)
        self.up_6 = ResUp(2 * self.dim, self.dim, self.dim)

        self.up_attn = nn.MultiheadAttention(2 * self.dim, self.n_heads)
        self.final_conv = Block(self.dim, self.out_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.in_conv(x)

        # Downsample
        x_1 = self.down_1(x, t)
        x_2 = self.down_2(x_1, t)
        x_3 = self.down_3(x_2, t)
        x_4 = self.down_4(x_3, t)

        b, c, h, w = x_4.shape
        x_4 = x_4.view(h * w, b, c)
        x_4, _ = self.down_attn(x_4, x_4, x_4)
        x_4 = x_4.view(b, c, h, w)
        x_5 = self.down_5(x_4, t)
        x_6 = self.down_6(x_5, t)

        print(x_6.shape)

        # Middle
        x = self.mid_block1(x_6, t)
        b, c, h, w = x.shape
        x = x.view(h * w, b, c)
        x, _ = self.mid_attn(x, x, x)
        x = x.view(b, c, h, w)
        x = self.mid_block2(x, t)

        # Upsample
        x = self.up_1(torch.cat([x, x_6], dim=1), t)
        x = self.up_2(torch.cat([x, x_5], dim=1), t)

        b, c, h, w = x.shape
        x = x.view(h * w, b, c)
        x, _ = self.up_attn(x, x, x)
        x = x.view(b, c, h, w)
        x = self.up_3(torch.cat([x, x_4], dim=1), t)
        x = self.up_4(torch.cat([x, x_3], dim=1), t)
        x = self.up_5(torch.cat([x, x_2], dim=1), t)
        x = self.up_6(torch.cat([x, x_1], dim=1), t)

        # Final output
        return self.final_conv(x)


if __name__ == "__main__":
    unet = Unet(64)
    sample = torch.randn(1, 3, 256, 256)
    out = unet(sample, torch.tensor([1]))
    print(out.shape)
