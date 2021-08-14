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


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        attn_resolutions=[0, 0, 0, 1],
        n_residual_blocks=1,
        dim_mults=[1, 2, 4, 8],
        groups=8,
        channels=3,
        dropout=0,
        n_heads=1,
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.out_dim = 3 if out_dim is None else out_dim
        self.attn_resolutions = attn_resolutions
        self.n_residual_blocks = n_residual_blocks
        self.dim_mults = dim_mults
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
        self.down_modules = nn.ModuleDict()
        d_in_dim = self.dim
        for idx, (attn, ch_mul) in enumerate(
            zip(self.attn_resolutions, self.dim_mults)
        ):
            # Add residual blocks for the current resolution
            res_modules = nn.ModuleList()
            for i in range(self.n_residual_blocks):
                res_in_dim = d_in_dim if i == 0 else ch_mul * self.dim
                res_modules.append(
                    ResnetBlock(
                        res_in_dim,
                        ch_mul * self.dim,
                        self.dim,
                        dropout=self.dropout,
                        num_groups=self.groups,
                    )
                )
                if attn:
                    res_modules.append(
                        nn.MultiheadAttention(ch_mul * self.dim, self.n_heads)
                    )
            if idx != len(self.dim_mults) - 1:
                res_modules.append(Downsample(ch_mul * self.dim))
            self.down_modules[f"{idx}"] = res_modules

            # Update the input channels for the next resolution
            d_in_dim = ch_mul * self.dim

        # Middle
        mid_in_dim = self.dim * self.dim_mults[-1]
        self.mid_block1 = ResnetBlock(
            mid_in_dim, mid_in_dim, self.dim, num_groups=groups, dropout=self.dropout
        )
        self.mid_attn = nn.MultiheadAttention(mid_in_dim, self.n_heads)
        self.mid_block2 = ResnetBlock(
            mid_in_dim, mid_in_dim, self.dim, num_groups=groups, dropout=self.dropout
        )

        # Upsampling
        self.up_modules = nn.ModuleDict()
        u_in_dim = 2 * mid_in_dim  # Due to concat
        up_dim_mults = [1] + self.dim_mults[:-1]
        for idx in reversed(range(len(up_dim_mults) + 1)):
            # Add residual blocks for the current resolution
            res_modules = nn.ModuleList()
            for i in range(self.n_residual_blocks):
                res_in_dim = u_in_dim if i == 0 else up_dim_mults[idx] * self.dim
                res_modules.append(
                    ResnetBlock(
                        res_in_dim,
                        up_dim_mults[idx] * self.dim,
                        self.dim,
                        dropout=self.dropout,
                        num_groups=self.groups,
                    )
                )
                if self.attn_resolutions[idx]:
                    res_modules.append(
                        nn.MultiheadAttention(
                            up_dim_mults[idx] * self.dim, self.n_heads
                        )
                    )
            if idx != len(self.dim_mults) - 1:
                res_modules.append(Upsample(up_dim_mults[idx] * self.dim))
            self.up_modules[f"{idx}"] = res_modules

            # Update the input channels for the next resolution
            u_in_dim = 2 * up_dim_mults[idx] * self.dim

        self.final_conv = Block(self.dim, self.out_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)

        # Downsample
        x = self.in_conv(x)
        down_outputs = {}
        for res, d_mod_list in self.down_modules.items():
            for d_mod in d_mod_list:
                if isinstance(d_mod, nn.MultiheadAttention):
                    # Reshape and attend!
                    b, c, h, w = x.size()
                    x = x.view(h * w, b, c)
                    x_attn, _ = d_mod(x, x, x)
                    x = x_attn.view(b, c, h, w)
                    continue

                if isinstance(d_mod, ResnetBlock):
                    x = d_mod(x, t)
                    continue

                # Else continue
                x = d_mod(x)
            down_outputs[res] = x

        # Middle bottleneck and attention
        x = self.mid_block1(x, t)
        b, c, h, w = x.size()
        x = x.view(h * w, b, c)
        x_attn, _ = self.mid_attn(x, x, x)
        x = x_attn.view(b, c, h, w)

        x = self.mid_block2(x, t)

        # Upsample
        for res, u_mod_list in self.up_modules.items():
            d_x = down_outputs[res]
            x = torch.cat([x, d_x], dim=1)
            for u_mod in u_mod_list:
                if isinstance(u_mod, nn.MultiheadAttention):
                    # Reshape and attend!
                    b, c, h, w = x.size()
                    x = x.view(h * w, b, c)
                    x_attn, _ = u_mod(x, x, x)
                    x = x_attn.view(b, c, h, w)
                    continue

                if isinstance(u_mod, ResnetBlock):
                    x = u_mod(x, t)
                    continue

                # Else continue
                x = u_mod(x)

        # Final output
        return self.final_conv(x)


if __name__ == "__main__":
    unet = Unet(
        128,
        attn_resolutions=[0, 0, 0, 1, 0],
        dim_mults=[1, 1, 2, 2, 4],
        n_residual_blocks=2,
    )
    sample = torch.randn(1, 3, 128, 128)
    out = unet(sample, torch.tensor([1]))
    print(out.shape)
