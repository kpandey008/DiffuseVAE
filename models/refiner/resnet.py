import torch
import torch.nn as nn
from models import backbone


from models.backbone import resnet as resnet_models


SUPPORTED_BACKBONES = ["resnet18", "resnet34", "resnet50"]

OUT_CH = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


class Upsample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.BatchNorm2d(dim_out), nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetRefiner(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        pretrained=False,
        output_stride=32,
        attn_embed=512,
        n_heads=4,
        dropout=0,
        dec_ch_mult=(2, 2, 2, 2),
        **kwargs,
    ):
        super().__init__()
        assert backbone in SUPPORTED_BACKBONES
        assert output_stride in [8, 16, 32]
        replace_stride_with_dilation = [False, False, False]
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]

        self.backbone = getattr(resnet_models, backbone)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            **kwargs,
        )

        out_ch = OUT_CH[backbone]

        self.post_enc_block1 = Block(out_ch, attn_embed)
        self.post_enc_block2 = Block(attn_embed, attn_embed)
        self.attn = nn.MultiheadAttention(
            attn_embed,
            num_heads=n_heads,
            dropout=dropout,
        )

        # Decoder modules
        self.dec_modules = nn.ModuleList()
        dim_out = attn_embed
        for idx in range(len(dec_ch_mult)):
            self.dec_modules.append(
                nn.Sequential(
                    Upsample(dim_out, scale=2),
                    ResnetBlock(dim_out, dim_out // dec_ch_mult[idx]),
                )
            )
            dim_out = dim_out // dec_ch_mult[idx]

        # Final conv
        self.out_conv = nn.Conv2d(dim_out, 3, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.post_enc_block1(x)

        # Attention
        b, c, h, w = x.size()
        q = x.view(h * w, b, c)
        x_attn, _ = self.attn(q, q, q)
        x = x_attn.view(b, c, h, w)

        x = self.post_enc_block2(x)

        # Decoder
        for mod in self.dec_modules:
            x = mod(x)

        out = self.out_conv(x)

        assert out.size() == identity.size()
        return out


if __name__ == "__main__":
    refiner = ResnetRefiner(backbone="resnet34", dec_ch_mult=[2] * 5)
    sample = torch.randn(1, 3, 128, 128)
    out = refiner(sample)
    print(out.shape)
