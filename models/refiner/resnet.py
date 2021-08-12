import torch
import torch.nn as nn


import models.backbone.resnet as resnet_models


__supported_backbones = ["resnet18", "resnet34", "resnet50"]


class ResnetRefiner(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=False):
        assert backbone in __supported_backbones
        self.backbone = resnet_models.getattr(backbone)(pretrained=pretrained)
        # self.attn = nn.MultiheadAttention()
        self.decoder = nn.Sequential()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Attention

        return x


if __name__ == "__main__":
    refiner = ResnetRefiner()
