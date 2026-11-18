"""Patch features from a pretrained ResNet-18, taken from the middle of the network.

Layers 2 and 3 describe local structure, such as a solder joint or a trace edge, without the
ImageNet-specific meaning of the last layer. Each feature is averaged over its 3 x 3 neighbourhood,
so a patch also carries a little of its surroundings, and layer 3 is brought up to layer 2's grid.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

STRIDE = 8      # one patch for every 8 x 8 pixels of the input
CHANNELS = 384  # 128 from layer 2 and 256 from layer 3


class Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) normalised images to (B, 384, H/8, W/8) patch features."""
        middle = self.layer2(self.stem(x))
        deep = self.layer3(middle)
        deep = F.interpolate(self.pool(deep), size=middle.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([self.pool(middle), deep], dim=1)
