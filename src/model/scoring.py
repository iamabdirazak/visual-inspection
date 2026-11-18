"""From patch features to a defect map and a part score.

A patch's anomaly is its distance to the nearest patch in the memory of normal appearance. The patch
grid is brought up to the input size and smoothed, which gives the defect map, and the part's score
is the map's highest value: one bad spot is enough to make a bad part.
"""
import torch
import torch.nn.functional as F
from torch import nn

SIGMA = 4  # smoothing of the defect map, in input pixels


def nearest(features: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """(Q, C) queries and (M, C) memory to the squared distance from each query to every memory patch."""
    return ((features * features).sum(1, keepdim=True) - 2 * features @ memory.T
            + (memory * memory).sum(1)[None]).clamp_min(0)


def kernel(sigma: float = SIGMA) -> torch.Tensor:
    radius = int(4 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def smooth(grid: torch.Tensor, size: tuple[int, int], k: torch.Tensor) -> torch.Tensor:
    """(B, 1, h, w) patch scores to a (B, 1, H, W) map, upsampled and blurred in two separable passes."""
    m = F.interpolate(grid, size=size, mode="bilinear", align_corners=False)
    r = len(k) // 2
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


class Inspection(nn.Module):
    """The whole model in one graph: image in, defect map and part score out."""

    def __init__(self, backbone: nn.Module, memory: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("memory", memory.float().contiguous())
        self.register_buffer("kernel", kernel())

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        b, c, h, w = features.shape
        queries = features.permute(0, 2, 3, 1).reshape(-1, c)
        grid = nearest(queries, self.memory).amin(1).sqrt().reshape(b, 1, h, w)
        defect_map = smooth(grid, image.shape[-2:], self.kernel)
        return defect_map, defect_map.flatten(1).amax(1)
