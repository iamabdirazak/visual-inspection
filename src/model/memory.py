"""The memory of normal appearance: patch features from normal parts, reduced to a coreset."""
import numpy as np
import torch

from src.acquisition.visa import Sample, prepare, read
from src.model.backbone import Backbone
from src.model.coreset import greedy

PATCHES_PER_IMAGE = 512  # a random share of each fit image's patches; the boards are aligned, so every
                         # location is still seen many times across the fit set


@torch.inference_mode()
def collect(backbone: Backbone, samples: list[Sample], resolution: int, seed: int = 0) -> torch.Tensor:
    """(N, 384) patch features sampled from the fit images."""
    rng, patches = np.random.default_rng(seed), []
    for sample in samples:
        features = backbone(torch.from_numpy(prepare(read(sample.image), resolution)))[0]
        features = features.flatten(1).T
        keep = rng.choice(len(features), min(PATCHES_PER_IMAGE, len(features)), replace=False)
        patches.append(features[keep])
    return torch.cat(patches)


def order(patches: torch.Tensor, largest: int) -> torch.Tensor:
    """The patches reordered by the coreset, so memory[:k] is the coreset of size k."""
    return patches[greedy(patches, largest)]
