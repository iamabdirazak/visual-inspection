"""Greedy coreset selection: a small set of patches that covers all the others.

Each step adds the patch farthest from everything chosen so far, so rare but normal structures are
kept while near-duplicates are dropped. The order is nested, so the first k patches are the coreset
of size k for every k, and one selection serves every memory size.
"""
import torch

PROJECTION = 128  # distances are compared in a random projection, which keeps them close and costs less


def greedy(features: torch.Tensor, size: int, seed: int = 0) -> torch.Tensor:
    """Indices of the selected patches, in the order they were chosen."""
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(features.shape[1], PROJECTION, generator=generator) / PROJECTION ** 0.5
    points = features @ projection
    size = min(size, len(points))
    chosen = torch.empty(size, dtype=torch.long)
    chosen[0] = torch.randint(len(points), (1,), generator=generator)
    nearest = torch.linalg.vector_norm(points - points[chosen[0]], dim=1)
    for i in range(1, size):
        chosen[i] = torch.argmax(nearest)
        nearest = torch.minimum(nearest, torch.linalg.vector_norm(points - points[chosen[i]], dim=1))
    return chosen
