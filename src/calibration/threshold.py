"""Setting the decision threshold from normal parts alone.

The calibration parts are normal and unseen by the memory, so their scores show how high a good part
can score. Taking the conformal quantile of those scores bounds the share of good parts rejected:
for a new good part, the chance of scoring above the threshold is at most the target.
"""
import math

import numpy as np


def from_normals(scores: np.ndarray, false_reject: float) -> float:
    """The ceil((n + 1)(1 - target))-th smallest calibration score."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - false_reject))
    if rank > n:
        raise ValueError(f"{n} calibration parts cannot support a false-reject target of {false_reject}")
    return float(np.sort(scores)[rank - 1])


def oracle(scores: np.ndarray, defective: np.ndarray) -> float:
    """The threshold with the highest F1 on the test parts. It needs defects, so it is a reference only."""
    best, threshold = -1.0, float(scores.max())
    for t in np.unique(scores):
        rejected = scores >= t
        tp = np.sum(rejected & defective)
        f1 = 2 * tp / (rejected.sum() + defective.sum())
        if f1 > best:
            best, threshold = f1, float(t)
    return threshold
