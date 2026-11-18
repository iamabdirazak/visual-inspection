"""How well parts are judged and defects located."""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

SIZES = [("Small", 0.0, 0.002), ("Medium", 0.002, 0.01), ("Large", 0.01, 1.0)]  # share of the part's area


def auroc(scores: np.ndarray, defective: np.ndarray) -> float:
    return float(roc_auc_score(defective, scores))


def operating_point(scores: np.ndarray, defective: np.ndarray, threshold: float) -> dict:
    """What the station does at a threshold: good parts rejected, and defective parts let through."""
    rejected = scores >= threshold
    return {"threshold": threshold,
            "false_reject_rate": float(rejected[~defective].mean()),
            "escape_rate": float((~rejected[defective]).mean())}


def size_bin(mask: np.ndarray) -> str:
    share = mask.mean()
    return next(name for name, low, high in SIZES if low <= share < high)


def caught_by_size(scores: np.ndarray, bins: list[str], threshold: float) -> dict:
    """Share of defective parts rejected, for each defect size."""
    out = {}
    for name, _, _ in SIZES:
        chosen = np.array([b == name for b in bins])
        out[name.lower()] = {"parts": int(chosen.sum()),
                             "caught": float((scores[chosen] >= threshold).mean()) if chosen.any() else None}
    return out


def aupro(maps: list[np.ndarray], masks: list[np.ndarray | None], limit: float = 0.3, steps: int = 200) -> float:
    """Area under the per-region overlap curve up to a 30% false-positive rate, normalised to 1.

    Each connected defect counts equally, however large, so small defects weigh as much as large ones.
    """
    normal = np.concatenate([m[~k].ravel() if k is not None else m.ravel() for m, k in zip(maps, masks)])
    fpr = np.linspace(0, limit, steps)
    thresholds = np.quantile(normal, 1 - fpr)
    regions = []
    for m, k in zip(maps, masks):
        if k is None:
            continue
        labels, count = ndimage.label(k)
        regions += [np.sort(m[labels == i]) for i in range(1, count + 1)]
    overlap = np.array([[1 - np.searchsorted(r, t) / len(r) for t in thresholds] for r in regions]).mean(0)
    return float(np.trapezoid(overlap, fpr) / limit)
