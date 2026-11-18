"""Scores the calibration and test parts under every setting, then writes one report per study.

python -m src.evaluation.run         after src.evaluation.latency, which fixes the released setting

Studies, each in reports/:
  resolution.json   the working resolution, at a memory of 2,000 patches
  memory.json       the memory size, at the released resolution
  threshold.json    the threshold from normal parts against the best threshold found with defects
  shared.json       one memory shared by the four boards against one memory per board
"""
import json

import cv2
import numpy as np
import torch

from src.acquisition.visa import mask, prepare, read, size_for, split
from src.calibration.threshold import from_normals, oracle
from src.evaluation.latency import MEMORY_SIZES, RESOLUTIONS, measure, released, session
from src.evaluation.metrics import aupro, auroc, caught_by_size, operating_point, size_bin
from src.export.onnx import export
from src.model.backbone import Backbone
from src.model.memory import collect, order
from src.model.scoring import Inspection, kernel, nearest, smooth
from src.paths import CACHE, FALSE_REJECT_TARGET, PRODUCTS, REPORTS

EVALUATION_SIZE = 512  # maps and masks are compared at one size, so every resolution is judged alike
STUDY_MEMORY = 2000


@torch.inference_mode()
def score(backbone, memory, sizes, samples, resolution, keep_grids=False):
    """Part scores (parts, sizes) for each memory prefix, and optionally the patch grids behind them."""
    k, scores, grids = kernel(), [], []
    for sample in samples:
        image = torch.from_numpy(prepare(read(sample.image), resolution))
        features = backbone(image)[0]
        c, h, w = features.shape
        distances = nearest(features.flatten(1).T, memory)
        part = []
        for size in sizes:
            grid = distances[:, :size].amin(1).sqrt().reshape(1, 1, h, w)
            part.append(smooth(grid, image.shape[-2:], k).max().item())
            if keep_grids:
                grids.append(grid[0, 0].numpy().astype(np.float16))
        scores.append(part)
    result = {"scores": np.array(scores)}
    if keep_grids:
        result["grids"] = np.stack(grids).reshape(len(samples), len(sizes), h, w)
    return result


def scored(product, resolution, backbone):
    """Calibration and test scores for one board at one resolution, cached so the studies can be recomputed."""
    path = CACHE / f"{product}-{resolution}.npz"
    if not path.exists():
        fit, calibration, test = split(product)
        memory = order(collect(backbone, fit, resolution), max(MEMORY_SIZES))
        calib = score(backbone, memory, MEMORY_SIZES, calibration, resolution)
        tested = score(backbone, memory, MEMORY_SIZES, test, resolution, keep_grids=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, calibration=calib["scores"], test=tested["scores"], grids=tested["grids"])
        print("scored", product, resolution, flush=True)
    return dict(np.load(path))


def ground_truth(product):
    """Defect labels, size bins and masks at the evaluation size, for the test parts."""
    test = split(product)[2]
    image_size = read(test[0].image).shape[:2]
    size = size_for(image_size, EVALUATION_SIZE)
    masks = [mask(s, size) if s.defective else None for s in test]
    return {"defective": np.array([s.defective for s in test]), "masks": masks, "size": size,
            "bins": [size_bin(m) for m in masks if m is not None], "image_size": image_size}


def maps(grids, truth, resolution):
    """Patch grids to defect maps at the evaluation size."""
    w, h = size_for(truth["image_size"], resolution)
    k = kernel()
    out = []
    for g in grids:
        m = smooth(torch.from_numpy(g.astype(np.float32))[None, None], (h, w), k)[0, 0].numpy()
        out.append(cv2.resize(m, truth["size"], interpolation=cv2.INTER_LINEAR))
    return out


def judge(scores, calibration, truth, grids=None, resolution=None):
    """Every measure for one board under one setting."""
    threshold = from_normals(calibration, FALSE_REJECT_TARGET)
    defective = truth["defective"]
    out = {"auroc": auroc(scores, defective), **operating_point(scores, defective, threshold),
           "caught_by_size": caught_by_size(scores[defective], truth["bins"], threshold)}
    if grids is not None:
        out["aupro"] = aupro(maps(grids, truth, resolution), truth["masks"])
    return out


def mean(per_product, key):
    return float(np.mean([r[key] for r in per_product.values()]))


def summary(per_product):
    keys = [k for k in ("auroc", "aupro", "false_reject_rate", "escape_rate") if k in next(iter(per_product.values()))]
    return {k: mean(per_product, k) for k in keys}


def write(name, report):
    (REPORTS / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(name, json.dumps(report.get("mean", ""), indent=None))


def main():
    timing = json.loads((REPORTS / "latency.json").read_text())
    setting = released(timing)
    time_of = {(r["resolution"], r["memory"]): r for r in timing["settings"]}
    backbone = Backbone()
    truths = {p: ground_truth(p) for p in PRODUCTS}
    cache = {(p, r): scored(p, r, backbone) for r in RESOLUTIONS for p in PRODUCTS}
    study = MEMORY_SIZES.index(STUDY_MEMORY)

    rows = []
    for r in RESOLUTIONS:
        per = {p: judge(cache[p, r]["test"][:, study], cache[p, r]["calibration"][:, study], truths[p],
                        cache[p, r]["grids"][:, study], r) for p in PRODUCTS}
        rows.append({"resolution": r, "memory": STUDY_MEMORY, **time_of[r, STUDY_MEMORY], "mean": summary(per),
                     "products": per})
    write("resolution", {"settings": rows, "mean": [row["mean"] for row in rows]})

    r, chosen = setting["resolution"], MEMORY_SIZES.index(setting["memory"])
    rows = []
    for i, size in enumerate(MEMORY_SIZES):
        per = {p: judge(cache[p, r]["test"][:, i], cache[p, r]["calibration"][:, i], truths[p],
                        cache[p, r]["grids"][:, i], r) for p in PRODUCTS}
        rows.append({"resolution": r, "memory": size, **time_of[r, size], "mean": summary(per), "products": per})
    write("memory", {"settings": rows, "mean": [row["mean"] for row in rows]})

    per = {}
    for p in PRODUCTS:
        scores, calibration, defective = cache[p, r]["test"][:, chosen], cache[p, r]["calibration"][:, chosen], truths[p]["defective"]
        per[p] = {"calibration_parts": len(calibration),
                  "from_normals": operating_point(scores, defective, from_normals(calibration, FALSE_REJECT_TARGET)),
                  "with_defects": operating_point(scores, defective, oracle(scores, defective))}
    write("threshold", {"setting": setting, "target_false_reject_rate": FALSE_REJECT_TARGET, "products": per,
                        "mean": {k: {m: float(np.mean([v[k][m] for v in per.values()]))
                                     for m in ("false_reject_rate", "escape_rate")}
                                 for k in ("from_normals", "with_defects")}})

    write("shared", shared(backbone, setting, cache, truths, time_of))


def shared(backbone, setting, cache, truths, time_of):
    """One memory for all four boards, the same size as one board's and four times larger, with one threshold."""
    r, size = setting["resolution"], setting["memory"]
    sizes = [size, 4 * size]
    splits = {p: split(p) for p in PRODUCTS}
    memory = order(torch.cat([collect(backbone, splits[p][0], r) for p in PRODUCTS]), max(sizes))
    calib = {p: score(backbone, memory, sizes, splits[p][1], r)["scores"] for p in PRODUCTS}
    tested = {p: score(backbone, memory, sizes, splits[p][2], r)["scores"] for p in PRODUCTS}
    chosen = MEMORY_SIZES.index(size)
    separate = {p: judge(cache[p, r]["test"][:, chosen], cache[p, r]["calibration"][:, chosen], truths[p])
                for p in PRODUCTS}
    rows = [{"model": "one per board", "memory": size, "p95_ms": time_of[r, size]["p95_ms"],
             "mean": summary(separate), "products": separate}]
    frame = read(splits["pcb1"][2][0].image)
    for i, s in enumerate(sizes):
        pooled = np.concatenate([calib[p][:, i] for p in PRODUCTS])
        per = {p: judge(tested[p][:, i], pooled, truths[p]) for p in PRODUCTS}
        timed = time_of.get((r, s))
        if timed is None:
            path = CACHE / "latency.onnx"
            export(Inspection(backbone, memory[:s]), path, r)
            timed = measure(session(path), frame, r)
        rows.append({"model": "shared", "memory": s, "p95_ms": timed["p95_ms"], "mean": summary(per), "products": per})
    return {"setting": setting, "models": rows, "mean": [row["mean"] for row in rows]}


if __name__ == "__main__":
    main()
