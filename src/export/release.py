"""Builds the released models, sets each threshold from normal parts and writes the model card.

python -m src.export.release         after src.evaluation.run

The setting comes from the latency rule, and the choice between one memory per board and one shared
memory follows a rule fixed in advance: the shared memory is released only if its mean AUROC is within
0.01 of the separate memories at the same size and it lets no more defective parts through.
"""
import json

import numpy as np
import torch

from src.acquisition.visa import prepare, read, split
from src.calibration.threshold import from_normals
from src.evaluation.latency import released, session
from src.export.onnx import export
from src.model.backbone import Backbone
from src.model.memory import collect, order
from src.model.scoring import SIGMA, Inspection
from src.paths import CALIBRATION_IMAGES, FALSE_REJECT_TARGET, FIT_IMAGES, MODELS, PRODUCTS, REPORTS, THREADS

AUROC_MARGIN = 0.01


def load(name):
    return json.loads((REPORTS / f"{name}.json").read_text())


def share(report) -> bool:
    separate, same_size = report["models"][0]["mean"], report["models"][1]["mean"]
    return (same_size["auroc"] >= separate["auroc"] - AUROC_MARGIN
            and same_size["escape_rate"] <= separate["escape_rate"])


def main():
    setting = released(load("latency"))
    resolution, size = setting["resolution"], setting["memory"]
    shared = share(load("shared"))
    groups = {"inspection": PRODUCTS} if shared else {p: [p] for p in PRODUCTS}
    backbone = Backbone()
    models = []
    for name, products in groups.items():
        splits = [split(p) for p in products]
        memory = order(torch.cat([collect(backbone, s[0], resolution) for s in splits]), size)
        path = MODELS / f"{name}.onnx"
        export(Inspection(backbone, memory), path, resolution)
        model = session(path)
        calibration = np.array([model.run(None, {"image": prepare(read(c.image), resolution)})[1][0]
                                for s in splits for c in s[1]])
        models.append({"file": path.name, "products": products, "size_mb": round(path.stat().st_size / 2**20, 1),
                       "threshold": round(from_normals(calibration, FALSE_REJECT_TARGET), 4)})
        print(models[-1], flush=True)

    parameters = sum(p.numel() for p in backbone.parameters())
    results = load("shared")["models"][1 if shared else 0]
    threshold = load("threshold")
    card = {
        "name": "Visual Inspection Model",
        "task": "Decide whether a printed circuit board is defective and locate the defect, learned from normal boards only",
        "models": models,
        "architecture": {
            "backbone": "ResNet-18 to layer 3, ImageNet weights, frozen; layers 2 and 3 averaged over 3 x 3 and joined",
            "patch_features": 384, "patch_stride": 8, "backbone_parameters": parameters,
            "memory": f"{size} normal patch features per model, chosen by greedy coreset selection",
            "score": "distance of each patch to its nearest normal patch; map upsampled and smoothed (sigma "
                     f"{SIGMA} px); part score is the map maximum",
        },
        "input": f"BGR frame, long side resized to {resolution} pixels, sides multiples of 16, ImageNet normalisation",
        "outputs": {"map": "defect map at the working resolution", "score": "part score; reject when at or above the threshold"},
        "fitting": {"data": "VisA, boards PCB1 to PCB4, official one-class split",
                    "fit_images_per_board": FIT_IMAGES, "calibration_images_per_board": CALIBRATION_IMAGES,
                    "patches_per_fit_image": 512, "defect_images_used": 0},
        "threshold": {"rule": "conformal quantile of calibration scores", "target_false_reject_rate": FALSE_REJECT_TARGET},
        "test": {"data": "VisA one-class test split, 100 normal and 100 defective boards per board type",
                 "results": results["mean"], "per_board": results["products"],
                 "threshold_with_defects_for_reference": threshold["mean"]["with_defects"]},
        "latency": {**{k: round(setting[k], 1) for k in ("p50_ms", "p95_ms")}, "threads": THREADS,
                    "measured": "camera frame to map and score, ONNX Runtime on a laptop CPU"},
        "license": "MIT for the code; backbone weights from torchvision (ImageNet); VisA under CC BY 4.0",
    }
    (MODELS / "model.json").write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
