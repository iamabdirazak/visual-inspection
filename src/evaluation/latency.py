"""Time per part on the CPU for every resolution and memory size, and the setting the station can afford.

The time runs from a camera frame to the defect map and score: resizing, normalising and the model.
Memory contents do not change the time, only its size, so random memories are timed on one real board.

python -m src.evaluation.latency
"""
import json
import time

import numpy as np
import onnxruntime as ort
import torch

from src.acquisition.visa import prepare, read, split
from src.export.onnx import export
from src.model.backbone import CHANNELS, Backbone
from src.model.scoring import Inspection
from src.paths import CACHE, LATENCY_BUDGET_MS, REPORTS, THREADS

RESOLUTIONS = [256, 384, 512]
MEMORY_SIZES = [500, 1000, 2000, 4000, 8000]


def session(path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def measure(model: ort.InferenceSession, frame: np.ndarray, resolution: int, runs: int = 60) -> dict:
    name = model.get_inputs()[0].name
    times = []
    for i in range(runs + 5):
        start = time.perf_counter()
        model.run(None, {name: prepare(frame, resolution)})
        if i >= 5:  # the first runs warm up the session
            times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def released(report: dict) -> dict:
    """The rule fixed before any result was seen: the highest resolution, then the largest memory, within budget."""
    fitting = [r for r in report["settings"] if r["p95_ms"] <= LATENCY_BUDGET_MS]
    return max(fitting, key=lambda r: (r["resolution"], r["memory"]))


def main():
    torch.manual_seed(0)
    backbone, frame = Backbone(), read(split("pcb1")[2][0].image)
    settings = []
    for resolution in RESOLUTIONS:
        for size in MEMORY_SIZES:
            path = CACHE / "latency.onnx"
            export(Inspection(backbone, torch.randn(size, CHANNELS)), path, resolution)
            settings.append({"resolution": resolution, "memory": size, **measure(session(path), frame, resolution)})
            print(settings[-1], flush=True)
    report = {"budget_ms": LATENCY_BUDGET_MS, "threads": THREADS, "settings": settings}
    report["released"] = released(report)
    (REPORTS / "latency.json").write_text(json.dumps(report, indent=2))
    print("released", report["released"])


if __name__ == "__main__":
    main()
