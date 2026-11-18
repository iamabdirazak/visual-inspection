"""The whole inspection model, memory included, as one ONNX graph with a free input size."""
from pathlib import Path

import torch

from src.model.scoring import Inspection


def export(model: Inspection, path: Path, resolution: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, resolution * 3 // 4 // 16 * 16, resolution)
    torch.onnx.export(model.eval(), (example,), str(path), input_names=["image"], output_names=["map", "score"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "map": {2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)
