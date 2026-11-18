"""The inspection station: a camera frame in, a decision and a defect map out."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.acquisition.visa import prepare
from src.paths import MODELS


@dataclass
class Verdict:
    defective: bool
    score: float
    map: np.ndarray  # defect map at the frame's size; values at or above the threshold mark the defect


class Inspector:
    def __init__(self, product: str, card: Path = MODELS / "model.json", threads: int = 4):
        card_data = json.loads(Path(card).read_text())
        entry = next(m for m in card_data["models"] if product in m["products"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(Path(card).parent / entry["file"]), options,
                                            providers=["CPUExecutionProvider"])
        self.threshold = entry["threshold"]
        self.resolution = int(card_data["input"].split("to ")[1].split(" ")[0])

    def __call__(self, frame: np.ndarray) -> Verdict:
        defect_map, score = self.session.run(None, {"image": prepare(frame, self.resolution)})
        h, w = frame.shape[:2]
        full = cv2.resize(defect_map[0, 0], (w, h), interpolation=cv2.INTER_LINEAR)
        return Verdict(bool(score[0] >= self.threshold), float(score[0]), full)
