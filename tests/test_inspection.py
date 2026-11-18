import numpy as np
import onnxruntime as ort
import torch

from src.calibration.threshold import from_normals, oracle
from src.evaluation.metrics import aupro, operating_point
from src.export.onnx import export
from src.model.backbone import Backbone
from src.model.coreset import greedy
from src.model.scoring import Inspection, nearest


def test_nearest_matches_direct_distances():
    a, b = torch.randn(7, 5), torch.randn(4, 5)
    assert torch.allclose(nearest(a, b), torch.cdist(a, b) ** 2, atol=1e-4)


def test_coreset_keeps_an_outlier_and_is_nested():
    points = torch.cat([torch.randn(200, 8) * 0.01, torch.full((1, 8), 5.0)])
    order = greedy(points, 10)
    assert 200 in order[:3].tolist()
    assert torch.equal(greedy(points, 5), order[:5])


def test_threshold_bounds_false_rejects_on_average_over_calibration_sets():
    rng = np.random.default_rng(0)
    rates = [(rng.normal(size=10_000) >= from_normals(rng.normal(size=200), 0.01)).mean() for _ in range(300)]
    assert np.mean(rates) <= 0.012


def test_oracle_separates_clean_scores():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    defective = np.array([False, False, False, True, True])
    assert operating_point(scores, defective, oracle(scores, defective))["escape_rate"] == 0


def test_aupro_is_one_for_a_perfect_map():
    mask = np.zeros((32, 32), bool)
    mask[4:8, 4:8] = True
    assert aupro([mask.astype(float), np.zeros((32, 32))], [mask, None]) > 0.99


def test_a_patch_unlike_the_memory_scores_highest():
    backbone = Backbone(pretrained=False)
    image = torch.zeros(1, 3, 128, 128)
    with torch.inference_mode():
        memory = backbone(image)[0].flatten(1).T
        defect = image.clone()
        defect[..., 60:76, 60:76] = 3.0
        defect_map, score = Inspection(backbone, memory)(defect)
    assert score.item() > 0
    y, x = np.unravel_index(defect_map[0, 0].argmax().item(), (128, 128))
    assert 44 <= y <= 92 and 44 <= x <= 92


def test_onnx_matches_torch_at_another_size(tmp_path):
    torch.manual_seed(0)
    model = Inspection(Backbone(pretrained=False), torch.randn(64, 384))
    export(model, tmp_path / "m.onnx", 128)
    image = torch.randn(1, 3, 96, 160)
    with torch.inference_mode():
        expected_map, expected_score = model(image)
    got_map, got_score = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"image": image.numpy()})
    assert got_map.shape == (1, 1, 96, 160)
    assert np.allclose(got_score, expected_score.numpy(), rtol=1e-3, atol=1e-3)
