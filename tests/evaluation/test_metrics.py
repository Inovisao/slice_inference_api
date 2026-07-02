import numpy as np
import pytest

from evaluation.metrics import MetricsCalculator


def _gt(*boxes):
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


@pytest.fixture
def calc():
    return MetricsCalculator()


class TestMapPerfectPredictions:
    def test_perfect_predictions_give_map50_one(self, calc):
        gt = _gt([0.0, 0.0, 0.5, 0.5])
        map50, _, _ = calc.compute_map([gt], [[gt[0].tolist()]], [[0.9]])
        assert abs(map50 - 1.0) < 1e-4

    def test_perfect_predictions_give_map_one(self, calc):
        gt = _gt([0.0, 0.0, 0.5, 0.5])
        _, _, map_all = calc.compute_map([gt], [[gt[0].tolist()]], [[0.9]])
        assert abs(map_all - 1.0) < 1e-4


class TestMapEmptyInputs:
    def test_no_predictions_gives_zero(self, calc):
        gt = _gt([0.0, 0.0, 0.5, 0.5])
        map50, map75, map_all = calc.compute_map([gt], [[]], [[]])
        assert map50 == 0.0
        assert map75 == 0.0
        assert map_all == 0.0

    def test_no_gt_gives_zero(self, calc):
        map50, _, _ = calc.compute_map(
            [_gt()], [[[0.0, 0.0, 0.5, 0.5]]], [[0.9]]
        )
        assert map50 == 0.0

    def test_all_wrong_predictions_gives_zero_map50(self, calc):
        gt = _gt([0.0, 0.0, 0.2, 0.2])
        pred = [[0.8, 0.8, 1.0, 1.0]]
        map50, _, _ = calc.compute_map([gt], [pred], [[0.9]])
        assert map50 == 0.0


class TestPRF:
    def test_perfect_predictions_precision_one(self, calc):
        gt = _gt([0.1, 0.1, 0.5, 0.5])
        p, r, f = calc.compute_prf([gt], [[gt[0].tolist()]], [[0.9]])
        assert abs(p - 1.0) < 1e-4

    def test_perfect_predictions_recall_one(self, calc):
        gt = _gt([0.1, 0.1, 0.5, 0.5])
        p, r, f = calc.compute_prf([gt], [[gt[0].tolist()]], [[0.9]])
        assert abs(r - 1.0) < 1e-4

    def test_all_fp_precision_near_zero(self, calc):
        gt = _gt()
        pred = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
        p, r, f = calc.compute_prf([gt], [pred], [[0.9, 0.8]])
        assert p < 1e-4

    def test_all_fn_recall_near_zero(self, calc):
        gt = _gt([0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0])
        p, r, f = calc.compute_prf([gt], [[]], [[]])
        assert r < 1e-4

    def test_fscore_is_harmonic_mean(self, calc):
        gt = _gt([0.1, 0.1, 0.5, 0.5])
        p, r, f = calc.compute_prf([gt], [[gt[0].tolist()]], [[0.9]])
        expected_f = 2 * p * r / (p + r + 1e-9)
        assert abs(f - expected_f) < 1e-4


class TestCountingMetrics:
    def test_perfect_count_mae_zero(self, calc):
        mae, _, _ = calc.compute_counting_metrics([5, 3, 7], [5, 3, 7])
        assert mae == 0.0

    def test_perfect_count_rmse_zero(self, calc):
        _, rmse, _ = calc.compute_counting_metrics([4, 2, 6], [4, 2, 6])
        assert rmse == 0.0

    def test_perfect_count_pearson_one(self, calc):
        _, _, r = calc.compute_counting_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert abs(r - 1.0) < 1e-6

    def test_inverse_count_pearson_minus_one(self, calc):
        _, _, r = calc.compute_counting_metrics([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(r + 1.0) < 1e-6

    def test_mae_is_mean_absolute_error(self, calc):
        mae, _, _ = calc.compute_counting_metrics([10, 0], [8, 3])
        assert abs(mae - 2.5) < 1e-6

    def test_rmse_penalizes_large_errors(self, calc):
        _, rmse_small, _ = calc.compute_counting_metrics([5, 5], [4, 6])
        _, rmse_large, _ = calc.compute_counting_metrics([5, 5], [0, 10])
        assert rmse_large > rmse_small

    def test_constant_gt_returns_r_zero(self, calc):
        # Pearson r is undefined when std=0 — must not raise, returns 0
        _, _, r = calc.compute_counting_metrics([3, 3, 3], [1, 2, 3])
        assert r == 0.0
