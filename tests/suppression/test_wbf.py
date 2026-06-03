import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from suppression.wbf import wbf


def _boxes(*rects) -> np.ndarray:
    return np.array(rects, dtype=float)


def _scores(*values) -> np.ndarray:
    return np.array(values, dtype=float)


def _labels(*values) -> np.ndarray:
    return np.array(values, dtype=int)


class TestWbfEmptyInput:
    def test_returns_empty_boxes(self):
        boxes, _, _ = wbf(np.empty((0, 4)), _scores(), _labels())
        assert boxes.shape == (0, 4)

    def test_returns_empty_scores(self):
        _, scores, _ = wbf(np.empty((0, 4)), _scores(), _labels())
        assert scores.shape == (0,)

    def test_returns_empty_labels(self):
        _, _, labels = wbf(np.empty((0, 4)), _scores(), _labels())
        assert labels.shape == (0,)


class TestWbfSingleBox:
    def test_box_is_preserved(self):
        box = [0.1, 0.1, 0.5, 0.5]
        boxes, scores, labels = wbf(_boxes(box), _scores(0.9), _labels(0))
        assert len(boxes) == 1
        np.testing.assert_allclose(boxes[0], box, atol=1e-4)

    def test_score_is_preserved(self):
        boxes, scores, labels = wbf(
            _boxes([0.1, 0.1, 0.5, 0.5]), _scores(0.9), _labels(0)
        )
        np.testing.assert_allclose(scores[0], 0.9, atol=1e-4)


class TestWbfHighOverlap:
    def test_two_overlapping_boxes_fused_into_one(self):
        b1 = [0.1, 0.1, 0.5, 0.5]
        b2 = [0.12, 0.12, 0.52, 0.52]
        boxes, _, _ = wbf(_boxes(b1, b2), _scores(0.9, 0.85), _labels(0, 0))
        assert len(boxes) == 1

    def test_fused_position_is_weighted_average(self):
        # IoU([0.1,0.1,0.6,0.6], [0.2,0.2,0.7,0.7]) ≈ 0.47 > iou_thr=0.45 → funde
        b1 = [0.1, 0.1, 0.6, 0.6]
        b2 = [0.2, 0.2, 0.7, 0.7]
        boxes, _, _ = wbf(_boxes(b1, b2), _scores(0.9, 0.9), _labels(0, 0))
        assert len(boxes) == 1
        fused = boxes[0]
        assert fused[0] > 0.1
        assert fused[2] < 0.7


class TestWbfLowOverlap:
    def test_non_overlapping_boxes_kept_separate(self):
        b1 = [0.0, 0.0, 0.2, 0.2]
        b2 = [0.8, 0.8, 1.0, 1.0]
        boxes, _, _ = wbf(_boxes(b1, b2), _scores(0.9, 0.85), _labels(0, 0))
        assert len(boxes) == 2

    def test_low_iou_boxes_kept_separate(self):
        b1 = [0.0, 0.0, 0.3, 0.3]
        b2 = [0.5, 0.5, 0.8, 0.8]
        boxes, _, _ = wbf(_boxes(b1, b2), _scores(0.9, 0.85), _labels(0, 0), iou_thr=0.45)
        assert len(boxes) == 2


class TestWbfDifferentClasses:
    def test_overlapping_boxes_different_classes_kept_separate(self):
        b1 = [0.1, 0.1, 0.5, 0.5]
        b2 = [0.1, 0.1, 0.5, 0.5]
        boxes, _, labels = wbf(_boxes(b1, b2), _scores(0.9, 0.85), _labels(0, 1))
        assert len(boxes) == 2
        assert set(labels.tolist()) == {0, 1}


class TestWbfSkipThreshold:
    def test_low_score_box_filtered_out(self):
        b1 = [0.1, 0.1, 0.5, 0.5]
        b2 = [0.6, 0.6, 0.9, 0.9]
        boxes, scores, _ = wbf(
            _boxes(b1, b2), _scores(0.9, 0.0005), _labels(0, 0), skip_thr=0.001
        )
        assert len(boxes) == 1
        np.testing.assert_allclose(scores[0], 0.9, atol=0.05)


class TestWbfOutputShape:
    def test_output_arrays_have_matching_lengths(self):
        boxes_in = _boxes(
            [0.0, 0.0, 0.3, 0.3],
            [0.1, 0.1, 0.4, 0.4],
            [0.7, 0.7, 1.0, 1.0],
        )
        boxes, scores, labels = wbf(boxes_in, _scores(0.9, 0.8, 0.7), _labels(0, 0, 1))
        assert len(boxes) == len(scores) == len(labels)

    def test_boxes_have_four_coordinates(self):
        boxes, _, _ = wbf(
            _boxes([0.1, 0.1, 0.5, 0.5]), _scores(0.9), _labels(0)
        )
        assert boxes.shape[1] == 4

    def test_labels_are_integers(self):
        _, _, labels = wbf(
            _boxes([0.1, 0.1, 0.5, 0.5]), _scores(0.9), _labels(2)
        )
        assert labels.dtype == int
