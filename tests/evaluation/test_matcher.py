import numpy as np
import pytest

from evaluation.matcher import DetectionMatcher, MatchResult


def _gt(*boxes):
    return np.array(boxes, dtype=np.float32)


def _preds(*boxes):
    return [list(b) for b in boxes]


def _scores(*values):
    return list(values)


@pytest.fixture
def matcher():
    return DetectionMatcher(iou_threshold=0.5)


class TestEmptyInputs:
    def test_no_gt_no_pred_returns_empty(self, matcher):
        r = matcher.match(_gt(), _preds(), _scores())
        assert r.tp_pred_indices == []
        assert r.fp_pred_indices == []
        assert r.fn_gt_indices == []

    def test_no_pred_all_gt_are_fn(self, matcher):
        gt = _gt([0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0])
        r = matcher.match(gt, _preds(), _scores())
        assert r.fn_gt_indices == [0, 1]
        assert r.tp_pred_indices == []
        assert r.fp_pred_indices == []

    def test_no_gt_all_preds_are_fp(self, matcher):
        r = matcher.match(_gt(), _preds([0.0, 0.0, 0.5, 0.5]), _scores(0.9))
        assert r.fp_pred_indices == [0]
        assert r.tp_pred_indices == []
        assert r.fn_gt_indices == []


class TestPerfectMatch:
    def test_identical_boxes_all_tp(self, matcher):
        box = [0.1, 0.1, 0.5, 0.5]
        gt = _gt(box)
        r = matcher.match(gt, _preds(box), _scores(0.9))
        assert r.tp_pred_indices == [0]
        assert r.fp_pred_indices == []
        assert r.fn_gt_indices == []

    def test_multiple_perfect_matches_all_tp(self, matcher):
        boxes = [[0.0, 0.0, 0.3, 0.3], [0.7, 0.7, 1.0, 1.0]]
        gt = _gt(*boxes)
        r = matcher.match(gt, _preds(*boxes), _scores(0.9, 0.8))
        assert len(r.tp_pred_indices) == 2
        assert r.fp_pred_indices == []
        assert r.fn_gt_indices == []


class TestNoOverlap:
    def test_disjoint_boxes_all_fp_and_fn(self, matcher):
        gt = _gt([0.0, 0.0, 0.2, 0.2])
        pred = _preds([0.8, 0.8, 1.0, 1.0])
        r = matcher.match(gt, pred, _scores(0.9))
        assert r.fp_pred_indices == [0]
        assert r.fn_gt_indices == [0]
        assert r.tp_pred_indices == []


class TestIoUThreshold:
    def test_iou_above_threshold_is_tp(self):
        # IoU ≈ 0.69 (strong overlap)
        matcher = DetectionMatcher(iou_threshold=0.5)
        gt = _gt([0.0, 0.0, 0.6, 0.6])
        pred = _preds([0.1, 0.1, 0.7, 0.7])
        r = matcher.match(gt, pred, _scores(0.9))
        assert r.tp_pred_indices == [0]

    def test_iou_below_threshold_is_fp(self):
        # IoU ≈ 0.14 (small overlap)
        matcher = DetectionMatcher(iou_threshold=0.5)
        gt = _gt([0.0, 0.0, 0.5, 0.5])
        pred = _preds([0.4, 0.4, 0.9, 0.9])
        r = matcher.match(gt, pred, _scores(0.9))
        assert r.fp_pred_indices == [0]
        assert r.fn_gt_indices == [0]


class TestGreedyPriority:
    def test_higher_score_pred_wins_the_gt(self, matcher):
        # Two preds overlap the same GT — only the higher-score one gets TP
        gt = _gt([0.1, 0.1, 0.6, 0.6])
        pred = _preds([0.1, 0.1, 0.6, 0.6], [0.1, 0.1, 0.6, 0.6])
        r = matcher.match(gt, pred, _scores(0.5, 0.9))
        assert len(r.tp_pred_indices) == 1
        assert len(r.fp_pred_indices) == 1
        # pred index 1 (score=0.9) should be TP
        assert 1 in r.tp_pred_indices

    def test_each_gt_matched_at_most_once(self, matcher):
        gt = _gt([0.1, 0.1, 0.5, 0.5])
        pred = _preds(
            [0.1, 0.1, 0.5, 0.5],
            [0.1, 0.1, 0.5, 0.5],
            [0.1, 0.1, 0.5, 0.5],
        )
        r = matcher.match(gt, pred, _scores(0.9, 0.8, 0.7))
        assert len(r.tp_pred_indices) == 1
        assert len(r.fp_pred_indices) == 2

    def test_total_count_equals_n_preds(self, matcher):
        gt = _gt([0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0])
        pred = _preds(
            [0.0, 0.0, 0.4, 0.4],
            [0.6, 0.6, 1.0, 1.0],
            [0.3, 0.3, 0.7, 0.7],
        )
        r = matcher.match(gt, pred, _scores(0.9, 0.85, 0.4))
        assert len(r.tp_pred_indices) + len(r.fp_pred_indices) == len(pred)
