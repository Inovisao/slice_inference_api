from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr
from torchvision.ops import box_iou


def _compute_ap(
    gt_list: List[np.ndarray],
    pred_boxes_list: List[List],
    pred_scores_list: List[List],
    iou_threshold: float,
) -> float:
    total_gt = sum(len(gt) for gt in gt_list)
    if total_gt == 0:
        return 0.0

    all_scores: List[float] = []
    all_tp: List[int] = []

    for gt_boxes, pred_boxes, pred_scores in zip(gt_list, pred_boxes_list, pred_scores_list):
        if len(pred_boxes) == 0:
            continue

        pred_arr = np.array(pred_boxes, dtype=np.float32)
        scores_arr = np.array(pred_scores, dtype=np.float32)

        if len(gt_boxes) == 0:
            all_scores.extend(scores_arr.tolist())
            all_tp.extend([0] * len(pred_boxes))
            continue

        iou_mat = box_iou(
            torch.tensor(pred_arr),
            torch.tensor(gt_boxes.astype(np.float32)),
        ).numpy()

        matched_gt: set = set()
        order = np.argsort(-scores_arr)

        for pred_idx in order:
            best_gt = int(np.argmax(iou_mat[pred_idx]))
            if iou_mat[pred_idx, best_gt] >= iou_threshold and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                all_tp.append(1)
            else:
                all_tp.append(0)
            all_scores.append(float(scores_arr[pred_idx]))

    if not all_scores:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp)[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)
    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    # 101-point COCO interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        ap += float(np.max(precisions[mask])) if mask.any() else 0.0
    return ap / 101


class MetricsCalculator:
    """Computes mAP, precision, recall, F1, MAE, RMSE, Pearson r."""

    _MAP_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

    def compute_map(
        self,
        gt_list: List[np.ndarray],
        pred_boxes_list: List[List],
        pred_scores_list: List[List],
    ) -> Tuple[float, float, float]:
        """Returns (mAP50, mAP75, mAP[.5:.95])."""
        aps = [
            _compute_ap(gt_list, pred_boxes_list, pred_scores_list, thr)
            for thr in self._MAP_THRESHOLDS
        ]
        return aps[0], aps[5], float(np.mean(aps))

    def compute_prf(
        self,
        gt_list: List[np.ndarray],
        pred_boxes_list: List[List],
        pred_scores_list: List[List],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        total_tp = total_fp = total_fn = 0

        for gt_boxes, pred_boxes, pred_scores in zip(gt_list, pred_boxes_list, pred_scores_list):
            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)

            if n_gt == 0 and n_pred == 0:
                continue

            if n_pred == 0:
                total_fn += n_gt
                continue

            if n_gt == 0:
                total_fp += n_pred
                continue

            pred_arr = np.array(pred_boxes, dtype=np.float32)
            scores_arr = np.array(pred_scores, dtype=np.float32)
            iou_mat = box_iou(
                torch.tensor(pred_arr),
                torch.tensor(gt_boxes.astype(np.float32)),
            ).numpy()

            matched_gt: set = set()
            order = np.argsort(-scores_arr)
            tp = 0

            for pred_idx in order:
                best_gt = int(np.argmax(iou_mat[pred_idx]))
                if iou_mat[pred_idx, best_gt] >= iou_threshold and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    tp += 1

            fp = n_pred - tp
            fn = n_gt - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        fscore = 2 * precision * recall / (precision + recall + 1e-9)
        return round(precision, 6), round(recall, 6), round(fscore, 6)

    def compute_counting_metrics(
        self,
        gt_counts: List[int],
        pred_counts: List[int],
    ) -> Tuple[float, float, float]:
        gt = np.array(gt_counts, dtype=np.float64)
        pred = np.array(pred_counts, dtype=np.float64)
        mae = float(np.mean(np.abs(gt - pred)))
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        r = float(pearsonr(gt, pred).statistic) if len(gt) > 1 and np.std(gt) > 0 else 0.0
        return round(mae, 6), round(rmse, 6), round(r, 6)
