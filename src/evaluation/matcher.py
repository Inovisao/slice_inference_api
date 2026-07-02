from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torchvision.ops import box_iou


@dataclass
class MatchResult:
    tp_pred_indices: List[int] = field(default_factory=list)
    fp_pred_indices: List[int] = field(default_factory=list)
    fn_gt_indices: List[int] = field(default_factory=list)


class DetectionMatcher:
    """Greedy IoU matching between GT boxes and predictions, sorted by confidence."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def match(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: List[List[float]],
        pred_scores: List[float],
    ) -> MatchResult:
        result = MatchResult()

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return result

        if len(pred_boxes) == 0:
            result.fn_gt_indices = list(range(len(gt_boxes)))
            return result

        if len(gt_boxes) == 0:
            result.fp_pred_indices = list(range(len(pred_boxes)))
            return result

        pred_arr = np.array(pred_boxes, dtype=np.float32)
        scores_arr = np.array(pred_scores, dtype=np.float32)

        iou_matrix = box_iou(
            torch.tensor(pred_arr),
            torch.tensor(gt_boxes),
        ).numpy()  # [N_pred, N_gt]

        matched_gt: set = set()
        order = np.argsort(-scores_arr)

        for pred_idx in order:
            if len(matched_gt) == len(gt_boxes):
                result.fp_pred_indices.append(int(pred_idx))
                continue

            best_gt = int(np.argmax(iou_matrix[pred_idx]))
            if iou_matrix[pred_idx, best_gt] >= self.iou_threshold and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                result.tp_pred_indices.append(int(pred_idx))
            else:
                result.fp_pred_indices.append(int(pred_idx))

        result.fn_gt_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        return result
