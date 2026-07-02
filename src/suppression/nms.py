import numpy as np
import torch
from torchvision.ops import nms as _tv_nms


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    """Standard IoU-NMS using torchvision (vectorized, GPU-friendly)."""
    if len(boxes) == 0:
        return boxes, scores
    t_boxes  = torch.from_numpy(boxes.astype(np.float32))
    t_scores = torch.from_numpy(scores.astype(np.float32))
    keep = _tv_nms(t_boxes, t_scores, iou_thresh).numpy()
    return boxes[keep], scores[keep]
