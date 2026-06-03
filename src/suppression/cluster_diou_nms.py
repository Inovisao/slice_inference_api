import numpy as np


def cluster_diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    order = np.argsort(-scores)
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    iou = inter / (areas[:, None] + areas[None, :] - inter + 1e-7)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rho2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
    c2 = (
        (np.maximum(x2[:, None], x2[None, :]) - np.minimum(x1[:, None], x1[None, :])) ** 2
        + (np.maximum(y2[:, None], y2[None, :]) - np.minimum(y1[:, None], y1[None, :])) ** 2
        + 1e-7
    )

    # Eq. 11: DIoU = IoU - ρ²(center, center_gt) / c²
    X = np.triu(iou - rho2 / c2, k=1)

    # Cluster-NMS: X_cluster = b × X, iterate until convergence
    b = np.ones(len(boxes), dtype=bool)
    for _ in range(len(boxes)):
        b_prev = b.copy()
        col_max = (X * b[:, None]).max(axis=0)
        b = col_max <= iou_thr
        b[0] = True
        if np.array_equal(b, b_prev):
            break

    return boxes[b], scores[b], labels[b]
