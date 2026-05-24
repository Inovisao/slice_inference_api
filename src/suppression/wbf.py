import numpy as np
from ensemble_boxes import weighted_boxes_fusion


def wbf(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.45,
    skip_thr: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Boxes Fusion — funde caixas sobrepostas ponderando por score.

    Recebe e devolve coordenadas normalizadas [0, 1] no formato [x1, y1, x2, y2].
    Diferente de NMS/BWS, WBF produz caixas fundidas em vez de apenas selecionar
    a de maior score, resultando em localização mais precisa.
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes_out, scores_out, labels_out = weighted_boxes_fusion(
        [boxes.tolist()],
        [scores.tolist()],
        [labels.tolist()],
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_thr,
    )
    return np.array(boxes_out), np.array(scores_out), np.array(labels_out, dtype=int)
