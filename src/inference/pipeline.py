from typing import Any

import numpy as np

from inference.engine import TileInferenceEngine
from inference.visualizer import draw_detections
from slicing.sahi import Sahi


def _apply_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.nms import nms

    kept_boxes, kept_scores = nms(boxes, scores, iou_thresh=iou_thr)
    # Recover labels for kept boxes by matching position
    kept_labels = _recover_labels(boxes, kept_boxes, labels)
    return kept_boxes, kept_scores, kept_labels


def _apply_bws(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.bws import bws

    kept_boxes, kept_scores = bws(boxes, scores, iou_thresh=iou_thr)
    kept_labels = _recover_labels(boxes, kept_boxes, labels)
    return kept_boxes, kept_scores, kept_labels


def _apply_nms_ioa(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.nms_ioa import nms_ioa

    kept_boxes, kept_scores = nms_ioa(boxes, scores, tau_0=iou_thr)
    kept_labels = _recover_labels(boxes, kept_boxes, labels)
    return kept_boxes, kept_scores, kept_labels


def _apply_wbf(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.wbf import wbf

    return wbf(boxes, scores, labels, iou_thr=iou_thr)


def _recover_labels(
    original_boxes: np.ndarray, kept_boxes: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Recupera labels para caixas mantidas após supressão por correspondência de posição."""
    recovered = []
    for kb in kept_boxes:
        dists = np.linalg.norm(original_boxes - kb, axis=1)
        recovered.append(labels[np.argmin(dists)])
    return np.array(recovered, dtype=int)


_SUPPRESSION_REGISTRY = {
    "nms": _apply_nms,
    "bws": _apply_bws,
    "nms_ioa": _apply_nms_ioa,
    "wbf": _apply_wbf,
}


class InferencePipeline:
    def __init__(
        self,
        engine: TileInferenceEngine,
        slicer: Sahi,
        suppression: str = "wbf",
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ):
        if suppression not in _SUPPRESSION_REGISTRY:
            raise ValueError(
                f"Supressão inválida: '{suppression}'. "
                f"Opções: {list(_SUPPRESSION_REGISTRY)}"
            )
        self.engine = engine
        self.slicer = slicer
        self.suppression = suppression
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    def run(self, image: np.ndarray, out_path: str) -> dict[str, Any]:
        raw_boxes, raw_scores, raw_labels = self.engine.predict_tiles(
            image,
            self.slicer.generate_tiles(image),
            conf_thr=self.conf_thr,
        )

        if raw_boxes:
            suppress_fn = _SUPPRESSION_REGISTRY[self.suppression]
            fin_boxes, fin_scores, fin_labels = suppress_fn(
                np.array(raw_boxes),
                np.array(raw_scores),
                np.array(raw_labels),
                self.iou_thr,
            )
            fin_boxes = fin_boxes.tolist()
            fin_scores = fin_scores.tolist()
            fin_labels = fin_labels.tolist()
        else:
            fin_boxes, fin_scores, fin_labels = [], [], []

        draw_detections(
            image,
            fin_boxes,
            fin_scores,
            fin_labels,
            self.engine.class_names,
            out_path,
            self.conf_thr,
        )

        scores_summary = (
            {
                "min": round(min(fin_scores), 3),
                "max": round(max(fin_scores), 3),
                "mean": round(float(np.mean(fin_scores)), 3),
            }
            if fin_scores
            else {}
        )

        return {
            "detections": len(fin_boxes),
            "raw_detections": len(raw_boxes),
            "duplicates_removed": len(raw_boxes) - len(fin_boxes),
            "scores": scores_summary,
        }
