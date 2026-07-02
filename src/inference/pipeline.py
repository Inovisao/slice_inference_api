from typing import Any, Protocol, runtime_checkable

import numpy as np

from inference.visualizer import draw_detections


@runtime_checkable
class Slicer(Protocol):
    def generate_tiles(self, image): ...


@runtime_checkable
class InferenceEngine(Protocol):
    class_names: dict

    def predict_tiles(self, image, tile_generator, conf_thr=0.25, batch_size=32): ...

    def predict_full_image(self, image, conf_thr=0.25): ...


def _apply_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.nms import nms

    return _apply_per_class(
        boxes, scores, labels, lambda b, s: nms(b, s, iou_thresh=iou_thr)
    )


def _apply_bws(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.bws import bws

    return _apply_per_class(
        boxes, scores, labels, lambda b, s: bws(b, s, iou_thresh=iou_thr)
    )


def _apply_nms_ioa(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.nms_ioa import nms_ioa

    return _apply_per_class(
        boxes, scores, labels, lambda b, s: nms_ioa(b, s, tau_0=iou_thr)
    )


def _apply_per_class(boxes, scores, labels, suppress):
    """Applies a class-agnostic suppression function independently per class."""
    out_boxes, out_scores, out_labels = [], [], []
    for label in np.unique(labels):
        mask = labels == label
        class_boxes = boxes[mask]
        kept_boxes, kept_scores = suppress(class_boxes, scores[mask])
        out_boxes.extend(kept_boxes)
        out_scores.extend(kept_scores)
        out_labels.extend([int(label)] * len(kept_boxes))
    if not out_boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    order = np.argsort(np.asarray(out_scores))[::-1]
    return (
        np.asarray(out_boxes)[order],
        np.asarray(out_scores)[order],
        np.asarray(out_labels, dtype=int)[order],
    )


def _apply_wbf(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.wbf import wbf

    return wbf(boxes, scores, labels, iou_thr=iou_thr)


def _apply_cluster_diou_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from suppression.cluster_diou_nms import cluster_diou_nms

    return cluster_diou_nms(boxes, scores, labels, iou_thr=iou_thr)


_SUPPRESSION_REGISTRY = {
    "nms": _apply_nms,
    "bws": _apply_bws,
    "nms_ioa": _apply_nms_ioa,
    "wbf": _apply_wbf,
    "cluster_diou_nms": _apply_cluster_diou_nms,
}


class InferencePipeline:
    def __init__(
        self,
        engine: InferenceEngine,
        slicer: Slicer,
        suppression: str = "wbf",
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
        include_full_inference: bool = False,
        batch_size: int = 32,
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
        self.include_full_inference = include_full_inference
        self.batch_size = batch_size

    def run(self, image: np.ndarray, out_path: str) -> dict[str, Any]:
        raw_boxes, raw_scores, raw_labels = self.engine.predict_tiles(
            image,
            self.slicer.generate_tiles(image),
            conf_thr=self.conf_thr,
            batch_size=self.batch_size,
        )

        if self.include_full_inference:
            fi_boxes, fi_scores, fi_labels = self.engine.predict_full_image(
                image, conf_thr=self.conf_thr
            )
            raw_boxes += fi_boxes
            raw_scores += fi_scores
            raw_labels += fi_labels

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
