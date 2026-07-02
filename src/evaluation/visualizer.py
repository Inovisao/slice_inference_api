from typing import List

import cv2
import numpy as np

from evaluation.matcher import MatchResult

# BGR colors
_COLOR_TILE    = (160, 160, 160)
_COLOR_OVERLAP = (0, 210, 255)   # yellow
_COLOR_GT      = (0, 200, 0)     # green
_COLOR_TP      = (255, 120, 0)   # blue
_COLOR_FP      = (0, 0, 220)     # red
_COLOR_FN      = (0, 150, 255)   # orange
_ALPHA_OVERLAP = 0.30


def _lw(shape) -> int:
    """Ultralytics-style adaptive line width based on image dimensions."""
    return max(round(sum(shape[:2]) / 2 * 0.003), 2)


def _to_px(box_norm: List[float], img_w: int, img_h: int) -> tuple:
    x1, y1, x2, y2 = box_norm
    return int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)


def _box_label(canvas: np.ndarray, p1: tuple, p2: tuple,
               label: str, color: tuple, lw: int,
               txt_color: tuple = (255, 255, 255)) -> None:
    """Box with filled label pill — Ultralytics style."""
    cv2.rectangle(canvas, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if not label:
        return
    tf = max(lw - 1, 1)
    sf = lw / 3
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, sf, tf)
    outside = p1[1] - th - 4 >= 0
    pill_tl = (p1[0], p1[1] - th - 4 if outside else p1[1])
    pill_br = (p1[0] + tw + 4, p1[1] if outside else p1[1] + th + 4)
    cv2.rectangle(canvas, pill_tl, pill_br, color, -1, cv2.LINE_AA)
    ty = p1[1] - 2 if outside else p1[1] + th + 2
    cv2.putText(canvas, label, (p1[0] + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, sf, txt_color, tf, cv2.LINE_AA)


def _draw_overlap_regions(canvas: np.ndarray, tile_coords: List[dict]) -> None:
    overlay = canvas.copy()
    n = len(tile_coords)
    for i in range(n):
        a = tile_coords[i]
        for j in range(i + 1, n):
            b = tile_coords[j]
            ox1 = max(a["x"], b["x"])
            oy1 = max(a["y"], b["y"])
            ox2 = min(a["x"] + a["width"],  b["x"] + b["width"])
            oy2 = min(a["y"] + a["height"], b["y"] + b["height"])
            if ox2 > ox1 and oy2 > oy1:
                cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), _COLOR_OVERLAP, -1)
    cv2.addWeighted(overlay, _ALPHA_OVERLAP, canvas, 1 - _ALPHA_OVERLAP, 0, canvas)


def _draw_tile_borders(canvas: np.ndarray, tile_coords: List[dict], lw: int) -> None:
    for c in tile_coords:
        x1, y1 = c["x"], c["y"]
        x2, y2 = x1 + c["width"], y1 + c["height"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_TILE,
                      max(1, lw // 2), cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, n_gt: int, n_tp: int,
              n_fp: int, n_fn: int, lw: int) -> None:
    sf = lw / 3
    tf = max(lw - 1, 1)
    text = f"GT:{n_gt}  TP:{n_tp}  FP:{n_fp}  FN:{n_fn}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sf, tf)
    pad = max(6, lw * 2)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, text, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, sf, (255, 255, 255), tf, cv2.LINE_AA)


def _draw_legend(canvas: np.ndarray, lw: int) -> None:
    sf = lw / 3
    tf = max(lw - 1, 1)
    items = [
        ("GT  anotacao real",  _COLOR_GT),
        ("TP  deteccao certa", _COLOR_TP),
        ("FP  falso alarme",   _COLOR_FP),
        ("FN  objeto perdido", _COLOR_FN),
    ]
    (_, th), _ = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, sf, tf)
    h = canvas.shape[0]
    pad = max(6, lw * 2)
    row_h = th + pad
    swatch_w = max(6, lw * 3)
    max_tw = max(cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, sf, tf)[0][0]
                 for lbl, _ in items)
    box_w = pad + swatch_w + pad + max_tw + pad
    total_h = row_h * len(items) + pad
    y0 = h - total_h - pad

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (box_w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    for i, (lbl, color) in enumerate(items):
        y = y0 + pad + i * row_h + th
        cv2.rectangle(canvas,
                      (pad, y - th + 2), (pad + swatch_w, y + 2), color, -1)
        cv2.putText(canvas, lbl, (pad + swatch_w + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sf, color, tf, cv2.LINE_AA)


def draw_eval_image(
    image: np.ndarray,
    tile_coords: List[dict],
    gt_boxes: np.ndarray,
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    match: MatchResult,
) -> np.ndarray:
    canvas = image.copy()
    img_h, img_w = canvas.shape[:2]
    lw = _lw(canvas.shape)

    _draw_overlap_regions(canvas, tile_coords)
    _draw_tile_borders(canvas, tile_coords, lw)

    # All GT boxes — thin green border, no label
    for box in gt_boxes.tolist():
        x1, y1, x2, y2 = _to_px(box, img_w, img_h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_GT,
                      max(1, lw // 2), cv2.LINE_AA)

    # TP predictions — blue pill with score
    for i in match.tp_pred_indices:
        x1, y1, x2, y2 = _to_px(pred_boxes[i], img_w, img_h)
        _box_label(canvas, (x1, y1), (x2, y2),
                   f"TP {pred_scores[i]:.2f}", _COLOR_TP, lw)

    # FP predictions — red pill with score
    for i in match.fp_pred_indices:
        x1, y1, x2, y2 = _to_px(pred_boxes[i], img_w, img_h)
        _box_label(canvas, (x1, y1), (x2, y2),
                   f"FP {pred_scores[i]:.2f}", _COLOR_FP, lw)

    # FN GT boxes — orange pill (drawn over the green GT border)
    for i in match.fn_gt_indices:
        x1, y1, x2, y2 = _to_px(gt_boxes[i].tolist(), img_w, img_h)
        _box_label(canvas, (x1, y1), (x2, y2), "FN", _COLOR_FN, lw)

    n_gt = len(gt_boxes)
    n_tp = len(match.tp_pred_indices)
    n_fp = len(match.fp_pred_indices)
    n_fn = len(match.fn_gt_indices)

    _draw_hud(canvas, n_gt, n_tp, n_fp, n_fn, lw)
    _draw_legend(canvas, lw)

    return canvas
