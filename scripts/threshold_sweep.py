"""Confidence-threshold analysis with a leakage-free protocol:

  1. Collect raw (pre-suppression) YOLO detections at a low confidence floor
     on both `val` and `test` splits, once per mode/fold.
  2. Sweep thresholds on `val` only -> per-mode operating curve, saved to CSV
     for plotting/reporting.
  3. Pick the best-F1 threshold per mode from the `val` curve.
  4. Apply that FIXED threshold to `test` (never re-tuned on test) to report
     the final per-fold operating-point metrics (P/R/F1/MAE) with std.
  5. Report mAP separately, computed on the full `test` detections (conf>=floor,
     not further threshold-gated) — mAP is a full precision-recall-curve
     metric and should not be truncated by a hard confidence cutoff.

Filtering must happen BEFORE suppression, not after — suppression depends on
which boxes are present, so re-filtering an already-suppressed set does not
reproduce what running inference with a higher conf_threshold would have
produced.

Suppression method is kept as configured per mode (nms for sahi,
cluster_diou_nms for asahi/asahi_rect) — it is part of each framework's
pipeline, not an independent variable to ablate here.
"""

import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm

from config.config_loader import ConfigLoader
from evaluation.loader import FoldTestLoader
from evaluation.metrics import MetricsCalculator
from inference.engine import make_engine
from slicing.service import make_slicer
from geraResultados import infer_image, apply_suppression, resolve_checkpoint

RAW_CACHE_DIR = "results/threshold_analysis/raw"
RESULTS_DIR = "results/threshold_analysis"
COLLECT_CONF = 0.1
# dataset folder names to evaluate; each maps to a slicing mode via config.yaml
TARGET_DATASETS = {"sahi", "asahi", "asahi_rect", "all_640"}
ARCHS = ["YOLO", "Faster", "Detr"]
# Suppression is a per-framework pipeline step, keyed by the dataset's slicing mode.
SUPPRESSION_BY_MODE = {
    "sahi": "nms",
    "asahi": "cluster_diou_nms",
    "asahi_rect": "cluster_diou_nms",
    "none": "nms",  # all_640 baseline: no tiling
}
# legacy alias kept for the YOLO-only sweep path below
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
IOU_THR = 0.5


def _cache_path(split: str, arch: str = "YOLO") -> str:
    """Per-architecture cache. YOLO keeps the legacy filename for back-compat."""
    if arch == "YOLO":
        return os.path.join(RAW_CACHE_DIR, f"threshold_sweep_raw_{split}.pkl")
    return os.path.join(RAW_CACHE_DIR, f"threshold_sweep_raw_{split}_{arch}.pkl")


def collect(split: str, arch: str = "YOLO"):
    """Collect raw pre-suppression detections for one architecture.

    split: 'val' or 'test'. Returns {(dataset_name, fold): [ {gt, boxes, scores, labels}, ... ]}.
    Keys use the dataset folder name (sahi/asahi/asahi_rect/all_640), not the slicing mode,
    so all_640 (mode 'none') is addressable distinctly.
    """
    loader = ConfigLoader("config.yaml")
    paths = loader.paths
    data = {}

    for proc in loader.processes:
        dataset_name = os.path.basename(proc.dataset.output_path.rstrip("/"))
        if dataset_name not in TARGET_DATASETS:
            continue
        mode = proc.slicing.slicing_mode

        for fold in range(1, proc.crossfolds.n_folds + 1):
            try:
                weights = resolve_checkpoint(paths.models, dataset_name, fold, arch)
            except FileNotFoundError as exc:
                print(f"[skip:{split}:{arch}] {dataset_name} fold_{fold}: {exc}")
                continue
            split_dir = os.path.join(proc.dataset.output_path, f"fold_{fold}", split)

            print(f"[collect:{split}:{arch}] {dataset_name} fold_{fold}: {weights}")
            engine = make_engine(arch, weights, device="cuda")
            slicer = make_slicer(mode, proc.slicing.overlap_ratio)
            fold_loader = FoldTestLoader(split_dir)
            images = fold_loader.list_images()

            records = []
            for image_name in tqdm(images, desc=f"{dataset_name} fold_{fold} [{arch}]", ncols=80):
                image, _ = fold_loader.load_image(image_name)
                gt_boxes = fold_loader.load_gt_boxes(image_name)
                _, raw_boxes, raw_scores, raw_labels, _ = infer_image(
                    image, slicer, engine, conf_thr=COLLECT_CONF, batch_size=32
                )
                records.append({
                    "gt": gt_boxes,
                    "boxes": np.array(raw_boxes) if raw_boxes else np.zeros((0, 4)),
                    "scores": np.array(raw_scores) if raw_scores else np.zeros((0,)),
                    "labels": np.array(raw_labels) if raw_labels else np.zeros((0,), dtype=int),
                })
            data[(dataset_name, fold)] = records
            del engine

    path = _cache_path(split, arch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved raw {split} [{arch}] detections (conf>={COLLECT_CONF}) to {path}")
    return data


def load_or_collect(split: str, arch: str = "YOLO"):
    path = _cache_path(split, arch)
    if os.path.exists(path):
        print(f"Loading cached {split} [{arch}] detections from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return collect(split, arch)


def _fold_metrics(records, thr, suppression, iou_thr=IOU_THR):
    """Returns (mAP50, precision, recall, f1, mae) for one fold at one threshold."""
    calculator = MetricsCalculator()
    all_gt, all_pred_boxes, all_pred_scores = [], [], []
    gt_counts, pred_counts = [], []
    for rec in records:
        keep = rec["scores"] >= thr
        boxes = rec["boxes"][keep].tolist()
        scores = rec["scores"][keep].tolist()
        labels = rec["labels"][keep].tolist()
        fin_boxes, fin_scores, _ = apply_suppression(boxes, scores, labels, suppression, iou_thr)
        all_gt.append(rec["gt"])
        all_pred_boxes.append(fin_boxes)
        all_pred_scores.append(fin_scores)
        gt_counts.append(len(rec["gt"]))
        pred_counts.append(len(fin_boxes))

    map50, _, _ = calculator.compute_map(all_gt, all_pred_boxes, all_pred_scores)
    p, r, f1 = calculator.compute_prf(all_gt, all_pred_boxes, all_pred_scores)
    mae, _, _ = calculator.compute_counting_metrics(gt_counts, pred_counts)
    return map50, p, r, f1, mae


def sweep_curve(data, mode):
    """Per-threshold, mean-across-folds curve for one mode. Returns list of dict rows."""
    suppression = SUPPRESSION_BY_MODE[mode]
    rows = []
    for thr in THRESHOLDS:
        per_fold = [
            _fold_metrics(data[(mode, fold)], thr, suppression)
            for fold in range(1, 6) if (mode, fold) in data
        ]
        map50, p, r, f1, mae = (np.mean(x) for x in zip(*per_fold))
        rows.append({"mode": mode, "threshold": thr, "mAP50": map50,
                      "precision": p, "recall": r, "fscore": f1, "MAE": mae})
    return rows


def select_best_threshold(val_curve_rows):
    best = max(val_curve_rows, key=lambda r: r["fscore"])
    return best["threshold"]


def evaluate_test_at_threshold(test_data, mode, thr):
    """Per-fold rows at the val-selected fixed threshold — for mean±std reporting."""
    suppression = SUPPRESSION_BY_MODE[mode]
    rows = []
    for fold in range(1, 6):
        if (mode, fold) not in test_data:
            continue
        map50, p, r, f1, mae = _fold_metrics(test_data[(mode, fold)], thr, suppression)
        rows.append({"mode": mode, "fold": fold, "threshold": thr,
                      "mAP50": map50, "precision": p, "recall": r, "fscore": f1, "MAE": mae})
    return rows


def main():
    val_data = load_or_collect("val")
    test_data = load_or_collect("test")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) val sweep curve -> CSV for plotting
    curve_path = os.path.join(RESULTS_DIR, "val_threshold_curve.csv")
    all_curve_rows = []
    for mode in sorted(TARGET_MODES):
        all_curve_rows += sweep_curve(val_data, mode)
    with open(curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "threshold", "mAP50", "precision", "recall", "fscore", "MAE"])
        writer.writeheader()
        writer.writerows(all_curve_rows)
    print(f"\nSaved val threshold curve to {curve_path}")

    # 2) pick best threshold per mode from val
    best_thr = {}
    for mode in sorted(TARGET_MODES):
        mode_rows = [r for r in all_curve_rows if r["mode"] == mode]
        best_thr[mode] = select_best_threshold(mode_rows)
        print(f"  {mode}: selected threshold = {best_thr[mode]:.2f} (val F1={max(r['fscore'] for r in mode_rows):.3f})")

    # 3) apply fixed threshold to test -> per-fold rows for mean±std
    test_path = os.path.join(RESULTS_DIR, "test_at_val_threshold.csv")
    all_test_rows = []
    for mode in sorted(TARGET_MODES):
        all_test_rows += evaluate_test_at_threshold(test_data, mode, best_thr[mode])
    with open(test_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "fold", "threshold", "mAP50", "precision", "recall", "fscore", "MAE"])
        writer.writeheader()
        writer.writerows(all_test_rows)
    print(f"Saved test-at-val-threshold per-fold results to {test_path}")

    print("\n=== Final: val-tuned threshold, evaluated on held-out test ===")
    for mode in sorted(TARGET_MODES):
        rows = [r for r in all_test_rows if r["mode"] == mode]
        for field in ("mAP50", "precision", "recall", "fscore", "MAE"):
            vals = [r[field] for r in rows]
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1)
            print(f"  {mode:12} thr={best_thr[mode]:.2f}  {field:10} {mean:.3f} ± {sd:.3f}")
        print()


if __name__ == "__main__":
    main()
