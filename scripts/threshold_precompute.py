"""Precompute threshold-sweep metrics offline (CPU) into a compact JSON the
dashboard loads directly — no in-browser suppression/matching needed.

Two payloads, written to results/threshold_analysis/dashboard_data.json:

  metrics: per (split, dataset, arch, threshold, fold) -> [mAP50, P, R, F1, MAE]
           fine threshold grid (0.10..0.90 step 0.02); drives curves + boxplots.

  scatter: per (dataset, arch) on the TEST split -> gt[] fixed once per image,
           plus pred[thr][] on a coarser grid (step 0.05) to keep the payload
           small. Drives the count scatter that follows the slider.

Suppression stays the per-framework step keyed by slicing mode
(SUPPRESSION_BY_MODE); nothing is re-tuned here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import pickle

from config.config_loader import ConfigLoader
from evaluation.metrics import MetricsCalculator
from geraResultados import apply_suppression
from scripts.threshold_sweep import (
    _cache_path, SUPPRESSION_BY_MODE, TARGET_DATASETS, ARCHS, IOU_THR,
)


def load_cache(split, arch):
    """Load a raw-detection cache if present. Never triggers GPU collection."""
    path = _cache_path(split, arch)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

OUT_PATH = "results/threshold_analysis/dashboard_data.json"
METRIC_THRESHOLDS = [round(0.10 + 0.02 * i, 2) for i in range(45)]   # 0.10..0.98
SCATTER_THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(19)]  # 0.10..1.00 (last clamped below)
SCATTER_THRESHOLDS = [t for t in SCATTER_THRESHOLDS if t <= 0.98]    # 0.10..0.95


def _mode_for_dataset(loader, dataset_name):
    for proc in loader.processes:
        if os.path.basename(proc.dataset.output_path.rstrip("/")) == dataset_name:
            return proc.slicing.slicing_mode
    raise KeyError(dataset_name)


def _fold_metrics_and_counts(records, thr, suppression):
    """Returns ((mAP50,P,R,F1,MAE), pred_counts_list) for one fold at one threshold."""
    calc = MetricsCalculator()
    all_gt, all_pred_boxes, all_pred_scores = [], [], []
    gt_counts, pred_counts = [], []
    for rec in records:
        keep = rec["scores"] >= thr
        boxes = rec["boxes"][keep].tolist()
        scores = rec["scores"][keep].tolist()
        labels = rec["labels"][keep].tolist()
        fin_boxes, fin_scores, _ = apply_suppression(boxes, scores, labels, suppression, IOU_THR)
        all_gt.append(rec["gt"])
        all_pred_boxes.append(fin_boxes)
        all_pred_scores.append(fin_scores)
        gt_counts.append(len(rec["gt"]))
        pred_counts.append(len(fin_boxes))

    map50, _, _ = calc.compute_map(all_gt, all_pred_boxes, all_pred_scores)
    p, r, f1 = calc.compute_prf(all_gt, all_pred_boxes, all_pred_scores)
    mae, _, _ = calc.compute_counting_metrics(gt_counts, pred_counts)
    return (round(map50, 4), round(p, 4), round(r, 4), round(f1, 4), round(mae, 3)), pred_counts


def main():
    loader = ConfigLoader("config.yaml")
    datasets = sorted(TARGET_DATASETS)

    # Load every cache up front; skip missing arch/split gracefully.
    caches = {}
    for split in ("val", "test"):
        for arch in ARCHS:
            c = load_cache(split, arch)
            if c is None:
                print(f"[warn] no cache for {split}/{arch} — run scripts/collect_raw_detections.py first")
            else:
                caches[(split, arch)] = c

    metrics = {}   # metrics[split][dataset][arch][thr_index] = [ [m,p,r,f1,mae] per fold ]
    for split in ("val", "test"):
        metrics[split] = {}
        for dataset in datasets:
            mode = _mode_for_dataset(loader, dataset)
            suppression = SUPPRESSION_BY_MODE[mode]
            metrics[split][dataset] = {}
            for arch in ARCHS:
                data = caches.get((split, arch))
                if not data:
                    continue
                folds = [f for f in range(1, 6) if (dataset, f) in data]
                if not folds:
                    continue
                per_thr = []
                for thr in METRIC_THRESHOLDS:
                    per_fold = []
                    for fold in folds:
                        vals, _ = _fold_metrics_and_counts(data[(dataset, fold)], thr, suppression)
                        per_fold.append(vals)
                    per_thr.append(per_fold)
                metrics[split][dataset][arch] = per_thr
                print(f"[metrics] {split} {dataset} {arch}: {len(folds)} folds x {len(METRIC_THRESHOLDS)} thr")

    # Scatter: TEST split only, coarser grid. Store gt once + pred per threshold.
    scatter = {}   # scatter[dataset][arch] = {gt:[...], fold:[...], img:[...], pred:{thr_str:[...]}}
    test_caches = {arch: caches.get(("test", arch)) for arch in ARCHS}
    for dataset in datasets:
        mode = _mode_for_dataset(loader, dataset)
        suppression = SUPPRESSION_BY_MODE[mode]
        scatter[dataset] = {}
        for arch in ARCHS:
            data = test_caches.get(arch)
            if not data:
                continue
            folds = [f for f in range(1, 6) if (dataset, f) in data]
            if not folds:
                continue
            gt_list, fold_list = [], []
            for fold in folds:
                for rec in data[(dataset, fold)]:
                    gt_list.append(int(len(rec["gt"])))
                    fold_list.append(fold)
            pred_by_thr = {}
            for thr in SCATTER_THRESHOLDS:
                preds = []
                for fold in folds:
                    _, pc = _fold_metrics_and_counts(data[(dataset, fold)], thr, suppression)
                    preds.extend(int(x) for x in pc)
                pred_by_thr[f"{thr:.2f}"] = preds
            scatter[dataset][arch] = {"gt": gt_list, "fold": fold_list, "pred": pred_by_thr}
            print(f"[scatter] {dataset} {arch}: {len(gt_list)} images x {len(SCATTER_THRESHOLDS)} thr")

    out = {
        "metric_thresholds": METRIC_THRESHOLDS,
        "scatter_thresholds": SCATTER_THRESHOLDS,
        "datasets": datasets,
        "archs": ARCHS,
        "suppression_by_dataset": {d: SUPPRESSION_BY_MODE[_mode_for_dataset(loader, d)] for d in datasets},
        "metrics": metrics,
        "scatter": scatter,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\nWrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
