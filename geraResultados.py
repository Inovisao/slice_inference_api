"""
Evaluation pipeline for slicing-based inference cross-fold experiments.

For each process × architecture × fold defined in config.yaml:
  1. Loads original test images and GT labels (YOLO format)
  2. Slices each image with SAHI or ASAHI (params from config)
  3. Runs inference on full image + tiles (YOLOv8 / Faster R-CNN / DETR)
  4. Reprojects, merges and applies suppression
  5. Matches predictions vs GT (IoU@0.5)
  6. Computes mAP50, mAP75, mAP, P, R, F1, MAE, RMSE, r
  7. Saves annotated images; appends to results CSVs after each fold
"""

import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── GPU sanity check ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"[GPU] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | cuDNN benchmark ON")
else:
    print("[WARNING] CUDA not available — running on CPU")

from evaluation.loader import FoldTestLoader
from evaluation.matcher import DetectionMatcher
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import draw_eval_image
from config.config_loader import ConfigLoader
from inference.engine import make_engine
from inference.pipeline import _SUPPRESSION_REGISTRY
from slicing.service import make_slicer


_ARCHS = ["YOLO", "Faster", "Detr"]

_ARCH_DIR = {"YOLO": "yolo", "Faster": "faster_rcnn", "Detr": "detr"}

_RESULTS_FIELDS  = ["models", "fold", "mAP50", "mAP75", "mAP", "precision", "recall",
                    "fscore", "MAE", "RMSE", "r", "slicing_time_ms_mean", "slicing_time_ms_total"]
_COUNTING_FIELDS = ["models", "fold", "image_name", "groundtruth", "predicted"]


def model_label(mode: str, arch: str) -> str:
    return f"{mode.upper()}_{arch}"


def resolve_checkpoint(models_root: str, mode: str, fold: int, arch: str) -> str:
    manifest_path = os.path.join(
        models_root, mode, f"fold_{fold}", _ARCH_DIR[arch], "manifest.json"
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    checkpoint = manifest.get("checkpoint", "")
    if checkpoint and not os.path.isabs(checkpoint):
        checkpoint = os.path.join(os.path.dirname(manifest_path), checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def apply_suppression(raw_boxes, raw_scores, raw_labels, method: str, iou_thr: float):
    if not raw_boxes:
        return [], [], []
    fn = _SUPPRESSION_REGISTRY[method]
    fin_boxes, fin_scores, fin_labels = fn(
        np.array(raw_boxes), np.array(raw_scores), np.array(raw_labels), iou_thr
    )
    return fin_boxes.tolist(), fin_scores.tolist(), fin_labels.tolist()


def infer_image(image, slicer, engine, conf_thr, batch_size=32):
    """Slices image, measures slicing time, runs engine (full + tiles).
    Returns (tile_coords, pred_boxes, pred_scores, slicing_ms)."""
    tiles, coords = [], []
    t0 = time.perf_counter()
    for tile, c in slicer.generate_tiles(image):
        tiles.append(tile)
        coords.append(c)
    slicing_ms = (time.perf_counter() - t0) * 1000

    raw_boxes, raw_scores, raw_labels = engine.predict_full_image(
        image, conf_thr=conf_thr
    )
    tile_boxes, tile_scores, tile_labels = engine.predict_tiles(
        image,
        iter(zip(tiles, coords)),
        conf_thr=conf_thr,
        batch_size=batch_size,
    )
    raw_boxes += tile_boxes
    raw_scores += tile_scores
    raw_labels += tile_labels
    return coords, raw_boxes, raw_scores, raw_labels, slicing_ms


def save_visualization(image, tile_coords, gt_boxes, pred_boxes, pred_scores, match, out_path):
    vis = draw_eval_image(image, tile_coords, gt_boxes, pred_boxes, pred_scores, match)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def _init_csv(path: str, fieldnames: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


def run_fold(process, arch: str, fold: int, paths, results_csv, counting_csv) -> None:
    slicing_mode = process.slicing.slicing_mode
    dataset_name = os.path.basename(process.dataset.output_path.rstrip("/"))
    overlap = process.slicing.overlap_ratio
    suppression = process.inference.suppression
    conf_thr = process.inference.conf_threshold
    iou_thr = process.inference.iou_threshold
    batch_size = process.inference.batch_size
    label       = model_label(dataset_name, arch)

    weights = resolve_checkpoint(paths.models, dataset_name, fold, arch)
    test_dir = os.path.join(process.dataset.output_path, f"fold_{fold}", "test")
    vis_dir = os.path.join(paths.results, dataset_name, arch, f"fold_{fold}")

    if not os.path.isdir(os.path.join(test_dir, "images")):
        tqdm.write(f"  [SKIP] test dir not found: {test_dir}")
        return

    tqdm.write(f"  Loading {arch}: {weights}")
    engine     = make_engine(arch, weights, device="cuda")
    slicer     = make_slicer(slicing_mode, overlap)
    loader     = FoldTestLoader(test_dir)
    matcher    = DetectionMatcher(iou_threshold=0.5)
    calculator = MetricsCalculator()

    images = loader.list_images()
    tqdm.write(f"  Test images: {len(images)}")
    if not images:
        raise ValueError(f"No test images found in {test_dir}")

    all_gt, all_pred_boxes, all_pred_scores = [], [], []
    gt_counts, pred_counts = [], []
    slicing_times_ms = []
    counting_rows = []

    for image_name in tqdm(images, desc=f"  fold_{fold} [{arch}]", unit="img", ncols=80, leave=True):
        image, _  = loader.load_image(image_name)
        gt_boxes  = loader.load_gt_boxes(image_name)

        tile_coords, raw_boxes, raw_scores, raw_labels, slicing_ms = infer_image(
            image, slicer, engine, conf_thr, batch_size
        )
        slicing_times_ms.append(slicing_ms)
        fin_boxes, fin_scores, _ = apply_suppression(
            raw_boxes, raw_scores, raw_labels, suppression, iou_thr
        )
        match = matcher.match(gt_boxes, fin_boxes, fin_scores)

        stem = os.path.splitext(image_name)[0]
        save_visualization(image, tile_coords, gt_boxes, fin_boxes, fin_scores, match,
                           out_path=os.path.join(vis_dir, f"{stem}_eval.jpg"))

        all_gt.append(gt_boxes)
        all_pred_boxes.append(fin_boxes)
        all_pred_scores.append(fin_scores)
        gt_counts.append(len(gt_boxes))
        pred_counts.append(len(fin_boxes))
        counting_rows.append({
            "models": label, "fold": fold,
            "image_name": image_name,
            "groundtruth": len(gt_boxes),
            "predicted": len(fin_boxes),
        })

    map50, map75, map_all = calculator.compute_map(all_gt, all_pred_boxes, all_pred_scores)
    precision, recall, fscore = calculator.compute_prf(all_gt, all_pred_boxes, all_pred_scores)
    mae, rmse, r = calculator.compute_counting_metrics(gt_counts, pred_counts)
    slicing_mean  = round(sum(slicing_times_ms) / len(slicing_times_ms), 4)
    slicing_total = round(sum(slicing_times_ms), 4)

    result_row = {
        "models": label, "fold": fold,
        "mAP50": round(map50, 6), "mAP75": round(map75, 6), "mAP": round(map_all, 6),
        "precision": precision, "recall": recall, "fscore": fscore,
        "MAE": mae, "RMSE": rmse, "r": r,
        "slicing_time_ms_mean": slicing_mean,
        "slicing_time_ms_total": slicing_total,
    }

    # Append to CSVs immediately — safe against mid-run crashes
    _append_csv(results_csv, _RESULTS_FIELDS, [result_row])
    _append_csv(counting_csv, _COUNTING_FIELDS, counting_rows)

    tqdm.write(f"  mAP50={map50:.3f}  mAP={map_all:.3f}  P={precision:.3f}  R={recall:.3f}  "
               f"F1={fscore:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:.3f}  "
               f"slice_ms_mean={slicing_mean:.2f}  slice_ms_total={slicing_total:.0f}")


def main():
    loader = ConfigLoader("config.yaml")
    processes = loader.processes
    paths = loader.paths
    results_csv = os.path.join(paths.results, "results.csv")
    counting_csv = os.path.join(paths.results, "counting.csv")

    _init_csv(results_csv, _RESULTS_FIELDS)
    _init_csv(counting_csv, _COUNTING_FIELDS)

    for process in processes:
        dataset_name = os.path.basename(process.dataset.output_path.rstrip("/"))
        n_folds = process.crossfolds.n_folds
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Process: {dataset_name.upper()}  |  suppression: {process.inference.suppression}")
        tqdm.write(f"{'='*60}")

        for fold in tqdm(range(1, n_folds + 1), desc="  folds", unit="fold", ncols=80, leave=True):
            tqdm.write(f"\n[Fold {fold}/{n_folds}]")
            for arch in _ARCHS:
                try:
                    run_fold(
                        process, arch, fold, paths, results_csv, counting_csv
                    )
                except Exception as exc:
                    tqdm.write(f"  [ERROR] {dataset_name.upper()} fold_{fold} {arch}: {exc}")

    tqdm.write(f"\nSalvo: {results_csv}")
    tqdm.write(f"Salvo: {counting_csv}")


if __name__ == "__main__":
    main()
