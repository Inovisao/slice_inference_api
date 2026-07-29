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
import argparse

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
from config.config_loader import ConfigLoader, engine_dir_for_model
from inference.engine import make_engine
from inference.pipeline import _SUPPRESSION_REGISTRY
from slicing.service import make_slicer


# Maps the manifest arch dir (from the model name's engine) to the inference engine key.
_ENGINE_KEY = {"yolo": "YOLO", "faster_rcnn": "Faster", "detr": "Detr"}

_RESULTS_FIELDS  = ["models", "fold", "mAP50", "mAP75", "mAP", "precision", "recall",
                    "fscore", "MAE", "RMSE", "r", "slicing_time_ms_mean", "slicing_time_ms_total"]
_COUNTING_FIELDS = [
    "models",
    "fold",
    "image_name",
    "groundtruth",
    "raw_predicted",
    "predicted",
    "removed_by_suppression",
    "TP",
    "FP",
    "FN",
]


def manifest_model_dir(model_name: str) -> str:
    """Return the manifest folder name for a configured model label."""
    upper = model_name.upper()
    if upper == "DETR":
        return "detr"
    if upper == "FASTER":
        return "Faster"
    return model_name


def model_label(output_name: str, model_name: str) -> str:
    """CSV label: <OUTPUT_NAME>_<MODEL>, e.g. ASAHI_RECT_YOLOV8L."""
    return f"{output_name.upper()}_{model_name}"


def engine_key_for_model(model_name: str) -> str:
    """Inference engine key ('YOLO'|'Faster'|'Detr') inferred from the model name."""
    return _ENGINE_KEY[engine_dir_for_model(model_name)]


def resolve_checkpoint(models_root: str, weights_dataset: str, fold: int, model_name: str) -> str:
    """Resolve a checkpoint by weights dataset (crop folder) + model name.

    weights_dataset is the crop folder that owns the weights (e.g. 'asahi_rect'),
    NOT the setup's output_name — several setups may share the same weights.
    model_name is the weights subfolder (e.g. 'YOLOV8L') under that fold.
    """
    model_dir = manifest_model_dir(model_name)
    manifest_path = os.path.join(
        models_root, weights_dataset, f"fold_{fold}", model_dir, "manifest.json"
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
    return infer_image_with_options(
        image,
        slicer,
        engine,
        conf_thr,
        batch_size=batch_size,
        include_full_image=True,
    )


def infer_image_with_options(
    image,
    slicer,
    engine,
    conf_thr,
    batch_size=32,
    include_full_image=True,
):
    """Slices image and returns raw predictions reprojected to full-image coordinates."""
    tiles, coords = [], []
    t0 = time.perf_counter()
    for tile, c in slicer.generate_tiles(image):
        tiles.append(tile)
        coords.append(c)
    slicing_ms = (time.perf_counter() - t0) * 1000

    raw_boxes, raw_scores, raw_labels = [], [], []
    if include_full_image:
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


def _ensure_csv_header(path: str, fieldnames: list) -> None:
    """Create the CSV with a header only if it does not exist yet.

    Never truncates: results accumulate across runs (append-only). If a setup is
    re-run, its rows are appended again — dedup by removing old rows manually.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        with open(path, newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header == fieldnames:
            return
        backup_path = f"{path}.backup_{time.strftime('%Y%m%d_%H%M%S')}"
        os.replace(path, backup_path)
        tqdm.write(
            f"[INFO] CSV com cabeçalho incompatível arquivado em {backup_path}"
        )

    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate configured slicing setups and detector checkpoints."
    )
    parser.add_argument(
        "--setup",
        action="append",
        help="Setup/output_name to evaluate. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model name to evaluate. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--folds",
        help="Folds to evaluate, comma-separated. Example: 1,2,3,4,5",
    )
    parser.add_argument(
        "--results-csv",
        help="Alternative results CSV path. Defaults to config paths.results/results.csv.",
    )
    parser.add_argument(
        "--counting-csv",
        help="Alternative counting CSV path. Defaults to config paths.results/counting.csv.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        help="Override inference confidence threshold for diagnostic runs.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        help="Override suppression IoU threshold for diagnostic runs.",
    )
    parser.add_argument(
        "--suppression",
        choices=sorted(_SUPPRESSION_REGISTRY),
        help="Override suppression method for diagnostic runs.",
    )
    parser.add_argument(
        "--no-full-image",
        action="store_true",
        help="Run only tiled inference, without the full-image pass.",
    )
    return parser.parse_args()


def _split_filter(values):
    if not values:
        return None
    parsed = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return parsed


def canonical_result_model_name(model_name: str) -> str:
    """Canonical model label used for CSV labels and results/<setup>/<model>/."""
    name = model_name.strip()
    upper = name.upper()
    if upper == "DETR":
        return "DETR"
    if upper in {"FASTER", "FASTERRCNN"}:
        return "Faster"
    if upper == "YOLO":
        raise ValueError(
            "Modelo 'YOLO' é ambíguo para resultados. Use a variante explícita, "
            "por exemplo YOLOV8N."
        )
    if upper.startswith("YOLO"):
        return upper
    return name


def run_fold(
    process,
    model_name: str,
    fold: int,
    paths,
    results_csv,
    counting_csv,
    *,
    conf_threshold_override=None,
    iou_threshold_override=None,
    suppression_override=None,
    include_full_image=True,
) -> bool:
    slicing_mode = process.slicing.slicing_mode
    output_name = process.output_name
    weights_dataset = os.path.basename(process.dataset.output_path.rstrip("/"))
    overlap = process.slicing.overlap_ratio
    suppression = suppression_override or process.inference.suppression
    conf_thr = (
        conf_threshold_override
        if conf_threshold_override is not None
        else process.inference.conf_threshold
    )
    iou_thr = (
        iou_threshold_override
        if iou_threshold_override is not None
        else process.inference.iou_threshold
    )
    batch_size = process.inference.batch_size
    label       = model_label(output_name, model_name)
    engine_key  = engine_key_for_model(model_name)

    weights = resolve_checkpoint(paths.models, weights_dataset, fold, model_name)
    test_dir = os.path.join(process.dataset.output_path, f"fold_{fold}", "test")
    vis_dir = os.path.join(paths.results, output_name, model_name, f"fold_{fold}")

    if not os.path.isdir(os.path.join(test_dir, "images")):
        tqdm.write(f"  [SKIP] test dir not found: {test_dir}")
        return False

    tqdm.write(f"  Loading {model_name} ({engine_key}): {weights}")
    engine     = make_engine(engine_key, weights, device="cuda")
    slicer     = make_slicer(slicing_mode, overlap, process.slicing.tile_size)
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
    total_tp = total_fp = total_fn = 0

    for image_name in tqdm(images, desc=f"  fold_{fold} [{model_name}]", unit="img", ncols=80, leave=True):
        image, _  = loader.load_image(image_name)
        gt_boxes  = loader.load_gt_boxes(image_name)

        tile_coords, raw_boxes, raw_scores, raw_labels, slicing_ms = infer_image_with_options(
            image,
            slicer,
            engine,
            conf_thr,
            batch_size=batch_size,
            include_full_image=include_full_image,
        )
        slicing_times_ms.append(slicing_ms)
        fin_boxes, fin_scores, _ = apply_suppression(
            raw_boxes, raw_scores, raw_labels, suppression, iou_thr
        )
        match = matcher.match(gt_boxes, fin_boxes, fin_scores)
        tp = len(match.tp_pred_indices)
        fp = len(match.fp_pred_indices)
        fn = len(match.fn_gt_indices)
        total_tp += tp
        total_fp += fp
        total_fn += fn

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
            "raw_predicted": len(raw_boxes),
            "predicted": len(fin_boxes),
            "removed_by_suppression": len(raw_boxes) - len(fin_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        })

    map50, map75, map_all = calculator.compute_map(all_gt, all_pred_boxes, all_pred_scores)
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    fscore = 2 * precision * recall / (precision + recall + 1e-9)
    precision, recall, fscore = round(precision, 6), round(recall, 6), round(fscore, 6)
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
    return True


def main():
    args = _parse_args()
    loader = ConfigLoader("config.yaml")
    processes = loader.processes
    paths = loader.paths
    setup_filter = _split_filter(args.setup)
    model_filter = _split_filter(args.model)
    fold_filter = (
        [int(item.strip()) for item in args.folds.split(",") if item.strip()]
        if args.folds
        else None
    )

    if setup_filter:
        allowed_setups = {item.lower() for item in setup_filter}
        processes = [p for p in processes if p.output_name.lower() in allowed_setups]

    if model_filter:
        allowed_models = {item.lower() for item in model_filter}
        for process in processes:
            process.models = [
                model for model in process.models if model.lower() in allowed_models
            ]
        processes = [p for p in processes if p.models]

    if not processes:
        raise ValueError("Nenhum setup/modelo selecionado pelos filtros informados.")

    results_csv = args.results_csv or os.path.join(paths.results, "results.csv")
    counting_csv = args.counting_csv or os.path.join(paths.results, "counting.csv")

    # Append-only: keep existing results, add the selected setups' rows.
    _ensure_csv_header(results_csv, _RESULTS_FIELDS)
    _ensure_csv_header(counting_csv, _COUNTING_FIELDS)

    tqdm.write(f"Setups selecionados: {[p.output_name for p in processes]}")
    successful_folds = 0
    failed_folds = 0

    for process in processes:
        output_name = process.output_name
        n_folds = process.crossfolds.n_folds
        models = [canonical_result_model_name(model_name) for model_name in process.models]
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Setup: {output_name.upper()}  |  modelos: {models}  |  "
                   f"suppression: {process.inference.suppression}")
        tqdm.write(f"{'='*60}")

        folds_to_run = fold_filter or list(range(1, n_folds + 1))
        for fold in tqdm(folds_to_run, desc="  folds", unit="fold", ncols=80, leave=True):
            if fold < 1 or fold > n_folds:
                raise ValueError(f"Fold inválido para {output_name}: {fold}. Esperado 1..{n_folds}")
            tqdm.write(f"\n[Fold {fold}/{n_folds}]")
            for model_name in models:
                try:
                    if run_fold(
                        process,
                        model_name,
                        fold,
                        paths,
                        results_csv,
                        counting_csv,
                        conf_threshold_override=args.conf_threshold,
                        iou_threshold_override=args.iou_threshold,
                        suppression_override=args.suppression,
                        include_full_image=not args.no_full_image,
                    ):
                        successful_folds += 1
                except Exception as exc:
                    failed_folds += 1
                    tqdm.write(f"  [ERROR] {output_name.upper()} fold_{fold} {model_name}: {exc}")

    tqdm.write(f"\nSalvo: {results_csv}")
    tqdm.write(f"Salvo: {counting_csv}")
    if successful_folds == 0:
        raise RuntimeError(
            f"Nenhum fold foi executado com sucesso ({failed_folds} falhas). "
            "Verifique manifests, checkpoints e diretórios de teste."
        )


if __name__ == "__main__":
    main()
