"""
Reconstrói fold_{i}_stats.json para folds já existentes no disco (gerados
antes da persistência de stats ser implementada).

Uso: python scripts/reconstruct_fold_stats.py [output_dir] [coco_json]
"""

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset.kfold_generator import FoldStats, ImageMetrics

_TILE_RE = re.compile(r"^(.+)_tile_\d+_\d+\.jpg$")


def _count_kept_annotations(labels_dir: str, stem: str) -> int:
    """Count unique annotation lines across all tile labels for one source image."""
    kept = 0
    for fname in os.listdir(labels_dir):
        if not fname.startswith(stem + "_tile_"):
            continue
        path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
        if os.path.isfile(path):
            with open(path) as f:
                kept += sum(1 for line in f if line.strip())
    return kept


def reconstruct_split(images_dir: str, labels_dir: str, coco_images: dict) -> list:
    tiles_per_image: dict = {}
    for fname in os.listdir(images_dir):
        m = _TILE_RE.match(fname)
        if m:
            stem = m.group(1)
            tiles_per_image[stem] = tiles_per_image.get(stem, 0) + 1

    metrics = []
    for stem, tile_count in tiles_per_image.items():
        coco_img = coco_images.get(stem)
        if coco_img is None:
            w, h, ann_orig = 0, 0, 0
        else:
            w = coco_img.get("width", 0)
            h = coco_img.get("height", 0)
            ann_orig = coco_img.get("ann_count", 0)

        ann_kept = _count_kept_annotations(labels_dir, stem)
        metrics.append(ImageMetrics(
            image_name=stem,
            width=w,
            height=h,
            tiles_generated=tile_count,
            slicing_time_ms=0.0,
            annotations_original=ann_orig,
            annotations_kept=ann_kept,
            annotations_discarded=max(0, ann_orig - ann_kept),
        ))
    return metrics


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dataset", "asahi")
    coco_path  = sys.argv[2] if len(sys.argv) > 2 else None

    # locate COCO JSON
    if coco_path is None:
        dataset_dir = os.path.join(ROOT, "dataset", "all")
        for fname in ("_annotations_clean.coco.json", "_annotations.coco.json"):
            candidate = os.path.join(dataset_dir, fname)
            if os.path.isfile(candidate):
                coco_path = candidate
                break
    if coco_path is None or not os.path.isfile(coco_path):
        sys.exit(f"COCO JSON não encontrado. Passe o caminho como segundo argumento.")

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    # index annotations per image
    ann_count: dict = {}
    for ann in coco.get("annotations", []):
        ann_count[ann["image_id"]] = ann_count.get(ann["image_id"], 0) + 1

    # index images by stem (sem extensão)
    coco_images: dict = {}
    for img in coco.get("images", []):
        stem = os.path.splitext(img["file_name"])[0]
        coco_images[stem] = {
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "ann_count": ann_count.get(img["id"], 0),
        }

    fold_index = 1
    while True:
        fold_dir = os.path.join(output_dir, f"fold_{fold_index}")
        info_dir = os.path.join(output_dir, "filesJSON_infos")
        os.makedirs(info_dir, exist_ok=True)
        stats_path = os.path.join(info_dir, f"fold_{fold_index}_stats.json")

        if not os.path.isdir(fold_dir):
            break
        if os.path.isfile(stats_path):
            print(f"fold_{fold_index}_stats.json já existe — pulando.")
            fold_index += 1
            continue

        train_images = os.path.join(fold_dir, "train", "images")
        train_labels = os.path.join(fold_dir, "train", "labels")
        val_images   = os.path.join(fold_dir, "val",   "images")
        val_labels   = os.path.join(fold_dir, "val",   "labels")

        if not os.path.isdir(train_images):
            print(f"fold_{fold_index}: diretório de imagens não encontrado — pulando.")
            fold_index += 1
            continue

        print(f"Reconstruindo fold_{fold_index}...", end=" ", flush=True)
        train_m = reconstruct_split(train_images, train_labels, coco_images)
        val_m   = reconstruct_split(val_images,   val_labels,   coco_images)
        stats = FoldStats(fold=fold_index, train_metrics=train_m, val_metrics=val_m)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_full_dict(), f, ensure_ascii=False)
        print(f"salvo ({len(train_m)} train imgs, {len(val_m)} val imgs).")
        fold_index += 1

    print("Concluído.")


if __name__ == "__main__":
    main()
