#!/usr/bin/env python3
"""Build a 640x640 full-image baseline dataset with the existing fold splits.

The script does not create a new random split. It reads the fold membership from
an already generated tiled dataset, collapses train tiles back to their original
image names, and writes a strict cross-fold dataset consumable by train_model.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
TILE_RE = re.compile(r"^(?P<stem>.+)_tile_\d+_\d+$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _source_annotation_path(source: Path) -> Path:
    clean = source / "_annotations_clean.coco.json"
    if clean.exists():
        return clean
    raw = source / "_annotations.coco.json"
    if raw.exists():
        return raw
    raise FileNotFoundError(
        f"COCO annotation not found. Expected {clean} or {raw}"
    )


def _original_stem(stem: str) -> str:
    if stem.endswith("_fi"):
        return stem[:-3]
    match = TILE_RE.match(stem)
    if match:
        return match.group("stem")
    return stem


def _resolve_original_name(file_name: str, names_by_stem: dict[str, str]) -> str:
    path = Path(file_name)
    if file_name in names_by_stem.values():
        return file_name

    stem = _original_stem(path.stem)
    try:
        return names_by_stem[stem]
    except KeyError as exc:
        raise KeyError(
            f"Could not map fold image '{file_name}' back to an original image."
        ) from exc


def _index_source(coco: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[int, list[dict[str, Any]]]]:
    images_by_name: dict[str, dict[str, Any]] = {}
    names_by_stem: dict[str, str] = {}

    for image in coco.get("images", []):
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        if stem in names_by_stem:
            raise ValueError(
                f"Duplicate image stem '{stem}' in source dataset. "
                "Unique stems are required to recover tiled fold membership."
            )
        images_by_name[file_name] = image
        names_by_stem[stem] = file_name

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    return images_by_name, names_by_stem, annotations_by_image


def _drop_unused_categories(coco: dict[str, Any]) -> None:
    present_ids = {
        int(annotation["category_id"])
        for annotation in coco.get("annotations", [])
    }
    if not present_ids:
        return
    coco["categories"] = [
        category
        for category in coco.get("categories", [])
        if int(category["id"]) in present_ids
    ]


def _fold_original_names(
    fold_source: Path,
    fold: str,
    split: str,
    names_by_stem: dict[str, str],
) -> list[str]:
    split_json = fold_source / "filesJSON" / f"{fold}_{split}.json"
    data = _load_json(split_json)

    names = {
        _resolve_original_name(str(image["file_name"]), names_by_stem)
        for image in data.get("images", [])
    }
    return sorted(names, key=lambda item: (Path(item).stem.zfill(12), item))


def _resize_image(source_path: Path, target_path: Path, size: int) -> None:
    import cv2

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {source_path}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target_path), resized):
        raise ValueError(f"Could not write image: {target_path}")


def _scale_bbox(
    bbox: list[float],
    source_width: float,
    source_height: float,
    target_size: int,
) -> list[float] | None:
    x, y, w, h = [float(value) for value in bbox]
    sx = target_size / source_width
    sy = target_size / source_height

    x1 = max(0.0, min(target_size, x * sx))
    y1 = max(0.0, min(target_size, y * sy))
    x2 = max(0.0, min(target_size, (x + w) * sx))
    y2 = max(0.0, min(target_size, (y + h) * sy))

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None
    return [x1, y1, new_w, new_h]


def _write_yolo_label(label_path: Path, annotations: list[dict[str, Any]], size: int) -> None:
    lines = []
    for annotation in annotations:
        x, y, w, h = [float(value) for value in annotation["bbox"]]
        class_index = int(annotation["category_id"]) - 1
        x_center = (x + w / 2.0) / size
        y_center = (y + h / 2.0) / size
        lines.append(
            f"{class_index} {x_center:.6f} {y_center:.6f} "
            f"{w / size:.6f} {h / size:.6f}\n"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(lines), encoding="utf-8")


def _build_split(
    source: Path,
    output: Path,
    source_coco: dict[str, Any],
    images_by_name: dict[str, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    fold: str,
    split: str,
    original_names: list[str],
    size: int,
) -> None:
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    new_image_id = 1
    new_annotation_id = 1

    image_dir = output / fold / split / "images"
    label_dir = output / fold / split / "labels"

    for file_name in original_names:
        source_image = images_by_name[file_name]
        source_width = float(source_image["width"])
        source_height = float(source_image["height"])
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid dimensions for {file_name}")

        _resize_image(source / file_name, image_dir / file_name, size)

        image_annotations: list[dict[str, Any]] = []
        for annotation in annotations_by_image.get(int(source_image["id"]), []):
            scaled_bbox = _scale_bbox(
                list(annotation["bbox"]), source_width, source_height, size
            )
            if scaled_bbox is None:
                continue
            new_annotation = dict(annotation)
            new_annotation["id"] = new_annotation_id
            new_annotation["image_id"] = new_image_id
            new_annotation["bbox"] = scaled_bbox
            new_annotation["area"] = scaled_bbox[2] * scaled_bbox[3]
            annotations_out.append(new_annotation)
            image_annotations.append(new_annotation)
            new_annotation_id += 1

        _write_yolo_label(
            label_dir / f"{Path(file_name).stem}.txt",
            image_annotations,
            size,
        )

        new_image = dict(source_image)
        new_image["id"] = new_image_id
        new_image["file_name"] = file_name
        new_image["width"] = size
        new_image["height"] = size
        images_out.append(new_image)
        new_image_id += 1

    split_payload = {
        "info": source_coco.get("info", {}),
        "licenses": source_coco.get("licenses", []),
        "categories": source_coco.get("categories", []),
        "images": images_out,
        "annotations": annotations_out,
    }
    _write_json(output / "filesJSON" / f"{fold}_{split}.json", split_payload)


def _write_fold_yaml(output: Path, fold: str, categories: list[dict[str, Any]]) -> None:
    names = [cat["name"] for cat in sorted(categories, key=lambda cat: int(cat["id"]))]
    yaml_path = output / "filesJSON_infos" / f"{fold}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {str((output / fold).resolve())}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                f"nc: {len(names)}",
                "names:",
                *[f"  - {name}" for name in names],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(
    output: Path,
    source: Path,
    fold_source: Path,
    size: int,
    seed: int,
    folds: list[str],
    categories: list[dict[str, Any]],
) -> None:
    manifest = {
        "contract_version": "1.0",
        "dataset_name": output.name,
        "dataset_type": "full_image_baseline_crossfold",
        "annotation_format": "coco",
        "category_id_base": 1,
        "splits": list(SPLITS),
        "folds": folds,
        "layout": {
            "annotations_dir": "filesJSON",
            "annotation_pattern": "{fold}_{split}.json",
            "image_dir_pattern": "{fold}/{split}/images",
            "label_dir_pattern": "{fold}/{split}/labels",
            "fold_info_dir": "filesJSON_infos",
            "fold_yaml_pattern": "{fold}.yaml",
        },
        "baseline": {
            "mode": "all_640",
            "resize": "direct_square",
            "width": size,
            "height": size,
            "seed_reference": seed,
            "split_source": str(fold_source),
            "source_dataset": str(source),
            "new_random_split": False,
        },
        "tiling": {
            "mode": "baseline_all_640",
            "evaluation_mode": "basic",
            "tile_shape": "none",
            "tile_width": size,
            "tile_height": size,
            "variable_tile_size": False,
            "tiles_are_primary_samples": False,
            "requires_reconstruction": False,
        },
        "classes": [
            {"id": int(category["id"]), "name": category["name"]}
            for category in categories
        ],
    }
    _write_json(output / "dataset_manifest.json", manifest)


def _discover_folds(fold_source: Path) -> list[str]:
    files_json = fold_source / "filesJSON"
    folds = {
        "_".join(path.stem.split("_")[:2])
        for path in files_json.glob("fold_*_*.json")
    }
    if not folds:
        raise FileNotFoundError(f"No fold JSONs found in {files_json}")
    return sorted(folds, key=lambda fold: int(fold.split("_")[1]))


def _collect_split_plan(
    fold_source: Path,
    folds: list[str],
    names_by_stem: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    plan: dict[str, dict[str, list[str]]] = {}
    for fold in folds:
        plan[fold] = {}
        for split in SPLITS:
            plan[fold][split] = _fold_original_names(
                fold_source, fold, split, names_by_stem
            )
    return plan


def _validate_split_ratios(
    plan: dict[str, dict[str, list[str]]],
    expected_val_ratio: float,
    expected_test_ratio: float,
    tolerance_images: int,
) -> None:
    errors: list[str] = []
    expected_train_ratio = 1.0 - expected_val_ratio - expected_test_ratio

    for fold, splits in plan.items():
        all_names = set().union(*(set(names) for names in splits.values()))
        total = len(all_names)
        if total == 0:
            errors.append(f"{fold}: no images discovered")
            continue

        expected_val = round(total * expected_val_ratio)
        expected_test = round(total * expected_test_ratio)
        expected = {
            "train": total - expected_val - expected_test,
            "val": expected_val,
            "test": expected_test,
        }
        observed = {split: len(splits[split]) for split in SPLITS}

        for split in SPLITS:
            delta = abs(observed[split] - expected[split])
            if delta > tolerance_images:
                errors.append(
                    f"{fold}: {split} has {observed[split]} images, "
                    f"expected ~{expected[split]} for "
                    f"{expected_train_ratio:.2f}/"
                    f"{expected_val_ratio:.2f}/"
                    f"{expected_test_ratio:.2f}"
                )

    if errors:
        preview = "\n  - ".join(errors[:15])
        raise ValueError(
            "Fold source split ratios do not match the expected baseline split.\n"
            f"  - {preview}\n"
            "Regenerate the tiled fold-source with the current config first, "
            "or pass --skip-ratio-check intentionally."
        )


def build_baseline(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    fold_source = args.fold_source.resolve()
    output = args.output.resolve()

    annotation_path = _source_annotation_path(source)
    source_coco = _load_json(annotation_path)
    _drop_unused_categories(source_coco)
    images_by_name, names_by_stem, annotations_by_image = _index_source(source_coco)
    folds = _discover_folds(fold_source)
    split_plan = _collect_split_plan(fold_source, folds, names_by_stem)

    if not args.skip_ratio_check:
        _validate_split_ratios(
            split_plan,
            expected_val_ratio=args.expected_val_ratio,
            expected_test_ratio=args.expected_test_ratio,
            tolerance_images=args.ratio_tolerance_images,
        )

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)

    for fold in folds:
        print(f"[baseline] {fold}")
        for split in SPLITS:
            original_names = split_plan[fold][split]
            print(f"  - {split}: {len(original_names)} original images")
            _build_split(
                source=source,
                output=output,
                source_coco=source_coco,
                images_by_name=images_by_name,
                annotations_by_image=annotations_by_image,
                fold=fold,
                split=split,
                original_names=original_names,
                size=args.size,
            )
        _write_fold_yaml(output, fold, source_coco.get("categories", []))

    _write_manifest(
        output=output,
        source=source,
        fold_source=fold_source,
        size=args.size,
        seed=args.seed,
        folds=folds,
        categories=source_coco.get("categories", []),
    )
    print(f"[baseline] done: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all_640 baseline dataset using existing seed-42 folds."
    )
    parser.add_argument("--source", type=Path, required=True, help="dataset/all")
    parser.add_argument(
        "--fold-source",
        type=Path,
        required=True,
        help="Dataset whose filesJSON define the fold membership, e.g. dataset/asahi_rect",
    )
    parser.add_argument("--output", type=Path, required=True, help="dataset/all_640")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded seed reference. No new random split is created.",
    )
    parser.add_argument(
        "--expected-val-ratio",
        type=float,
        default=0.12,
        help="Expected global validation ratio in the fold-source. Default matches kfold_holdout with n_folds=5 and val_ratio=0.15.",
    )
    parser.add_argument(
        "--expected-test-ratio",
        type=float,
        default=0.20,
        help="Expected global test ratio in the fold-source. Default matches kfold_holdout with n_folds=5.",
    )
    parser.add_argument("--ratio-tolerance-images", type=int, default=1)
    parser.add_argument(
        "--skip-ratio-check",
        action="store_true",
        help="Allow building from a fold-source whose split ratios do not match the expected ratios.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_baseline(parse_args())
