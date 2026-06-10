import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold, KFold

from config.settings import SlicingConfig
from slicing.asahi import Asahi
from slicing.service import make_slicer

_COCO_FILENAME = "_annotations.coco.json"
_CLEAN_FILENAME = "_annotations_clean.coco.json"
_DEFAULT_IOA_THRESHOLD = 0.20


@dataclass
class GeometryParams:
    width: int
    height: int
    tile_size_p: int
    cols: int
    rows: int
    tiles_per_image: int
    redundant_pixels: int
    redundancy_pct: float


@dataclass
class ImageMetrics:
    image_name: str
    width: int
    height: int
    tiles_generated: int
    slicing_time_ms: float
    annotations_original: int
    annotations_kept: int
    annotations_discarded: int

    def to_row(self) -> dict:
        return {
            "image_name": self.image_name,
            "width": self.width,
            "height": self.height,
            "tiles_generated": self.tiles_generated,
            "slicing_time_ms": round(self.slicing_time_ms, 2),
            "annotations_original": self.annotations_original,
            "annotations_kept": self.annotations_kept,
            "annotations_discarded": self.annotations_discarded,
        }


@dataclass
class FoldStats:
    fold: int
    train_metrics: List[ImageMetrics] = field(default_factory=list)
    val_metrics: List[ImageMetrics] = field(default_factory=list)

    @property
    def train_images(self) -> int:
        return len(self.train_metrics)

    @property
    def val_images(self) -> int:
        return len(self.val_metrics)

    @property
    def train_tiles(self) -> int:
        return sum(m.tiles_generated for m in self.train_metrics)

    @property
    def val_tiles(self) -> int:
        return sum(m.tiles_generated for m in self.val_metrics)

    def to_dict(self) -> dict:
        all_m = self.train_metrics + self.val_metrics
        return {
            "fold": self.fold,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "train_tiles": self.train_tiles,
            "val_tiles": self.val_tiles,
            "annotations_discarded": sum(m.annotations_discarded for m in all_m),
            "mean_slicing_time_ms": round(
                sum(m.slicing_time_ms for m in all_m) / len(all_m), 2
            ) if all_m else 0.0,
        }


def compute_geometry(slicer, img_w: int, img_h: int) -> "GeometryParams":
    """Standalone helper — usable outside the validator context (e.g. in main.py)."""
    if isinstance(slicer, Asahi):
        p = slicer.compute_tile_size(img_w, img_h)
        cols, rows = slicer.compute_grid(img_w, img_h, p)
        tile_area = p * p
    else:
        tile_w, tile_h = slicer.slicing_config.tile_size
        stride_x, stride_y = slicer.compute_stride()
        cols = len(slicer._axis_positions(img_w, tile_w, stride_x))
        rows = len(slicer._axis_positions(img_h, tile_h, stride_y))
        p = tile_w
        tile_area = tile_w * tile_h

    tiles = cols * rows
    redundant_pixels = max(0, tiles * tile_area - img_w * img_h)
    redundancy_pct = round(redundant_pixels / (img_w * img_h) * 100, 2)

    return GeometryParams(
        width=img_w, height=img_h,
        tile_size_p=p, cols=cols, rows=rows,
        tiles_per_image=tiles,
        redundant_pixels=redundant_pixels,
        redundancy_pct=redundancy_pct,
    )


class AsahiKFoldValidator:
    """
    Generates YOLO-ready k-fold datasets from high-resolution images using SAHI or
    ASAHI tiling. Each fold materialises tiles on disk, writes a fold_{i}.yaml,
    and can be cleaned up after training to prevent cross-contamination.

    Expects COCO annotations at:
        dataset_path/_annotations_clean.coco.json  (preferred)
        dataset_path/_annotations.coco.json        (fallback)

    Output layout:
        output_root/
          fold_{i}/train/images|labels
          fold_{i}/val/images|labels
          fold_{i}.yaml
          summary_report.json
          resolution_groups.csv
          per_image_metrics.csv
    """

    def __init__(
        self,
        dataset_path: str,
        slicing_config: SlicingConfig,
        n_splits: int = 5,
        output_root: str = "datasets/kfold_run",
        seed: int = 42,
        ioa_threshold: float = _DEFAULT_IOA_THRESHOLD,
        groups: Optional[Dict[str, str]] = None,
    ):
        self.dataset_path = dataset_path
        self.slicing_config = slicing_config
        self.n_splits = n_splits
        self.output_root = output_root
        self.seed = seed
        self.ioa_threshold = ioa_threshold
        self.groups = groups

        self._slicer = make_slicer(slicing_config.slicing_mode, slicing_config.overlap_ratio)
        self._coco = self._load_coco()
        self._category_map = self._build_category_map()
        self._ann_by_image = self._index_annotations()
        self._geometry_cache: Dict[Tuple[int, int], GeometryParams] = {}

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _load_coco(self) -> dict:
        clean = os.path.join(self.dataset_path, _CLEAN_FILENAME)
        fallback = os.path.join(self.dataset_path, _COCO_FILENAME)
        path = clean if os.path.isfile(clean) else fallback
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"COCO annotation file not found in: {self.dataset_path}"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _build_category_map(self) -> Dict[int, int]:
        """Maps COCO category_id → zero-based YOLO class index."""
        return {
            cat["id"]: i
            for i, cat in enumerate(
                sorted(self._coco["categories"], key=lambda c: c["id"])
            )
        }

    def _index_annotations(self) -> Dict[int, List[dict]]:
        index: Dict[int, List[dict]] = {}
        for ann in self._coco["annotations"]:
            index.setdefault(ann["image_id"], []).append(ann)
        return index

    # ------------------------------------------------------------------ #
    # Dataset split                                                        #
    # ------------------------------------------------------------------ #

    def _valid_images(self) -> List[dict]:
        """Images that exist on disk and have at least one annotation."""
        return [
            img for img in self._coco["images"]
            if self._ann_by_image.get(img["id"])
            and os.path.isfile(os.path.join(self.dataset_path, img["file_name"]))
        ]

    def _make_splits(
        self, images: List[dict]
    ) -> List[Tuple[List[dict], List[dict]]]:
        indices = np.arange(len(images))

        if self.groups is not None:
            group_labels = np.array(
                [self.groups.get(img["file_name"], img["file_name"]) for img in images]
            )
            splitter = GroupKFold(n_splits=self.n_splits)
            raw = list(splitter.split(indices, groups=group_labels))
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            raw = list(splitter.split(indices))

        return [
            ([images[i] for i in train_idx], [images[i] for i in val_idx])
            for train_idx, val_idx in raw
        ]

    # ------------------------------------------------------------------ #
    # Geometry cache                                                       #
    # ------------------------------------------------------------------ #

    def _compute_geometry(self, img_w: int, img_h: int) -> GeometryParams:
        return compute_geometry(self._slicer, img_w, img_h)

    def _get_geometry(self, img_w: int, img_h: int) -> GeometryParams:
        key = (img_w, img_h)
        if key not in self._geometry_cache:
            self._geometry_cache[key] = self._compute_geometry(img_w, img_h)
        return self._geometry_cache[key]

    # ------------------------------------------------------------------ #
    # IoA guardrail + label conversion                                     #
    # ------------------------------------------------------------------ #

    def _ioa(
        self,
        bbox: Tuple[float, float, float, float],
        x_off: int,
        y_off: int,
        p: int,
    ) -> Tuple[float, Tuple[float, float, float, float]]:
        """
        Returns (ioa, clipped_bbox_in_tile_coords).
        bbox is COCO format (x, y, w, h) in absolute pixel space.
        Clipped bbox is (x, y, w, h) relative to tile origin.
        """
        bx, by, bw, bh = bbox
        bbox_area = bw * bh
        if bbox_area <= 0:
            return 0.0, (0.0, 0.0, 0.0, 0.0)

        ix1 = max(bx, x_off)
        iy1 = max(by, y_off)
        ix2 = min(bx + bw, x_off + p)
        iy2 = min(by + bh, y_off + p)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        ioa = (iw * ih) / bbox_area
        return ioa, (ix1 - x_off, iy1 - y_off, iw, ih)

    def _yolo_line(
        self, clipped: Tuple[float, float, float, float], p: int, cls: int
    ) -> str:
        cx = np.clip((clipped[0] + clipped[2] / 2) / p, 0.0, 1.0)
        cy = np.clip((clipped[1] + clipped[3] / 2) / p, 0.0, 1.0)
        w = np.clip(clipped[2] / p, 0.0, 1.0)
        h = np.clip(clipped[3] / p, 0.0, 1.0)
        return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    # ------------------------------------------------------------------ #
    # Tile materialisation                                                 #
    # ------------------------------------------------------------------ #

    def _process_image(
        self, img_meta: dict, images_dir: str, labels_dir: str
    ) -> Optional[ImageMetrics]:
        img_path = os.path.join(self.dataset_path, img_meta["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            return None

        img_h, img_w = image.shape[:2]
        annotations = self._ann_by_image.get(img_meta["id"], [])
        stem = Path(img_meta["file_name"]).stem
        surviving_ann_ids: set = set()
        tiles_written = 0

        t0 = time.perf_counter()
        for tile, coords in self._slicer.generate_tiles(image):
            x_off, y_off, p = coords["x"], coords["y"], coords["width"]
            tile_stem = f"{stem}_tile_{x_off}_{y_off}"

            yolo_lines: List[str] = []
            for ann in annotations:
                ioa, clipped = self._ioa(tuple(ann["bbox"]), x_off, y_off, p)
                if ioa < self.ioa_threshold:
                    continue
                surviving_ann_ids.add(ann["id"])
                cls = self._category_map[ann["category_id"]]
                yolo_lines.append(self._yolo_line(clipped, p, cls))

            cv2.imwrite(os.path.join(images_dir, f"{tile_stem}.jpg"), tile)
            tiles_written += 1
            with open(os.path.join(labels_dir, f"{tile_stem}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))

        slicing_time_ms = (time.perf_counter() - t0) * 1000
        self._get_geometry(img_w, img_h)

        annotations_original = len(annotations)
        annotations_kept = len(surviving_ann_ids)

        return ImageMetrics(
            image_name=img_meta["file_name"],
            width=img_w,
            height=img_h,
            tiles_generated=tiles_written,
            slicing_time_ms=slicing_time_ms,
            annotations_original=annotations_original,
            annotations_kept=annotations_kept,
            annotations_discarded=annotations_original - annotations_kept,
        )

    def _process_split(self, images: List[dict], split_dir: str) -> List[ImageMetrics]:
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        metrics: List[ImageMetrics] = []
        for img in images:
            m = self._process_image(img, images_dir, labels_dir)
            if m is not None:
                metrics.append(m)
        return metrics

    # ------------------------------------------------------------------ #
    # YAML                                                                 #
    # ------------------------------------------------------------------ #

    def _write_yaml(self, fold_index: int, train_dir: str, val_dir: str):
        names = {
            i: cat["name"]
            for i, cat in enumerate(
                sorted(self._coco["categories"], key=lambda c: c["id"])
            )
        }
        content = {
            "path": str(Path(self.output_root).resolve()),
            "train": str((Path(train_dir) / "images").resolve()),
            "val": str((Path(val_dir) / "images").resolve()),
            "nc": len(names),
            "names": names,
        }
        yaml_path = os.path.join(self.output_root, f"fold_{fold_index}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, allow_unicode=True)

    # ------------------------------------------------------------------ #
    # Report writers                                                       #
    # ------------------------------------------------------------------ #

    def _write_reports(self, folds: List[FoldStats]):
        # Deduplicate by image_name — same image appears in train/val across folds
        seen: Dict[str, ImageMetrics] = {}
        for fold in folds:
            for m in fold.train_metrics + fold.val_metrics:
                if m.image_name not in seen:
                    seen[m.image_name] = m

        unique_metrics = list(seen.values())
        if not unique_metrics:
            return

        # per_image_metrics.csv
        img_csv = os.path.join(self.output_root, "per_image_metrics.csv")
        with open(img_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(unique_metrics[0].to_row()))
            writer.writeheader()
            writer.writerows(m.to_row() for m in unique_metrics)

        # resolution_groups.csv
        res_count: Dict[Tuple[int, int], int] = {}
        for m in unique_metrics:
            res_count[(m.width, m.height)] = res_count.get((m.width, m.height), 0) + 1

        res_csv = os.path.join(self.output_root, "resolution_groups.csv")
        res_fields = ["resolution", "image_count", "tile_size_p", "cols", "rows",
                      "tiles_per_image", "redundant_pixels", "redundancy_pct"]
        with open(res_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=res_fields)
            writer.writeheader()
            for (w, h), count in sorted(res_count.items()):
                geom = self._geometry_cache[(w, h)]
                writer.writerow({
                    "resolution": f"{w}x{h}",
                    "image_count": count,
                    "tile_size_p": geom.tile_size_p,
                    "cols": geom.cols,
                    "rows": geom.rows,
                    "tiles_per_image": geom.tiles_per_image,
                    "redundant_pixels": geom.redundant_pixels,
                    "redundancy_pct": geom.redundancy_pct,
                })

        # summary_report.json
        all_fold_metrics = [
            m for fold in folds for m in fold.train_metrics + fold.val_metrics
        ]
        total_tiles_all_folds = sum(m.tiles_generated for m in all_fold_metrics)
        slicing_times = [m.slicing_time_ms for m in unique_metrics]
        orig_annotations = sum(m.annotations_original for m in unique_metrics)
        discarded = sum(m.annotations_discarded for m in unique_metrics)

        summary = {
            "slicing_mode": self.slicing_config.slicing_mode,
            "n_splits": self.n_splits,
            "overlap_ratio": self.slicing_config.overlap_ratio,
            "dataset_summary": {
                "original_images": len(unique_metrics),
                "total_tiles_generated_all_folds": total_tiles_all_folds,
                "mean_tiles_per_image": round(
                    total_tiles_all_folds / len(unique_metrics), 2
                ),
            },
            "time_profiling_ms": {
                "mean_slicing_time_cpu": round(float(np.mean(slicing_times)), 2),
                "std_slicing_time_cpu": round(float(np.std(slicing_times)), 2),
                "mean_inference_time_gpu_batch": None,
                "mean_nms_reprojection_cpu": None,
            },
            "label_integrity": {
                "original_annotations": orig_annotations,
                "kept_annotations": orig_annotations - discarded,
                "discarded_by_ioa": discarded,
                "global_discard_rate_pct": round(
                    discarded / orig_annotations * 100, 2
                ) if orig_annotations else 0.0,
            },
            "folds": [f.to_dict() for f in folds],
        }

        json_path = os.path.join(self.output_root, "summary_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nReports written to {self.output_root}/")
        print(f"  summary_report.json")
        print(f"  resolution_groups.csv")
        print(f"  per_image_metrics.csv")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_fold(
        self,
        fold_index: int,
        train_images: List[dict],
        val_images: List[dict],
    ) -> FoldStats:
        """Materialises one fold on disk and writes its fold_{i}.yaml."""
        fold_dir = os.path.join(self.output_root, f"fold_{fold_index}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")

        train_metrics = self._process_split(train_images, train_dir)
        val_metrics = self._process_split(val_images, val_dir)
        self._write_yaml(fold_index, train_dir, val_dir)

        return FoldStats(fold=fold_index, train_metrics=train_metrics, val_metrics=val_metrics)

    def cleanup_fold(self, fold_index: int):
        """Deletes the physical fold directory and any .cache files ultralytics
        may have written alongside the images/labels dirs."""
        fold_dir = os.path.join(self.output_root, f"fold_{fold_index}")
        if os.path.isdir(fold_dir):
            shutil.rmtree(fold_dir)

        for root, _dirs, files in os.walk(self.output_root):
            for fname in files:
                if fname.endswith(".cache"):
                    os.remove(os.path.join(root, fname))

    def run(self) -> List[FoldStats]:
        """Generates all folds on disk and writes experiment reports."""
        images = self._valid_images()
        if len(images) < self.n_splits:
            raise ValueError(
                f"Dataset has {len(images)} images but {self.n_splits} splits were requested."
            )

        os.makedirs(self.output_root, exist_ok=True)
        results: List[FoldStats] = []

        for fold_index, (train_imgs, val_imgs) in enumerate(self._make_splits(images), start=1):
            print(f"[Fold {fold_index}/{self.n_splits}] Generating tiles...")
            stats = self.generate_fold(fold_index, train_imgs, val_imgs)
            print(
                f"  train {stats.train_tiles} tiles / {stats.train_images} images  |  "
                f"val {stats.val_tiles} tiles / {stats.val_images} images"
            )
            results.append(stats)

        self._write_reports(results)
        return results
