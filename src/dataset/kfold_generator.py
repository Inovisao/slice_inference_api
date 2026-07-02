import csv
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from config.settings import SlicingConfig
from slicing.asahi import Asahi
from slicing.service import make_slicer

_COCO_FILENAME = "_annotations.coco.json"
_CLEAN_FILENAME = "_annotations_clean.coco.json"
_DEFAULT_IOA_THRESHOLD = 0.20

# (tile_img, stem, yolo_lines)
_AnnotatedTile = Tuple[np.ndarray, str, List[str]]
# (tile_img, stem)
_EmptyTile = Tuple[np.ndarray, str]


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
    empty_tiles_kept: int = 0
    empty_tiles_discarded: int = 0

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
            "empty_tiles_kept": self.empty_tiles_kept,
            "empty_tiles_discarded": self.empty_tiles_discarded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageMetrics":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class FoldStats:
    fold: int
    train_metrics: List[ImageMetrics] = field(default_factory=list)
    val_metrics: List[ImageMetrics] = field(default_factory=list)
    test_metrics: List[ImageMetrics] = field(default_factory=list)

    @property
    def train_images(self) -> int:
        return len(self.train_metrics)

    @property
    def val_images(self) -> int:
        return len(self.val_metrics)

    @property
    def test_images(self) -> int:
        return len(self.test_metrics)

    @property
    def train_tiles(self) -> int:
        return sum(m.tiles_generated for m in self.train_metrics)

    @property
    def val_tiles(self) -> int:
        return sum(m.tiles_generated for m in self.val_metrics)

    @property
    def test_tiles(self) -> int:
        return sum(m.tiles_generated for m in self.test_metrics)

    def to_dict(self) -> dict:
        all_m = self.train_metrics + self.val_metrics + self.test_metrics
        return {
            "fold": self.fold,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "test_images": self.test_images,
            "train_tiles": self.train_tiles,
            "val_tiles": self.val_tiles,
            "test_tiles": self.test_tiles,
            "annotations_discarded": sum(m.annotations_discarded for m in all_m),
            "mean_slicing_time_ms": round(
                sum(m.slicing_time_ms for m in all_m) / len(all_m), 2
            ) if all_m else 0.0,
        }

    def to_full_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train_metrics": [m.to_row() for m in self.train_metrics],
            "val_metrics": [m.to_row() for m in self.val_metrics],
            "test_metrics": [m.to_row() for m in self.test_metrics],
        }

    @classmethod
    def from_full_dict(cls, d: dict) -> "FoldStats":
        return cls(
            fold=d["fold"],
            train_metrics=[ImageMetrics.from_dict(m) for m in d["train_metrics"]],
            val_metrics=[ImageMetrics.from_dict(m) for m in d["val_metrics"]],
            test_metrics=[ImageMetrics.from_dict(m) for m in d.get("test_metrics", [])],
        )


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

    Train split: letterbox FI image + tiled images (ASAHI resized with smart interpolation).
                 Empty tiles sampled globally per fold at empty_tile_ratio.
    Val/Test splits: original full-resolution images + YOLO labels normalised by original dims.
                     No tiling — ultralytics letterboxes internally during evaluation.

    Expects COCO annotations at:
        dataset_path/_annotations_clean.coco.json  (preferred)
        dataset_path/_annotations.coco.json        (fallback)

    Output layout:
        output_root/
          fold_{i}/train/images|labels
          fold_{i}/val/images|labels
          fold_{i}/test/images|labels
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
        empty_tile_ratio: float = 0.08,
        val_ratio: float = 0.15,
        groups: Optional[Dict[str, str]] = None,
    ):
        self.dataset_path = dataset_path
        self.slicing_config = slicing_config
        self.n_splits = n_splits
        self.output_root = output_root
        self.seed = seed
        self.ioa_threshold = ioa_threshold
        self.empty_tile_ratio = empty_tile_ratio
        self.val_ratio = val_ratio
        self.groups = groups

        self._target_size: Tuple[int, int] = slicing_config.tile_size  # (640, 640)
        self._slicer = make_slicer(slicing_config.slicing_mode, slicing_config.overlap_ratio)
        self._is_asahi = isinstance(self._slicer, Asahi)
        self._rng = np.random.default_rng(seed)

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
        images = [
            img for img in self._coco["images"]
            if self._ann_by_image.get(img["id"])
            and os.path.isfile(os.path.join(self.dataset_path, img["file_name"]))
        ]
        stems = [Path(img["file_name"]).stem for img in images]
        duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
        if duplicates:
            raise ValueError(
                "Image filenames must have unique stems to create YOLO labels; "
                f"duplicates: {duplicates}"
            )
        return images

    def _make_splits(
        self, images: List[dict]
    ) -> List[Tuple[List[dict], List[dict], List[dict]]]:
        """Returns list of (train, val, test) triplets for each fold."""
        n = len(images)

        if self.groups is not None:
            raw = self._group_kfold(images)
        else:
            raw = self._kfold(n)

        # raw yields (train_idx, test_idx); carve val from the non-test pool
        splits = []
        for train_idx, test_idx in raw:
            pool = [images[i] for i in train_idx]
            test_imgs = [images[i] for i in test_idx]
            n_val = max(1, round(len(pool) * self.val_ratio))
            val_imgs = pool[:n_val]
            train_imgs = pool[n_val:]
            splits.append((train_imgs, val_imgs, test_imgs))

        return splits

    def _kfold(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(n)
        np.random.default_rng(self.seed).shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        chunks, cur = [], 0
        for size in fold_sizes:
            chunks.append(indices[cur : cur + size])
            cur += size
        return [
            (np.concatenate([chunks[j] for j in range(self.n_splits) if j != i]), chunks[i])
            for i in range(self.n_splits)
        ]

    def _group_kfold(self, images: List[dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        group_labels = [self.groups.get(img["file_name"], img["file_name"]) for img in images]
        unique_groups = list(dict.fromkeys(group_labels))
        group_sizes = {g: sum(1 for lbl in group_labels if lbl == g) for g in unique_groups}
        fold_counts = [0] * self.n_splits
        group_to_fold: Dict[str, int] = {}
        for g in sorted(unique_groups, key=lambda g: -group_sizes[g]):
            smallest = int(np.argmin(fold_counts))
            group_to_fold[g] = smallest
            fold_counts[smallest] += group_sizes[g]
        fold_indices: List[List[int]] = [[] for _ in range(self.n_splits)]
        for idx, g in enumerate(group_labels):
            fold_indices[group_to_fold[g]].append(idx)
        return [
            (
                np.array([idx for f2, chunk in enumerate(fold_indices) if f2 != f for idx in chunk]),
                np.array(fold_indices[f]),
            )
            for f in range(self.n_splits)
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
    # Letterbox (FI)                                                       #
    # ------------------------------------------------------------------ #

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
        """Resize mantendo proporção; preenche bordas com cinza 114."""
        h, w = img.shape[:2]
        tw, th = self._target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_x = (tw - nw) // 2
        pad_y = (th - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, pad_x, pad_y, scale

    def _fi_yolo_lines(
        self, img_meta: dict, pad_x: int, pad_y: int, scale: float
    ) -> List[str]:
        """COCO bbox → YOLO normalizado no espaço do canvas letterboxado."""
        tw, th = self._target_size
        lines = []
        for ann in self._ann_by_image.get(img_meta["id"], []):
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx = float(np.clip(((bx + bw / 2) * scale + pad_x) / tw, 0.0, 1.0))
            cy = float(np.clip(((by + bh / 2) * scale + pad_y) / th, 0.0, 1.0))
            w_n = float(np.clip(bw * scale / tw, 0.0, 1.0))
            h_n = float(np.clip(bh * scale / th, 0.0, 1.0))
            cls = self._category_map[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}")
        return lines

    # ------------------------------------------------------------------ #
    # Tile collection (train) — sem I/O                                   #
    # ------------------------------------------------------------------ #

    def _collect_tiles(
        self, img_meta: dict
    ) -> Optional[Tuple[List[_AnnotatedTile], List[_EmptyTile], ImageMetrics]]:
        """
        Lê a imagem, gera FI + tiles, classifica em annotated/empty.
        Não escreve nada em disco — retorna listas para amostragem global.
        """
        img_path = os.path.join(self.dataset_path, img_meta["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            return None

        img_h, img_w = image.shape[:2]
        annotations = self._ann_by_image.get(img_meta["id"], [])
        stem = Path(img_meta["file_name"]).stem
        surviving_ann_ids: set = set()

        annotated: List[_AnnotatedTile] = []
        empty: List[_EmptyTile] = []

        t0 = time.perf_counter()

        # FI: letterbox da imagem original inteira
        lb_img, pad_x, pad_y, scale = self._letterbox(image)
        fi_lines = self._fi_yolo_lines(img_meta, pad_x, pad_y, scale)
        annotated.append((lb_img, f"{stem}_fi", fi_lines))

        # Tiles
        tw, th = self._target_size
        for tile, coords in self._slicer.generate_tiles(image):
            x_off, y_off, p = coords["x"], coords["y"], coords["width"]
            tile_stem = f"{stem}_tile_{x_off}_{y_off}"

            # ASAHI: resize com interpolação inteligente baseada na direção de escala
            if self._is_asahi:
                interp = cv2.INTER_AREA if p > tw else cv2.INTER_CUBIC
                tile = cv2.resize(tile, (tw, th), interpolation=interp)

            yolo_lines: List[str] = []
            for ann in annotations:
                ioa, clipped = self._ioa(tuple(ann["bbox"]), x_off, y_off, p)
                if ioa < self.ioa_threshold:
                    continue
                surviving_ann_ids.add(ann["id"])
                cls = self._category_map[ann["category_id"]]
                yolo_lines.append(self._yolo_line(clipped, p, cls))

            if yolo_lines:
                annotated.append((tile, tile_stem, yolo_lines))
            else:
                empty.append((tile, tile_stem))

        slicing_time_ms = (time.perf_counter() - t0) * 1000
        self._get_geometry(img_w, img_h)

        annotations_original = len(annotations)
        annotations_kept = len(surviving_ann_ids)

        metrics = ImageMetrics(
            image_name=img_meta["file_name"],
            width=img_w,
            height=img_h,
            tiles_generated=len(annotated),  # actualizado após amostragem global
            slicing_time_ms=slicing_time_ms,
            annotations_original=annotations_original,
            annotations_kept=annotations_kept,
            annotations_discarded=annotations_original - annotations_kept,
        )
        return annotated, empty, metrics

    # ------------------------------------------------------------------ #
    # Holdout split (val / test) — originais sem tiling                   #
    # ------------------------------------------------------------------ #

    def _write_holdout_split(
        self, images: List[dict], split_dir: str, desc: str = ""
    ) -> List[ImageMetrics]:
        """
        Copia originais intactos + labels YOLO normalizados pela resolução original.
        Sem tiling, sem resize — ultralytics aplica letterbox internamente na avaliação.
        """
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        metrics: List[ImageMetrics] = []
        for img_meta in tqdm(images, desc=f"    {desc}", unit="img", leave=False):
            src = os.path.join(self.dataset_path, img_meta["file_name"])
            if not os.path.isfile(src):
                continue

            shutil.copy2(src, os.path.join(images_dir, img_meta["file_name"]))

            img_w = img_meta.get("width") or 0
            img_h = img_meta.get("height") or 0

            # Fallback: ler dimensões do disco se ausentes no JSON
            if not img_w or not img_h:
                probe = cv2.imread(src)
                if probe is not None:
                    img_h, img_w = probe.shape[:2]

            anns = self._ann_by_image.get(img_meta["id"], [])
            lines: List[str] = []
            if img_w and img_h:
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]
                    if bw <= 0 or bh <= 0:
                        continue
                    cx = float(np.clip((bx + bw / 2) / img_w, 0.0, 1.0))
                    cy = float(np.clip((by + bh / 2) / img_h, 0.0, 1.0))
                    w_n = float(np.clip(bw / img_w, 0.0, 1.0))
                    h_n = float(np.clip(bh / img_h, 0.0, 1.0))
                    cls = self._category_map[ann["category_id"]]
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}")

            stem = Path(img_meta["file_name"]).stem
            with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(lines))

            metrics.append(ImageMetrics(
                image_name=img_meta["file_name"],
                width=img_w,
                height=img_h,
                tiles_generated=1,
                slicing_time_ms=0.0,
                annotations_original=len(anns),
                annotations_kept=len(lines),
                annotations_discarded=0,
            ))
        return metrics

    # ------------------------------------------------------------------ #
    # Train split — buffer global + escrita consolidada                   #
    # ------------------------------------------------------------------ #

    def _process_split(
        self, images: List[dict], split_dir: str, desc: str = ""
    ) -> List[ImageMetrics]:
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        raw_metrics: List[ImageMetrics] = []
        empty_candidates: List[Tuple[str, str, int]] = []
        annotated_count = 0
        empty_dir = tempfile.mkdtemp(prefix=".empty_tiles_", dir=split_dir)

        try:
            for img_meta in tqdm(images, desc=f"    {desc}", unit="img", leave=False):
                result = self._collect_tiles(img_meta)
                if result is None:
                    continue
                annotated, empty, metrics = result
                metric_index = len(raw_metrics)
                raw_metrics.append(metrics)

                # Annotated samples can be persisted immediately. This bounds
                # memory usage to the tiles of one source image.
                for arr, stem, lines in annotated:
                    cv2.imwrite(os.path.join(images_dir, f"{stem}.jpg"), arr)
                    with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
                        f.write("\n".join(lines))
                annotated_count += len(annotated)

                # Empty samples are staged on disk until the global sample size
                # is known; keeping every empty tile in RAM can exhaust memory.
                for candidate_index, (arr, stem) in enumerate(empty):
                    temp_name = f"{metric_index}_{candidate_index}_{stem}.jpg"
                    temp_path = os.path.join(empty_dir, temp_name)
                    cv2.imwrite(temp_path, arr)
                    empty_candidates.append((temp_path, stem, metric_index))

            n_keep = min(
                math.ceil(annotated_count * self.empty_tile_ratio),
                len(empty_candidates),
            )
            chosen = set(
                self._rng.choice(len(empty_candidates), n_keep, replace=False).tolist()
                if n_keep > 0 else []
            )

            for candidate_index, (temp_path, stem, metric_index) in enumerate(empty_candidates):
                metrics = raw_metrics[metric_index]
                if candidate_index in chosen:
                    shutil.move(temp_path, os.path.join(images_dir, f"{stem}.jpg"))
                    open(os.path.join(labels_dir, f"{stem}.txt"), "w").close()
                    metrics.empty_tiles_kept += 1
                    metrics.tiles_generated += 1
                else:
                    metrics.empty_tiles_discarded += 1
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

        return raw_metrics

    # ------------------------------------------------------------------ #
    # YAML                                                                 #
    # ------------------------------------------------------------------ #

    def _write_yaml(self, fold_index: int, train_dir: str, val_dir: str, test_dir: str):
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
            "test": str((Path(test_dir) / "images").resolve()),
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
        seen: Dict[str, ImageMetrics] = {}
        for fold in folds:
            for m in fold.train_metrics + fold.val_metrics + fold.test_metrics:
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
                if (w, h) not in self._geometry_cache:
                    continue
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
            m for fold in folds for m in fold.train_metrics + fold.val_metrics + fold.test_metrics
        ]
        total_tiles_all_folds = sum(m.tiles_generated for m in all_fold_metrics)
        slicing_times = [m.slicing_time_ms for m in unique_metrics]
        orig_annotations = sum(m.annotations_original for m in unique_metrics)
        discarded = sum(m.annotations_discarded for m in unique_metrics)
        total_empty_kept = sum(m.empty_tiles_kept for m in unique_metrics)
        total_empty_discarded = sum(m.empty_tiles_discarded for m in unique_metrics)

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
            "tile_filtering": {
                "empty_tile_ratio": self.empty_tile_ratio,
                "empty_tiles_kept": total_empty_kept,
                "empty_tiles_discarded": total_empty_discarded,
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

    def _stats_path(self, fold_index: int) -> str:
        return os.path.join(self.output_root, f"fold_{fold_index}_stats.json")

    def _save_fold_stats(self, stats: FoldStats):
        with open(self._stats_path(stats.fold), "w", encoding="utf-8") as f:
            json.dump(stats.to_full_dict(), f, ensure_ascii=False)

    def _load_fold_stats(self, fold_index: int) -> Optional[FoldStats]:
        path = self._stats_path(fold_index)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return FoldStats.from_full_dict(json.load(f))

    def generate_fold(
        self,
        fold_index: int,
        train_images: List[dict],
        val_images: List[dict],
        test_images: Optional[List[dict]] = None,
    ) -> FoldStats:
        """Materialises one fold on disk and writes its fold_{i}.yaml."""
        fold_dir = os.path.join(self.output_root, f"fold_{fold_index}")
        if os.path.isdir(fold_dir):
            shutil.rmtree(fold_dir)
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")
        test_dir = os.path.join(fold_dir, "test")

        train_metrics = self._process_split(
            train_images, train_dir, f"fold {fold_index} train"
        )
        val_metrics = self._write_holdout_split(
            val_images, val_dir, f"fold {fold_index} val  "
        )
        test_metrics = self._write_holdout_split(
            test_images or [], test_dir, f"fold {fold_index} test "
        )
        self._write_yaml(fold_index, train_dir, val_dir, test_dir)

        stats = FoldStats(
            fold=fold_index,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        self._save_fold_stats(stats)
        return stats

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

    def run(self, resume_from: int = 1) -> List[FoldStats]:
        """Generates all folds on disk and writes experiment reports.

        resume_from: skip folds with index < resume_from (1-based). Useful
        when the pipeline was interrupted mid-run; the caller is responsible
        for deleting any partial fold directory before resuming.
        """
        images = self._valid_images()
        if len(images) < self.n_splits:
            raise ValueError(
                f"Dataset has {len(images)} images but {self.n_splits} splits were requested."
            )

        os.makedirs(self.output_root, exist_ok=True)
        results: List[FoldStats] = []

        for fold_index, (train_imgs, val_imgs, test_imgs) in enumerate(
            self._make_splits(images), start=1
        ):
            if fold_index < resume_from:
                stats = self._load_fold_stats(fold_index)
                if stats is None:
                    raise FileNotFoundError(
                        f"fold_{fold_index}_stats.json not found in {self.output_root} — "
                        f"run scripts/reconstruct_fold_stats.py first."
                    )
                print(f"[Fold {fold_index}/{self.n_splits}] Skipping (loaded from disk).")
                for m in stats.train_metrics + stats.val_metrics + stats.test_metrics:
                    if m.width and m.height:
                        self._get_geometry(m.width, m.height)
                results.append(stats)
                continue

            print(f"[Fold {fold_index}/{self.n_splits}] Generating tiles...")
            stats = self.generate_fold(fold_index, train_imgs, val_imgs, test_imgs)
            print(
                f"  train {stats.train_tiles} tiles / {stats.train_images} images  |  "
                f"val {stats.val_tiles} imgs / {stats.val_images} images  |  "
                f"test {stats.test_tiles} imgs / {stats.test_images} images"
            )
            results.append(stats)

        self._write_reports(results)
        return results
