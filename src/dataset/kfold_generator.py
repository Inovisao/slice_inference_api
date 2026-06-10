import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold, KFold

from config.settings import SlicingConfig
from slicing.asahi import Asahi

_COCO_FILENAME = "_annotations.coco.json"
_CLEAN_FILENAME = "_annotations_clean.coco.json"
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_DEFAULT_IOA_THRESHOLD = 0.20


@dataclass
class FoldStats:
    fold: int
    train_images: int
    val_images: int
    train_tiles: int
    val_tiles: int
    train_labels: int
    val_labels: int

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "train_tiles": self.train_tiles,
            "val_tiles": self.val_tiles,
            "train_labels": self.train_labels,
            "val_labels": self.val_labels,
        }


class AsahiKFoldValidator:
    """
    Generates YOLO-ready k-fold datasets from high-resolution images using ASAHI
    adaptive tiling. Each fold materializes tiles on disk, writes a fold_{i}.yaml,
    and is cleaned up after training to prevent cross-contamination.

    Expects COCO annotations at:
        dataset_path/_annotations_clean.coco.json  (preferred)
        dataset_path/_annotations.coco.json        (fallback)

    Output layout:
        output_root/
          fold_{i}/
            train/images|labels
            val/images|labels
          fold_{i}.yaml
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

        self._slicer = Asahi(slicing_config)
        self._coco = self._load_coco()
        self._category_map = self._build_category_map()
        self._ann_by_image = self._index_annotations()

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
    ) -> Tuple[int, int]:
        """
        Slices one image with ASAHI, applies IoA filter, writes tiles and YOLO
        label files. Returns (tiles_written, annotations_written).
        """
        img_path = os.path.join(self.dataset_path, img_meta["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            return 0, 0

        annotations = self._ann_by_image.get(img_meta["id"], [])
        stem = Path(img_meta["file_name"]).stem
        tiles_written = 0
        annotations_written = 0

        for tile, coords in self._slicer.generate_tiles(image):
            x_off, y_off, p = coords["x"], coords["y"], coords["width"]
            tile_stem = f"{stem}_tile_{x_off}_{y_off}"

            yolo_lines: List[str] = []
            for ann in annotations:
                ioa, clipped = self._ioa(tuple(ann["bbox"]), x_off, y_off, p)
                if ioa < self.ioa_threshold:
                    continue
                cls = self._category_map[ann["category_id"]]
                yolo_lines.append(self._yolo_line(clipped, p, cls))

            cv2.imwrite(os.path.join(images_dir, f"{tile_stem}.jpg"), tile)
            tiles_written += 1

            with open(os.path.join(labels_dir, f"{tile_stem}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))
            annotations_written += len(yolo_lines)

        return tiles_written, annotations_written

    def _process_split(
        self, images: List[dict], split_dir: str
    ) -> Tuple[int, int]:
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        total_tiles = total_labels = 0
        for img in images:
            t, l = self._process_image(img, images_dir, labels_dir)
            total_tiles += t
            total_labels += l
        return total_tiles, total_labels

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

        train_tiles, train_labels = self._process_split(train_images, train_dir)
        val_tiles, val_labels = self._process_split(val_images, val_dir)
        self._write_yaml(fold_index, train_dir, val_dir)

        return FoldStats(
            fold=fold_index,
            train_images=len(train_images),
            val_images=len(val_images),
            train_tiles=train_tiles,
            val_tiles=val_tiles,
            train_labels=train_labels,
            val_labels=val_labels,
        )

    def cleanup_fold(self, fold_index: int):
        """
        Deletes the physical fold directory and any .cache files ultralytics
        may have written alongside the images/labels dirs.
        """
        fold_dir = os.path.join(self.output_root, f"fold_{fold_index}")
        if os.path.isdir(fold_dir):
            shutil.rmtree(fold_dir)

        # ultralytics silently drops .cache next to the dataset dirs
        for root, _dirs, files in os.walk(self.output_root):
            for fname in files:
                if fname.endswith(".cache"):
                    os.remove(os.path.join(root, fname))

    def run(self) -> List[FoldStats]:
        """Generates all folds on disk. Returns one FoldStats per fold."""
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
                f"  train {stats.train_tiles} tiles / {stats.train_labels} labels  |  "
                f"val {stats.val_tiles} tiles / {stats.val_labels} labels"
            )
            results.append(stats)

        return results
