import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2

_COCO_FILENAME = "_annotations.coco.json"
_CLEAN_FILENAME = "_annotations_clean.coco.json"


@dataclass
class PreprocessReport:
    images_original: int = 0
    images_removed_no_file: int = 0
    images_removed_no_annotation: int = 0
    images_dimensions_filled: int = 0
    images_kept: int = 0

    annotations_original: int = 0
    annotations_removed_unknown_image: int = 0
    annotations_removed_unknown_category: int = 0
    annotations_removed_malformed_bbox: int = 0
    annotations_removed_degenerate: int = 0
    annotations_clamped: int = 0
    annotations_area_fixed: int = 0
    annotations_kept: int = 0

    output_path: str = ""

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "output": self.output_path,
            "images": {
                "original": self.images_original,
                "removed_no_file": self.images_removed_no_file,
                "removed_no_annotation": self.images_removed_no_annotation,
                "dimensions_filled_from_disk": self.images_dimensions_filled,
                "kept": self.images_kept,
            },
            "annotations": {
                "original": self.annotations_original,
                "removed_unknown_image": self.annotations_removed_unknown_image,
                "removed_unknown_category": self.annotations_removed_unknown_category,
                "removed_malformed_bbox": self.annotations_removed_malformed_bbox,
                "removed_degenerate_after_clamp": self.annotations_removed_degenerate,
                "clamped_to_bounds": self.annotations_clamped,
                "area_recalculated": self.annotations_area_fixed,
                "kept": self.annotations_kept,
            },
        }


class DatasetPreprocessor:
    """
    Validates and normalises a COCO dataset before fold generation.

    Normalisations applied (in order):
      1. Fill missing width/height on image entries by reading from disk.
      2. Remove images whose file is missing from disk.
      3. Drop annotations referencing unknown image_id or category_id.
      4. Drop annotations with malformed bbox (not 4 numbers).
      5. Clamp bbox to image bounds (negative origin → 0, overflow → edge).
      6. Drop annotations that are degenerate after clamping (w≤0 or h≤0).
      7. Recalculate area field to match w*h.
      8. Remove images with no surviving annotations.

    Writes the result to _annotations_clean.coco.json (read by AsahiKFoldValidator).
    The original _annotations.coco.json is never modified.
    """

    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.coco_path = os.path.join(dataset_path, _COCO_FILENAME)
        self.output_path = os.path.join(dataset_path, _CLEAN_FILENAME)

    def run(self) -> dict:
        if not os.path.isfile(self.coco_path):
            raise FileNotFoundError(
                f"Annotation file not found: {self.coco_path}"
            )

        with open(self.coco_path, encoding="utf-8") as f:
            coco = json.load(f)

        images: List[dict] = [dict(img) for img in coco.get("images", [])]
        annotations: List[dict] = [dict(ann) for ann in coco.get("annotations", [])]
        categories: List[dict] = coco.get("categories", [])

        report = PreprocessReport(
            images_original=len(images),
            annotations_original=len(annotations),
            output_path=self.output_path,
        )

        valid_category_ids: Set[int] = {cat["id"] for cat in categories}

        # Step 1 — fill missing width/height from disk
        images = self._fill_dimensions(images, report)

        # Step 2 — remove images missing from disk
        images, on_disk_ids = self._filter_missing_files(images, report)

        # Step 3-7 — normalise annotations
        annotations = self._normalise_annotations(
            annotations, on_disk_ids, valid_category_ids, images, report
        )

        # Step 8 — remove images with no surviving annotations
        annotated_ids = {ann["image_id"] for ann in annotations}
        kept_images = []
        for img in images:
            if img["id"] in annotated_ids:
                kept_images.append(img)
            else:
                report.images_removed_no_annotation += 1

        report.images_kept = len(kept_images)
        report.annotations_kept = len(annotations)

        clean_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": categories,
            "images": kept_images,
            "annotations": annotations,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(clean_coco, f, indent=2, ensure_ascii=False)

        return report.to_dict()

    # ------------------------------------------------------------------ #
    # Step 1 — fill width/height                                          #
    # ------------------------------------------------------------------ #

    def _fill_dimensions(
        self, images: List[dict], report: PreprocessReport
    ) -> List[dict]:
        for img in images:
            if img.get("width") and img.get("height"):
                continue
            full_path = os.path.join(self.path, img.get("file_name", ""))
            if not os.path.isfile(full_path):
                continue
            frame = cv2.imread(full_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            img["width"] = w
            img["height"] = h
            report.images_dimensions_filled += 1
        return images

    # ------------------------------------------------------------------ #
    # Step 2 — remove missing files                                       #
    # ------------------------------------------------------------------ #

    def _filter_missing_files(
        self, images: List[dict], report: PreprocessReport
    ) -> Tuple[List[dict], Set[int]]:
        kept = []
        on_disk_ids: Set[int] = set()
        for img in images:
            fname = img.get("file_name", "")
            if os.path.isfile(os.path.join(self.path, fname)):
                kept.append(img)
                on_disk_ids.add(img["id"])
            else:
                report.images_removed_no_file += 1
        return kept, on_disk_ids

    # ------------------------------------------------------------------ #
    # Steps 3-7 — normalise annotations                                   #
    # ------------------------------------------------------------------ #

    def _normalise_annotations(
        self,
        annotations: List[dict],
        valid_image_ids: Set[int],
        valid_category_ids: Set[int],
        images: List[dict],
        report: PreprocessReport,
    ) -> List[dict]:
        dims: Dict[int, Tuple[int, int]] = {
            img["id"]: (img.get("width", 0), img.get("height", 0))
            for img in images
        }

        kept = []
        for ann in annotations:
            iid = ann.get("image_id")
            cid = ann.get("category_id")
            bbox = ann.get("bbox", [])

            # Step 3a — unknown image
            if iid not in valid_image_ids:
                report.annotations_removed_unknown_image += 1
                continue

            # Step 3b — unknown category
            if valid_category_ids and cid not in valid_category_ids:
                report.annotations_removed_unknown_category += 1
                continue

            # Step 4 — malformed bbox
            if len(bbox) != 4:
                report.annotations_removed_malformed_bbox += 1
                continue

            ann = dict(ann)
            x, y, w, h = map(float, bbox)
            img_w, img_h = dims.get(iid, (0, 0))

            # Step 5 — clamp to image bounds
            if img_w and img_h:
                x2, y2 = x + w, y + h
                cx = max(0.0, min(x, img_w))
                cy = max(0.0, min(y, img_h))
                cx2 = max(0.0, min(x2, img_w))
                cy2 = max(0.0, min(y2, img_h))
                cw, ch = cx2 - cx, cy2 - cy

                clamped = (cx != x or cy != y or cw != w or ch != h)
                x, y, w, h = cx, cy, cw, ch

                if clamped:
                    report.annotations_clamped += 1

            # Step 6 — degenerate after clamp
            if w <= 0 or h <= 0:
                report.annotations_removed_degenerate += 1
                continue

            # Step 7 — fix area
            computed_area = w * h
            declared_area = ann.get("area")
            if declared_area is None or abs(declared_area - computed_area) > 1.0:
                ann["area"] = computed_area
                report.annotations_area_fixed += 1

            ann["bbox"] = [x, y, w, h]
            kept.append(ann)

        return kept
