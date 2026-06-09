import json
import os
from dataclasses import dataclass, field
from typing import List


_COCO_FILENAME = "_annotations.coco.json"
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "errors"
        if self.warnings:
            return "warnings"
        return "ok"

    def to_dict(self, summary: dict) -> dict:
        return {
            "status": self.status,
            "summary": summary,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DatasetValidator:
    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.coco_path = os.path.join(dataset_path, _COCO_FILENAME)

    def validate(self) -> dict:
        report = ValidationReport()

        if not os.path.isdir(self.path):
            report.errors.append(f"Dataset directory not found: '{self.path}'")
            return report.to_dict(self._empty_summary())

        if not os.path.isfile(self.coco_path):
            report.errors.append(f"Annotation file not found: '{self.coco_path}'")
            return report.to_dict(self._empty_summary())

        try:
            with open(self.coco_path) as f:
                coco = json.load(f)
        except json.JSONDecodeError as e:
            report.errors.append(f"Invalid JSON in annotation file: {e}")
            return report.to_dict(self._empty_summary())

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        self._check_structure(coco, report)
        self._check_categories(categories, report)

        image_index = self._check_images(images, report)
        category_ids = {c["id"] for c in categories}

        self._check_annotations(annotations, image_index, category_ids, report)
        self._check_orphan_images(images, report)

        disk_images = [
            f for f in os.listdir(self.path)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        ]
        referenced_files = {img["file_name"] for img in images}

        summary = {
            "images_in_json": len(images),
            "images_on_disk": len(disk_images),
            "annotations": len(annotations),
            "categories": len(categories),
            "missing_files": sum(
                1 for img in images
                if not os.path.isfile(os.path.join(self.path, img["file_name"]))
            ),
            "orphan_files": sum(
                1 for f in disk_images if f not in referenced_files
            ),
        }

        return report.to_dict(summary)

    def _empty_summary(self) -> dict:
        return {
            "images_in_json": 0,
            "images_on_disk": 0,
            "annotations": 0,
            "categories": 0,
            "missing_files": 0,
            "orphan_files": 0,
        }

    def _check_structure(self, coco: dict, report: ValidationReport):
        for key in ("images", "annotations", "categories"):
            if key not in coco:
                report.errors.append(f"Missing required COCO key: '{key}'")

    def _check_categories(self, categories: list, report: ValidationReport):
        if not categories:
            report.warnings.append("No categories defined in annotation file")
            return

        seen_ids = set()
        for cat in categories:
            cid = cat.get("id")
            if cid in seen_ids:
                report.errors.append(f"Duplicate category id: {cid}")
            seen_ids.add(cid)
            if not cat.get("name"):
                report.warnings.append(f"Category id={cid} has no name")

    def _check_images(self, images: list, report: ValidationReport) -> dict:
        image_index = {}
        seen_ids = set()

        for img in images:
            iid = img.get("id")
            fname = img.get("file_name", "")

            if iid in seen_ids:
                report.errors.append(f"Duplicate image id: {iid}")
            seen_ids.add(iid)

            if not fname:
                report.errors.append(f"Image id={iid} has empty file_name")
                continue

            full_path = os.path.join(self.path, fname)
            if not os.path.isfile(full_path):
                report.errors.append(f"Missing file on disk: '{fname}' (image_id={iid})")
            else:
                image_index[iid] = img

            if not img.get("width") or not img.get("height"):
                report.warnings.append(
                    f"Image '{fname}' (id={iid}) has no width/height in JSON"
                )

        return image_index

    def _check_annotations(
        self,
        annotations: list,
        image_index: dict,
        category_ids: set,
        report: ValidationReport,
    ):
        seen_ids = set()

        for ann in annotations:
            aid = ann.get("id")
            iid = ann.get("image_id")
            cid = ann.get("category_id")
            bbox = ann.get("bbox", [])

            if aid in seen_ids:
                report.errors.append(f"Duplicate annotation id: {aid}")
            seen_ids.add(aid)

            if iid not in image_index:
                report.errors.append(
                    f"Annotation id={aid} references unknown image_id={iid}"
                )
                continue

            if category_ids and cid not in category_ids:
                report.errors.append(
                    f"Annotation id={aid} references unknown category_id={cid}"
                )

            if len(bbox) != 4:
                report.errors.append(
                    f"Annotation id={aid} has malformed bbox: {bbox}"
                )
                continue

            x, y, w, h = bbox
            img_meta = image_index[iid]
            img_w = img_meta.get("width", 0)
            img_h = img_meta.get("height", 0)

            if w <= 0 or h <= 0:
                report.errors.append(
                    f"Annotation id={aid} has degenerate bbox (w={w}, h={h})"
                )

            if x < 0 or y < 0:
                report.warnings.append(
                    f"Annotation id={aid} has negative coordinates (x={x}, y={y})"
                )

            if img_w and img_h:
                if x + w > img_w or y + h > img_h:
                    report.warnings.append(
                        f"Annotation id={aid} bbox exceeds image bounds "
                        f"[x={x}, y={y}, w={w}, h={h}] vs [{img_w}×{img_h}]"
                    )

            declared_area = ann.get("area")
            if declared_area is not None:
                expected_area = w * h
                if abs(declared_area - expected_area) > 1.0:
                    report.warnings.append(
                        f"Annotation id={aid} area mismatch: "
                        f"declared={declared_area:.1f}, computed={expected_area:.1f}"
                    )

    def _check_orphan_images(self, images: list, report: ValidationReport):
        referenced = {img["file_name"] for img in images if img.get("file_name")}
        disk_files = [
            f for f in os.listdir(self.path)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        ]
        for fname in disk_files:
            if fname not in referenced:
                report.warnings.append(
                    f"Image on disk not referenced in annotations: '{fname}'"
                )
