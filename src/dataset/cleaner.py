import json
import os
from dataclasses import dataclass, field
from typing import List


_COCO_FILENAME = "_annotations.coco.json"
_CLEAN_FILENAME = "_annotations_clean.coco.json"


@dataclass
class CleanReport:
    kept: int = 0
    removed_no_file: List[str] = field(default_factory=list)
    removed_no_annotation: List[str] = field(default_factory=list)
    annotations_kept: int = 0
    annotations_removed: int = 0
    output_path: str = ""

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "kept_images": self.kept,
            "removed_no_file": {
                "count": len(self.removed_no_file),
                "files": self.removed_no_file,
            },
            "removed_no_annotation": {
                "count": len(self.removed_no_annotation),
                "files": self.removed_no_annotation,
            },
            "annotations_kept": self.annotations_kept,
            "annotations_removed": self.annotations_removed,
            "output": self.output_path,
        }


class DatasetCleaner:
    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.coco_path = os.path.join(dataset_path, _COCO_FILENAME)
        self.output_path = os.path.join(dataset_path, _CLEAN_FILENAME)

    def clean(self) -> dict:
        with open(self.coco_path) as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]

        ann_by_image: dict[int, list] = {}
        for ann in annotations:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        report = CleanReport()
        kept_ids: set[int] = set()

        for img in images:
            iid = img["id"]
            fname = img.get("file_name", "")
            exists = os.path.isfile(os.path.join(self.path, fname))
            has_annotation = bool(ann_by_image.get(iid))

            if not exists:
                report.removed_no_file.append(fname)
            elif not has_annotation:
                report.removed_no_annotation.append(fname)
            else:
                kept_ids.add(iid)

        clean_images = [img for img in images if img["id"] in kept_ids]
        clean_annotations = [ann for ann in annotations if ann["image_id"] in kept_ids]

        report.kept = len(clean_images)
        report.annotations_kept = len(clean_annotations)
        report.annotations_removed = len(annotations) - len(clean_annotations)
        report.output_path = self.output_path

        clean_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": clean_images,
            "annotations": clean_annotations,
        }

        with open(self.output_path, "w") as f:
            json.dump(clean_coco, f, indent=2)

        return report.to_dict()
