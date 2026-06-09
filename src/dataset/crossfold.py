import json
import os
import random
from dataclasses import dataclass, field
from typing import List


_CLEAN_FILENAME = "_annotations_clean.coco.json"
_FALLBACK_FILENAME = "_annotations.coco.json"


@dataclass
class FoldReport:
    n_folds: int
    val_ratio: float
    folds: List[dict] = field(default_factory=list)
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "n_folds": self.n_folds,
            "val_ratio": self.val_ratio,
            "output_dir": self.output_dir,
            "folds": self.folds,
        }


class CrossFoldSplitter:
    def __init__(self, dataset_path: str, output_dir: str | None = None):
        self.dataset_path = dataset_path
        self.output_dir = output_dir or os.path.join(dataset_path, "filesJSON")

        clean = os.path.join(dataset_path, _CLEAN_FILENAME)
        fallback = os.path.join(dataset_path, _FALLBACK_FILENAME)
        self.coco_path = clean if os.path.isfile(clean) else fallback

    def split(self, n_folds: int = 5, val_ratio: float = 0.3, seed: int = 42) -> dict:
        with open(self.coco_path, encoding="utf-8") as f:
            coco = json.load(f)

        info = coco.get("info", {})
        licenses = coco.get("licenses", [])
        categories = coco["categories"]
        images = coco["images"]
        annotations = coco["annotations"]

        ann_by_image: dict[int, list] = {}
        for ann in annotations:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # keep only images that exist on disk and have at least one annotation
        images = [
            img for img in images
            if ann_by_image.get(img["id"])
            and os.path.isfile(os.path.join(self.dataset_path, img["file_name"]))
        ]

        shuffled = images.copy()
        random.seed(seed)
        random.shuffle(shuffled)

        os.makedirs(self.output_dir, exist_ok=True)
        for f in os.listdir(self.output_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.output_dir, f))

        fold_size = len(shuffled) // n_folds
        groups = [
            shuffled[i * fold_size: (i + 1) * fold_size]
            for i in range(n_folds - 1)
        ]
        groups.append(shuffled[(n_folds - 1) * fold_size:])  # last fold gets remainder

        report = FoldReport(n_folds=n_folds, val_ratio=val_ratio, output_dir=self.output_dir)

        for i in range(n_folds):
            test_imgs = groups[i]
            train_val_imgs = [img for j, g in enumerate(groups) if j != i for img in g]

            val_size = max(1, round(len(train_val_imgs) * val_ratio))
            val_imgs = train_val_imgs[:val_size]
            train_imgs = train_val_imgs[val_size:]

            prefix = os.path.join(self.output_dir, f"fold_{i + 1}")
            self._save(f"{prefix}_train.json", info, licenses, categories, train_imgs, ann_by_image)
            self._save(f"{prefix}_val.json",   info, licenses, categories, val_imgs,   ann_by_image)
            self._save(f"{prefix}_test.json",  info, licenses, categories, test_imgs,  ann_by_image)

            report.folds.append({
                "fold": i + 1,
                "train": len(train_imgs),
                "val": len(val_imgs),
                "test": len(test_imgs),
                "files": {
                    "train": f"fold_{i + 1}_train.json",
                    "val": f"fold_{i + 1}_val.json",
                    "test": f"fold_{i + 1}_test.json",
                },
            })

        return report.to_dict()

    def _save(
        self,
        path: str,
        info: dict,
        licenses: list,
        categories: list,
        images: list,
        ann_by_image: dict,
    ):
        image_ids = {img["id"] for img in images}
        annotations = [
            ann for iid in image_ids for ann in ann_by_image.get(iid, [])
        ]
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "info": info,
                    "licenses": licenses,
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                f,
                indent=2,
                sort_keys=True,
            )
