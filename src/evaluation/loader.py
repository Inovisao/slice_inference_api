import os
from typing import List, Tuple

import cv2
import numpy as np


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FoldTestLoader:
    """Loads original test images and YOLO-format GT labels from a fold test directory."""

    def __init__(self, test_dir: str):
        self.images_dir = os.path.join(test_dir, "images")
        self.labels_dir = os.path.join(test_dir, "labels")

    def list_images(self) -> List[str]:
        return sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        ])

    def load_image(self, image_name: str) -> Tuple[np.ndarray, str]:
        path = os.path.join(self.images_dir, image_name)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img, path

    def load_gt_boxes(self, image_name: str) -> np.ndarray:
        """Returns GT boxes as normalized xyxy [N, 4]. YOLO cx cy w h → xyxy."""
        stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.labels_dir, stem + ".txt")

        if not os.path.isfile(label_path):
            return np.zeros((0, 4), dtype=np.float32)

        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
