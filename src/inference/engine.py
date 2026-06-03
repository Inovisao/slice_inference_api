from typing import Generator, Tuple
import numpy as np
from ultralytics import YOLO

class TileInferenceEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.class_names: dict = self.model.names

    def predict_tiles(
        self,
        image: np.ndarray,
        tile_generator: Generator[Tuple[np.ndarray, dict], None, None],
        conf_thr: float = 0.25,
    ) -> tuple[list, list, list]:
        img_h, img_w = image.shape[:2]
        raw_boxes, raw_scores, raw_labels = [], [], []

        for tile, coords in tile_generator:
            x_off, y_off = coords["x"], coords["y"]
            results = self.model.predict(tile, conf=conf_thr, verbose=False)

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    gx1 = max(0.0, min(1.0, (bx1 + x_off) / img_w))
                    gy1 = max(0.0, min(1.0, (by1 + y_off) / img_h))
                    gx2 = max(0.0, min(1.0, (bx2 + x_off) / img_w))
                    gy2 = max(0.0, min(1.0, (by2 + y_off) / img_h))
                    if gx2 > gx1 and gy2 > gy1:
                        raw_boxes.append([gx1, gy1, gx2, gy2])
                        raw_scores.append(float(box.conf[0]))
                        raw_labels.append(int(box.cls[0]))

        return raw_boxes, raw_scores, raw_labels

    def predict_full_image(
        self, image: np.ndarray, conf_thr: float = 0.25
    ) -> tuple[list, list, list]:
        img_h, img_w = image.shape[:2]
        results = self.model.predict(image, conf=conf_thr, verbose=False)
        boxes, scores, labels = [], [], []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                gx1 = max(0.0, min(1.0, bx1 / img_w))
                gy1 = max(0.0, min(1.0, by1 / img_h))
                gx2 = max(0.0, min(1.0, bx2 / img_w))
                gy2 = max(0.0, min(1.0, by2 / img_h))
                if gx2 > gx1 and gy2 > gy1:
                    boxes.append([gx1, gy1, gx2, gy2])
                    scores.append(float(box.conf[0]))
                    labels.append(int(box.cls[0]))
        return boxes, scores, labels
