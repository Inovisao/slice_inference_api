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
        batch_size: int = 32,
    ) -> tuple[list, list, list]:
        img_h, img_w = image.shape[:2]
        raw_boxes, raw_scores, raw_labels = [], [], []

        batch_tiles: list[np.ndarray] = []
        batch_coords: list[dict] = []

        for tile, coords in tile_generator:
            batch_tiles.append(tile)
            batch_coords.append(coords)
            if len(batch_tiles) == batch_size:
                self._infer_batch(
                    batch_tiles, batch_coords, img_w, img_h, conf_thr,
                    raw_boxes, raw_scores, raw_labels,
                )
                batch_tiles, batch_coords = [], []

        if batch_tiles:
            self._infer_batch(
                batch_tiles, batch_coords, img_w, img_h, conf_thr,
                raw_boxes, raw_scores, raw_labels,
            )

        return raw_boxes, raw_scores, raw_labels

    def _infer_batch(
        self,
        tiles: list[np.ndarray],
        coords_list: list[dict],
        img_w: int,
        img_h: int,
        conf_thr: float,
        raw_boxes: list,
        raw_scores: list,
        raw_labels: list,
    ) -> None:
        results = self.model.predict(tiles, conf=conf_thr, imgsz=640, verbose=False)
        for r, coords in zip(results, coords_list):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            x_off, y_off = coords["x"], coords["y"]
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
