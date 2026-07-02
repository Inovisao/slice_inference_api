from typing import Generator, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── YOLOv8 ──────────────────────────────────────────────────────────────────

class TileInferenceEngine:
    def __init__(self, model_path: str, device: str = "cuda"):
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
                self._infer_batch(batch_tiles, batch_coords, img_w, img_h, conf_thr,
                                  raw_boxes, raw_scores, raw_labels)
                batch_tiles, batch_coords = [], []
        if batch_tiles:
            self._infer_batch(batch_tiles, batch_coords, img_w, img_h, conf_thr,
                              raw_boxes, raw_scores, raw_labels)

        return raw_boxes, raw_scores, raw_labels

    def predict_full_image(
        self, image: np.ndarray, conf_thr: float = 0.25
    ) -> tuple[list, list, list]:
        """Runs one YOLO pass over the complete image."""
        img_h, img_w = image.shape[:2]
        boxes, scores, labels = [], [], []
        results = self.model.predict(image, conf=conf_thr, imgsz=640, verbose=False)
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

    def _infer_batch(self, tiles, coords_list, img_w, img_h, conf_thr,
                     raw_boxes, raw_scores, raw_labels):
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


# ─── Faster R-CNN ─────────────────────────────────────────────────────────────

class FasterInferenceEngine:
    _NUM_CLASSES = 2
    _RESIZE_TO   = 640

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model.to(self.device).eval()

    def _preprocess(self, frame: np.ndarray) -> tuple[torch.Tensor, float]:
        orig_h, orig_w = frame.shape[:2]
        scale = self._RESIZE_TO / max(orig_h, orig_w)
        if scale != 1.0:
            frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1), scale

    def _boxes_to_global(self, pred, scale, x_off, y_off, img_w, img_h,
                         conf_thr, raw_boxes, raw_scores, raw_labels):
        for box, score, label in zip(pred["boxes"].cpu().tolist(),
                                     pred["scores"].cpu().tolist(),
                                     pred["labels"].cpu().tolist()):
            if score < conf_thr:
                continue
            tx1, ty1, tx2, ty2 = box[0]/scale, box[1]/scale, box[2]/scale, box[3]/scale
            gx1 = max(0.0, min(1.0, (tx1 + x_off) / img_w))
            gy1 = max(0.0, min(1.0, (ty1 + y_off) / img_h))
            gx2 = max(0.0, min(1.0, (tx2 + x_off) / img_w))
            gy2 = max(0.0, min(1.0, (ty2 + y_off) / img_h))
            if gx2 > gx1 and gy2 > gy1:
                raw_boxes.append([gx1, gy1, gx2, gy2])
                raw_scores.append(float(score))
                raw_labels.append(int(label))

    def predict_tiles(
        self,
        image: np.ndarray,
        tile_generator,
        conf_thr: float = 0.25,
        batch_size: int = 32,
    ) -> tuple[list, list, list]:
        img_h, img_w = image.shape[:2]
        raw_boxes, raw_scores, raw_labels = [], [], []

        all_items = list(tile_generator)
        if not all_items:
            return raw_boxes, raw_scores, raw_labels
        tiles, coords_list = zip(*all_items)

        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            batch_coords = coords_list[i:i + batch_size]
            tensors, scales = zip(*[self._preprocess(t) for t in batch])
            tensors = [t.to(self.device, non_blocking=True) for t in tensors]
            with torch.no_grad():
                preds = self.model(tensors)
            for pred, coords, sc in zip(preds, batch_coords, scales):
                self._boxes_to_global(pred, sc,
                                      coords["x"], coords["y"],
                                      img_w, img_h,
                                      conf_thr, raw_boxes, raw_scores, raw_labels)

        return raw_boxes, raw_scores, raw_labels

    def predict_full_image(
        self, image: np.ndarray, conf_thr: float = 0.25
    ) -> tuple[list, list, list]:
        """Runs one Faster R-CNN pass over the complete image."""
        img_h, img_w = image.shape[:2]
        boxes, scores, labels = [], [], []
        tensor, scale = self._preprocess(image)
        with torch.no_grad():
            pred = self.model([tensor.to(self.device)])[0]
        self._boxes_to_global(
            pred, scale, 0, 0, img_w, img_h, conf_thr, boxes, scores, labels
        )
        return boxes, scores, labels


# ─── DETR ─────────────────────────────────────────────────────────────────────

class _DETRWrapper(nn.Module):
    """Thin wrapper that loads a DETR checkpoint with torch.compile prefixes stripped.
    The checkpoint uses DETR's original class_embed head (shape [N_classes, 256]).
    No extra linear is added — pred_logits is used as-is from the base model."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/detr", "detr_resnet50",
                                    pretrained=False, verbose=False)

    def forward(self, images):
        return self.model(images)


class DetrInferenceEngine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = _DETRWrapper()
        state = torch.load(model_path, map_location="cpu")
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"DETR checkpoint missing keys: {missing[:5]}")
        self.model = model.to(self.device).eval()
        # Number of output classes = class_embed output size minus the DETR no-object slot
        self._n_classes = model.model.class_embed.out_features - 1

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def _decode_batch(self, outputs, batch_coords, img_w, img_h, conf_thr,
                      raw_boxes, raw_scores, raw_labels):
        pred_logits = outputs["pred_logits"]
        pred_boxes  = outputs["pred_boxes"]
        for b_idx, coords in enumerate(batch_coords):
            probs = F.softmax(pred_logits[b_idx], dim=-1)
            # exclude the DETR no-object slot (last class) — use only object classes
            scores, labels = probs[..., :self._n_classes].max(dim=-1)
            keep = scores > conf_thr
            if not keep.any():
                continue
            x_off, y_off   = coords["x"], coords["y"]
            tile_w, tile_h = coords["width"], coords["height"]
            for score, label, box in zip(scores[keep], labels[keep],
                                         pred_boxes[b_idx][keep]):
                cx, cy, bw, bh = box.cpu().tolist()
                tx1 = (cx - bw / 2) * tile_w
                ty1 = (cy - bh / 2) * tile_h
                tx2 = (cx + bw / 2) * tile_w
                ty2 = (cy + bh / 2) * tile_h
                gx1 = max(0.0, min(1.0, (tx1 + x_off) / img_w))
                gy1 = max(0.0, min(1.0, (ty1 + y_off) / img_h))
                gx2 = max(0.0, min(1.0, (tx2 + x_off) / img_w))
                gy2 = max(0.0, min(1.0, (ty2 + y_off) / img_h))
                if gx2 > gx1 and gy2 > gy1:
                    raw_boxes.append([gx1, gy1, gx2, gy2])
                    raw_scores.append(float(score))
                    raw_labels.append(int(label))

    def predict_tiles(
        self,
        image: np.ndarray,
        tile_generator,
        conf_thr: float = 0.25,
        batch_size: int = 32,
    ) -> tuple[list, list, list]:
        img_h, img_w = image.shape[:2]
        raw_boxes, raw_scores, raw_labels = [], [], []

        all_items = list(tile_generator)
        if not all_items:
            return raw_boxes, raw_scores, raw_labels
        tiles, coords_list = zip(*all_items)

        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            batch_coords = coords_list[i:i + batch_size]
            tensors = torch.stack([self._preprocess(t) for t in batch]).to(self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(tensors)
            self._decode_batch(outputs, batch_coords, img_w, img_h, conf_thr,
                               raw_boxes, raw_scores, raw_labels)

        return raw_boxes, raw_scores, raw_labels

    def predict_full_image(
        self, image: np.ndarray, conf_thr: float = 0.25
    ) -> tuple[list, list, list]:
        """Runs one DETR pass over the complete image."""
        img_h, img_w = image.shape[:2]
        boxes, scores, labels = [], [], []
        tensor = self._preprocess(image).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        with torch.no_grad():
            outputs = self.model(tensor)
        full_coord = [{"x": 0, "y": 0, "width": img_w, "height": img_h}]
        self._decode_batch(
            outputs, full_coord, img_w, img_h, conf_thr, boxes, scores, labels
        )
        return boxes, scores, labels


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_engine(arch: str, model_path: str, device: str = "cuda"):
    if arch in ("YOLO", "YOLOv8"):
        return TileInferenceEngine(model_path, device=device)
    if arch == "Faster":
        return FasterInferenceEngine(model_path, device=device)
    if arch == "Detr":
        return DetrInferenceEngine(model_path, device=device)
    raise ValueError(f"Unknown architecture: {arch!r}")
