"""
Gera figura side-by-side SAHI vs ASAHI vs ASAHI-Rect para artigo.
Uso: python scripts/visualize_slicing.py [caminho_imagem] [saida.png] [annotations.json]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from slicing.service import make_slicer

# ── Paleta para os tiles (RGB) ────────────────────────────────────────────── #
_PALETTE = [
    (0.18, 0.55, 0.95),
    (0.95, 0.45, 0.13),
    (0.22, 0.78, 0.44),
    (0.88, 0.20, 0.55),
    (0.60, 0.40, 0.95),
    (0.95, 0.80, 0.10),
    (0.20, 0.80, 0.90),
    (0.95, 0.35, 0.35),
]
_GT_COLOR = (0.95, 0.05, 0.05)


def _get_tile_coords(slicer, img: np.ndarray) -> list[dict]:
    return [coords for _, coords in slicer.generate_tiles(img)]


def _load_overlap(config_path: str, default: float) -> float:
    if not os.path.exists(config_path):
        return default
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("overlap_ratio:"):
                return float(stripped.split(":", 1)[1].strip())
    return default


def _overlap_title(name: str, overlap: float) -> str:
    return f"{name} - {overlap * 100:g}% overlap"


def _load_coco_boxes(annotation_path: str, image_path: str) -> list[list[float]]:
    if not annotation_path or not os.path.exists(annotation_path):
        return []

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_name = os.path.basename(image_path)
    image_entry = next(
        (img for img in coco.get("images", []) if os.path.basename(img.get("file_name", "")) == image_name),
        None,
    )
    if image_entry is None:
        return []

    image_id = image_entry["id"]
    return [
        ann["bbox"]
        for ann in coco.get("annotations", [])
        if ann.get("image_id") == image_id and len(ann.get("bbox", [])) == 4
    ]


def _draw_annotations(ax, boxes: list[list[float]]) -> None:
    for x, y, w, h in boxes:
        rect = mpatches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2.0,
            edgecolor=_GT_COLOR,
            facecolor=(*_GT_COLOR, 0.16),
        )
        ax.add_patch(rect)


def _draw_grid(ax, img_rgb: np.ndarray, coords: list[dict], boxes: list[list[float]], title: str):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")

    for i, c in enumerate(coords):
        color = _PALETTE[i % len(_PALETTE)]
        rect = mpatches.FancyBboxPatch(
            (c["x"], c["y"]),
            c["width"],
            c["height"],
            boxstyle="square,pad=0",
            linewidth=2.5,
            edgecolor=color,
            facecolor=(*color, 0.08),
        )
        ax.add_patch(rect)
        # tile index label at centre
        cx = c["x"] + c["width"] / 2
        cy = c["y"] + c["height"] / 2
        ax.text(
            cx, cy, str(i + 1),
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.75),
        )

    n = len(coords)
    _draw_annotations(ax, boxes)
    ax.set_xlabel(f"{n} tile{'s' if n != 1 else ''} | {len(boxes)} annotations", fontsize=11)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "dataset", "DJI_0626.JPG")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "slicing_comparison.png")
    annotation_path = (
        sys.argv[3] if len(sys.argv) > 3
        else os.path.join(root, "dataset", "all", "_annotations_clean.coco.json")
    )

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Imagem não encontrada: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    sahi_overlap = _load_overlap(os.path.join(root, "configs", "sahi.yaml"), 0.10)
    asahi_overlap = _load_overlap(os.path.join(root, "configs", "asahi.yaml"), 0.15)
    asahi_rect_overlap = _load_overlap(os.path.join(root, "configs", "asahi_rect.yaml"), 0.15)

    sahi  = make_slicer("sahi",  overlap_ratio=sahi_overlap)
    asahi = make_slicer("asahi", overlap_ratio=asahi_overlap)
    asahi_rect = make_slicer("asahi_rect", overlap_ratio=asahi_rect_overlap)

    sahi_coords  = _get_tile_coords(sahi,  img_bgr)
    asahi_coords = _get_tile_coords(asahi, img_bgr)
    asahi_rect_coords = _get_tile_coords(asahi_rect, img_bgr)
    boxes = _load_coco_boxes(annotation_path, img_path)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=200)

    _draw_grid(axes[0], img_rgb, sahi_coords, boxes, _overlap_title("SAHI", sahi_overlap))
    _draw_grid(axes[1], img_rgb, asahi_coords, boxes, _overlap_title("ASAHI", asahi_overlap))
    _draw_grid(axes[2], img_rgb, asahi_rect_coords, boxes, _overlap_title("ASAHI-Rect", asahi_rect_overlap))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
