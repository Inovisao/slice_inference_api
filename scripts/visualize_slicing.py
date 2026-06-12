"""
Gera figura side-by-side SAHI vs ASAHI para artigo.
Uso: python scripts/visualize_slicing.py [caminho_imagem] [saida.png]
"""

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


def _get_tile_coords(slicer, img: np.ndarray) -> list[dict]:
    return [coords for _, coords in slicer.generate_tiles(img)]


def _draw_grid(ax, img_rgb: np.ndarray, coords: list[dict], title: str):
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
    ax.set_xlabel(f"{n} tile{'s' if n != 1 else ''}", fontsize=11)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "dataset", "DJI_0626.JPG")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "slicing_comparison.png")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Imagem não encontrada: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    sahi  = make_slicer("sahi",  overlap_ratio=0.10)
    asahi = make_slicer("asahi", overlap_ratio=0.15)

    sahi_coords  = _get_tile_coords(sahi,  img_bgr)
    asahi_coords = _get_tile_coords(asahi, img_bgr)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)
    fig.suptitle(
        f"Estratégias de fatiamento — {os.path.basename(img_path)}  "
        f"({img_bgr.shape[1]}×{img_bgr.shape[0]} px)",
        fontsize=13, y=1.01,
    )

    _draw_grid(axes[0], img_rgb, sahi_coords,  "SAHI  (grade fixa, sobreposição 10%)")
    _draw_grid(axes[1], img_rgb, asahi_coords, "ASAHI  (tile adaptativo, sobreposição 15%)")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
