"""
Testes de visualização para SAHI e ASAHI.

Contexto: imagens agrícolas de 4032×2268 com insetos minúsculos.
Resize global para 640×640 destrói a escala dos objetos — daí o slicing.

Cada teste valida uma propriedade do pipeline E salva uma imagem em
tests/output/slicing/ para inspeção visual humana.

Questões cobertas:
  1. Acúmulo de cobertura nas bordas (SAHI vs ASAHI)
  2. Uniformidade do stride (ASAHI não deve ter âncora de borda)
  3. Derivação adaptativa do tile size (ASAHI, Eq. 2 do paper)
  4. Visualização do grid sobre imagem sintética agrícola
  5. Visualizações sobre a imagem real do dataset (4.jpg)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config.settings import SlicingConfig
from slicing.sahi import Sahi
from slicing.asahi import Asahi

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "slicing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 4032, 2268
OVERLAP = 0.15


def _make_sahi(overlap: float = OVERLAP) -> Sahi:
    cfg = SlicingConfig(slicing_mode="sahi", tile_size=(640, 640), overlap_ratio=overlap, min_object_coverage=0.5)
    return Sahi(cfg)


def _make_asahi(overlap: float = OVERLAP) -> Asahi:
    cfg = SlicingConfig(slicing_mode="asahi", tile_size=(640, 640), overlap_ratio=overlap, min_object_coverage=0.5)
    return Asahi(cfg)


def _synthetic_field(w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = np.linspace(40, 120, w, dtype=np.uint8)
    noise = np.random.RandomState(42).randint(0, 30, (h, w), dtype=np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] + noise, 0, 255).astype(np.uint8)
    return img


def _coverage_map(slicer, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """Conta quantos tiles cobrem cada pixel."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.int32)
    for _, coords in slicer.generate_tiles(image):
        x, y, tw, th = coords["x"], coords["y"], coords["width"], coords["height"]
        coverage[y: y + th, x: x + tw] += 1
    return coverage


def _draw_grid(image: np.ndarray, slicer, color=(0, 255, 0), thickness=3) -> np.ndarray:
    out = image.copy()
    for _, coords in slicer.generate_tiles(image):
        x, y, tw, th = coords["x"], coords["y"], coords["width"], coords["height"]
        cv2.rectangle(out, (x, y), (x + tw, y + th), color, thickness)
    return out


def _draw_overlap_zones(image: np.ndarray, slicer, alpha: float = 0.35) -> np.ndarray:
    """
    Pinta cada tile com camada semi-transparente.
    Onde tiles se sobrepõem, as camadas acumulam — tornando visível
    a zona de overlap tanto horizontal quanto vertical.
    """
    overlay = image.copy().astype(np.float32)
    tile_color = np.array([60, 180, 60], dtype=np.float32)

    for _, coords in slicer.generate_tiles(image):
        x, y, tw, th = coords["x"], coords["y"], coords["width"], coords["height"]
        region = overlay[y: y + th, x: x + tw]
        overlay[y: y + th, x: x + tw] = region * (1 - alpha) + tile_color * alpha

    out = np.clip(overlay, 0, 255).astype(np.uint8)
    for _, coords in slicer.generate_tiles(image):
        x, y, tw, th = coords["x"], coords["y"], coords["width"], coords["height"]
        cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 220, 60), 3)
    return out


def _heatmap_bgr(coverage: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(coverage, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _blend_heatmap(image: np.ndarray, coverage: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    normalized = cv2.normalize(coverage, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


# ---------------------------------------------------------------------------
# 1. Acúmulo de cobertura nas bordas
# ---------------------------------------------------------------------------

class TestCoverageAccumulation:

    def test_sahi_has_higher_coverage_at_corners_than_interior(self):
        coverage = _coverage_map(_make_sahi())
        corner_coverage = np.mean([
            coverage[:640, :640],
            coverage[:640, -640:],
            coverage[-640:, :640],
            coverage[-640:, -640:],
        ])
        center_h, center_w = IMG_H // 2, IMG_W // 2
        interior_coverage = np.mean(coverage[center_h - 320: center_h + 320, center_w - 320: center_w + 320])
        assert corner_coverage > interior_coverage

    def test_asahi_coverage_is_uniform(self):
        coverage = _coverage_map(_make_asahi())
        assert coverage.std() < 1.0

    def test_asahi_coverage_is_determined_by_overlap_not_border_artifact(self):
        """
        ASAHI: cobertura máxima deve estar no interior (zonas de sobreposição
        geométrica), não nas bordas por âncora artificial.
        """
        coverage = _coverage_map(_make_asahi())
        border_strip = 50
        border_pixels = np.concatenate([
            coverage[:border_strip, :].ravel(),
            coverage[-border_strip:, :].ravel(),
            coverage[:, :border_strip].ravel(),
            coverage[:, -border_strip:].ravel(),
        ])
        interior = coverage[border_strip:-border_strip, border_strip:-border_strip]
        assert interior.mean() >= border_pixels.mean()

    def test_sahi_border_anchor_creates_coverage_spike_at_edge(self):
        sahi = _make_sahi()
        stride_x, _ = sahi.compute_stride()
        positions = sahi._axis_positions(IMG_W, 640, stride_x)
        last_gap = positions[-1] - positions[-2]
        regular_gap = positions[1] - positions[0]
        assert last_gap < regular_gap

    def test_saves_coverage_heatmap_comparison(self):
        sahi_map = _heatmap_bgr(_coverage_map(_make_sahi()))
        asahi_map = _heatmap_bgr(_coverage_map(_make_asahi()))
        scale = 0.15
        sahi_small = cv2.resize(sahi_map, (0, 0), fx=scale, fy=scale)
        asahi_small = cv2.resize(asahi_map, (0, 0), fx=scale, fy=scale)
        sep = np.full((sahi_small.shape[0], 6, 3), 255, dtype=np.uint8)
        comparison = np.hstack([sahi_small, sep, asahi_small])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "SAHI (bordas acumulam)", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "ASAHI (cobertura uniforme)", (sahi_small.shape[1] + 16, 30), font, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / "coverage_heatmap_comparison.jpg"), comparison)


# ---------------------------------------------------------------------------
# 2. Uniformidade do stride
# ---------------------------------------------------------------------------

class TestStrideUniformity:

    def test_sahi_has_irregular_last_gap(self):
        sahi = _make_sahi()
        stride_x, _ = sahi.compute_stride()
        positions = sahi._axis_positions(IMG_W, 640, stride_x)
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        assert gaps[-1] != gaps[0]

    def test_asahi_all_gaps_are_nearly_equal(self):
        asahi = _make_asahi()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, _ = asahi.compute_grid(IMG_W, IMG_H, p)
        positions = asahi._axis_positions(IMG_W, p, a)
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        assert max(gaps) - min(gaps) <= 1

    def test_asahi_last_tile_ends_exactly_at_image_edge(self):
        asahi = _make_asahi()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, b = asahi.compute_grid(IMG_W, IMG_H, p)
        positions_x = asahi._axis_positions(IMG_W, p, a)
        positions_y = asahi._axis_positions(IMG_H, p, b)
        assert positions_x[-1] + p == IMG_W
        assert positions_y[-1] + p == IMG_H

    def test_saves_gap_distribution(self):
        sahi = _make_sahi()
        asahi = _make_asahi()
        stride_x, _ = sahi.compute_stride()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, _ = asahi.compute_grid(IMG_W, IMG_H, p)
        sahi_pos = sahi._axis_positions(IMG_W, 640, stride_x)
        asahi_pos = asahi._axis_positions(IMG_W, p, a)
        height = 120
        out = np.ones((height * 2 + 20, IMG_W, 3), dtype=np.uint8) * 30
        for x in sahi_pos:
            cv2.line(out, (x, 0), (x, height), (0, 200, 80), 2)
        for x in asahi_pos:
            cv2.line(out, (x, height + 20), (x, height * 2 + 20), (80, 150, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, f"SAHI  ({len(sahi_pos)} colunas, tile=640px)", (10, height - 10), font, 0.7, (0, 200, 80), 2)
        cv2.putText(out, f"ASAHI ({len(asahi_pos)} colunas, tile={p}px)", (10, height * 2 + 15), font, 0.7, (80, 150, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / "stride_positions_x.jpg"), cv2.resize(out, (0, 0), fx=0.3, fy=1.0))


# ---------------------------------------------------------------------------
# 3. Derivação adaptativa do tile size (ASAHI Eq. 2)
# ---------------------------------------------------------------------------

class TestAsahiTileDerivation:

    def test_tile_size_covers_image_with_given_overlap(self):
        asahi = _make_asahi()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, b = asahi.compute_grid(IMG_W, IMG_H, p)
        covered_w = p + (a - 1) * p * (1 - OVERLAP)
        assert covered_w >= IMG_W

    def test_grid_has_expected_block_count_for_large_image(self):
        asahi = _make_asahi()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, b = asahi.compute_grid(IMG_W, IMG_H, p)
        assert 3 <= a <= 6
        assert 1 <= b <= 4

    @pytest.mark.parametrize("w, h", [
        (1920, 1080),
        (4032, 2268),
        (3840, 2160),
        (2560, 1440),
    ])
    def test_tile_size_never_zero_for_common_resolutions(self, w, h):
        assert _make_asahi().compute_tile_size(w, h) > 0

    @pytest.mark.parametrize("w, h", [
        (1920, 1080),
        (4032, 2268),
        (3840, 2160),
        (2560, 1440),
    ])
    def test_asahi_grid_covers_full_image_for_common_resolutions(self, w, h):
        asahi = _make_asahi()
        p = asahi.compute_tile_size(w, h)
        a, b = asahi.compute_grid(w, h, p)
        assert asahi._axis_positions(w, p, a)[-1] + p >= w
        assert asahi._axis_positions(h, p, b)[-1] + p >= h


# ---------------------------------------------------------------------------
# 4. Grid sobre imagem sintética
# ---------------------------------------------------------------------------

class TestGridVisualization:

    def test_saves_sahi_grid_on_field_image(self):
        img = _synthetic_field()
        grid = _draw_grid(img, _make_sahi(), color=(0, 255, 0))
        sahi = _make_sahi()
        stride_x, stride_y = sahi.compute_stride()
        cv2.putText(grid, f"SAHI | tile=640px | stride={stride_x}x{stride_y} | overlap={int(OVERLAP*100)}%",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imwrite(str(OUTPUT_DIR / "sahi_grid.jpg"), cv2.resize(grid, (0, 0), fx=0.2, fy=0.2))

    def test_saves_asahi_grid_on_field_image(self):
        img = _synthetic_field()
        asahi = _make_asahi()
        p = asahi.compute_tile_size(IMG_W, IMG_H)
        a, b = asahi.compute_grid(IMG_W, IMG_H, p)
        grid = _draw_grid(img, asahi, color=(80, 150, 255))
        cv2.putText(grid, f"ASAHI | tile={p}px | grade={a}x{b} | overlap={int(OVERLAP*100)}%",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 150, 255), 3)
        cv2.imwrite(str(OUTPUT_DIR / "asahi_grid.jpg"), cv2.resize(grid, (0, 0), fx=0.2, fy=0.2))

    def test_saves_side_by_side_grid_comparison(self):
        img = _synthetic_field()
        sahi_grid = cv2.resize(_draw_grid(img, _make_sahi(), color=(0, 255, 0)), (0, 0), fx=0.2, fy=0.2)
        asahi_grid = cv2.resize(_draw_grid(img, _make_asahi(), color=(80, 150, 255)), (0, 0), fx=0.2, fy=0.2)
        sep = np.full((sahi_grid.shape[0], 8, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / "grid_comparison.jpg"), np.hstack([sahi_grid, sep, asahi_grid]))

    def test_tile_count_sahi_vs_asahi(self):
        image = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        sahi_tiles = sum(1 for _ in _make_sahi().generate_tiles(image))
        asahi_tiles = sum(1 for _ in _make_asahi().generate_tiles(image))
        assert asahi_tiles < sahi_tiles


# ---------------------------------------------------------------------------
# 5. Visualizações sobre a imagem real do dataset (4.jpg)
# ---------------------------------------------------------------------------

DATASET_IMAGE = Path(__file__).resolve().parents[2] / "dataset" / "4.jpg"


def _load_real_image():
    img = cv2.imread(str(DATASET_IMAGE))
    if img is None:
        pytest.skip(f"Imagem não encontrada: {DATASET_IMAGE}")
    return img


class TestRealImageVisualization:

    def test_saves_sahi_grid_on_real_image(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        sahi = _make_sahi()
        stride_x, stride_y = sahi.compute_stride()
        n_tiles = sum(1 for _ in sahi.generate_tiles(img))
        grid = _draw_grid(img, sahi, color=(0, 220, 60), thickness=4)
        label = f"SAHI  |  tile=640px  |  stride={stride_x}x{stride_y}  |  {n_tiles} tiles  |  overlap={int(OVERLAP*100)}%"
        cv2.putText(grid, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(grid, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 60), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_sahi_grid.jpg"), cv2.resize(grid, (0, 0), fx=0.25, fy=0.25))

    def test_saves_asahi_grid_on_real_image(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        asahi = _make_asahi()
        p = asahi.compute_tile_size(w, h)
        a, b = asahi.compute_grid(w, h, p)
        n_tiles = sum(1 for _ in asahi.generate_tiles(img))
        grid = _draw_grid(img, asahi, color=(80, 150, 255), thickness=6)
        label = f"ASAHI  |  tile={p}px  |  grade={a}x{b}  |  {n_tiles} tiles  |  overlap={int(OVERLAP*100)}%"
        cv2.putText(grid, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(grid, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 150, 255), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_asahi_grid.jpg"), cv2.resize(grid, (0, 0), fx=0.25, fy=0.25))

    def test_saves_sahi_overlap_zones_on_real_image(self):
        img = _load_real_image()
        sahi = _make_sahi()
        stride_x, _ = sahi.compute_stride()
        overlap_px = 640 - stride_x
        out = _draw_overlap_zones(img, sahi)
        label = f"SAHI  overlap zones  |  {overlap_px}px por tile  |  bordas acumulam"
        cv2.putText(out, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(out, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 60), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_sahi_overlap_zones.jpg"), cv2.resize(out, (0, 0), fx=0.25, fy=0.25))

    def test_saves_asahi_overlap_zones_on_real_image(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        asahi = _make_asahi()
        p = asahi.compute_tile_size(w, h)
        a, b = asahi.compute_grid(w, h, p)
        step_x = round((w - p) / (a - 1)) if a > 1 else w
        step_y = round((h - p) / (b - 1)) if b > 1 else h
        out = _draw_overlap_zones(img, asahi)
        label = f"ASAHI  tile={p}px  |  overlap X={p - step_x}px  overlap Y={p - step_y}px  |  grade {a}x{b}"
        cv2.putText(out, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(out, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 150, 255), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_asahi_overlap_zones.jpg"), cv2.resize(out, (0, 0), fx=0.25, fy=0.25))

    def test_saves_overlap_zones_comparison(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        asahi = _make_asahi()
        p = asahi.compute_tile_size(w, h)
        a, b = asahi.compute_grid(w, h, p)
        sahi_zones  = cv2.resize(_draw_overlap_zones(img, _make_sahi()), (0, 0), fx=0.25, fy=0.25)
        asahi_zones = cv2.resize(_draw_overlap_zones(img, asahi),        (0, 0), fx=0.25, fy=0.25)
        sahi_stride = _make_sahi().compute_stride()[0]
        cv2.putText(sahi_zones,  f"SAHI  overlap={640 - sahi_stride}px", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 60),   3)
        cv2.putText(asahi_zones, f"ASAHI tile={p}px  grade={a}x{b}",     (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 150, 255), 3)
        sep = np.full((sahi_zones.shape[0], 6, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / "real_overlap_zones_comparison.jpg"), np.hstack([sahi_zones, sep, asahi_zones]))

    def test_saves_sahi_coverage_heatmap_on_real_image(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        coverage = _coverage_map(_make_sahi(), w, h)
        blended = _blend_heatmap(img, coverage)
        max_cov, min_cov = int(coverage.max()), int(coverage.min())
        cv2.putText(blended, f"SAHI  coverage: min={min_cov}  max={max_cov}  range={max_cov - min_cov}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(blended, f"SAHI  coverage: min={min_cov}  max={max_cov}  range={max_cov - min_cov}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_sahi_heatmap.jpg"), cv2.resize(blended, (0, 0), fx=0.25, fy=0.25))

    def test_saves_asahi_coverage_heatmap_on_real_image(self):
        img = _load_real_image()
        h, w = img.shape[:2]
        coverage = _coverage_map(_make_asahi(), w, h)
        blended = _blend_heatmap(img, coverage)
        max_cov, min_cov = int(coverage.max()), int(coverage.min())
        cv2.putText(blended, f"ASAHI  coverage: min={min_cov}  max={max_cov}  range={max_cov - min_cov}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 9)
        cv2.putText(blended, f"ASAHI  coverage: min={min_cov}  max={max_cov}  range={max_cov - min_cov}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        cv2.imwrite(str(OUTPUT_DIR / "real_asahi_heatmap.jpg"), cv2.resize(blended, (0, 0), fx=0.25, fy=0.25))

    def test_saves_full_comparison_on_real_image(self):
        """Painel 2×2: grid SAHI | grid ASAHI / heatmap SAHI | heatmap ASAHI"""
        img = _load_real_image()
        h, w = img.shape[:2]
        scale = 0.25
        sahi = _make_sahi()
        asahi = _make_asahi()
        p = asahi.compute_tile_size(w, h)
        sahi_grid  = cv2.resize(_draw_grid(img, sahi,  color=(0, 220, 60),   thickness=4), (0, 0), fx=scale, fy=scale)
        asahi_grid = cv2.resize(_draw_grid(img, asahi, color=(80, 150, 255), thickness=6), (0, 0), fx=scale, fy=scale)
        sahi_heat  = cv2.resize(_blend_heatmap(img, _coverage_map(sahi,  w, h)), (0, 0), fx=scale, fy=scale)
        asahi_heat = cv2.resize(_blend_heatmap(img, _coverage_map(asahi, w, h)), (0, 0), fx=scale, fy=scale)
        sep_v = np.full((sahi_grid.shape[0], 6, 3), 200, dtype=np.uint8)
        sep_h = np.full((6, sahi_grid.shape[1] * 2 + 6, 3), 200, dtype=np.uint8)
        top = np.hstack([sahi_grid, sep_v, asahi_grid])
        bot = np.hstack([sahi_heat,  sep_v, asahi_heat])
        panel = np.vstack([top, sep_h, bot])
        tw = sahi_grid.shape[1]
        cv2.putText(panel, f"SAHI  640px  {sum(1 for _ in sahi.generate_tiles(img))} tiles",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60),   2)
        cv2.putText(panel, f"ASAHI {p}px  {sum(1 for _ in asahi.generate_tiles(img))} tiles", (tw + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 150, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / "real_full_comparison.jpg"), panel)
