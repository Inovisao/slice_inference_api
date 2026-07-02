import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from collections import Counter
from typing import Dict, List, Optional, Tuple

from config.config_loader import ConfigLoader, ProcessConfig
from config.settings import SlicingConfig
from dataset.kfold_generator import AsahiKFoldValidator, GeometryParams, compute_geometry
from dataset.preprocessor import DatasetPreprocessor
from slicing.service import make_slicer

_COCO_CLEAN    = "_annotations_clean.coco.json"
_COCO_RAW      = "_annotations.coco.json"
_SEP           = "─" * 62
_JPEG_FACTOR   = 0.08   # empirical JPEG compression ratio for 640px tiles (~96 KB/tile)
_LABEL_BYTES   = 512    # average .txt label file size per tile


# ------------------------------------------------------------------ #
# COCO helpers                                                         #
# ------------------------------------------------------------------ #

def _find_coco(dataset_path: str) -> Optional[str]:
    for fname in (_COCO_CLEAN, _COCO_RAW):
        p = os.path.join(dataset_path, fname)
        if os.path.isfile(p):
            return p
    return None


def _resolutions_from_coco(coco_path: str) -> Dict[Tuple[int, int], int]:
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)
    counter: Counter = Counter()
    for img in coco.get("images", []):
        w, h = img.get("width", 0), img.get("height", 0)
        if w and h:
            counter[(w, h)] += 1
    return dict(counter)


# ------------------------------------------------------------------ #
# Disk estimation                                                      #
# ------------------------------------------------------------------ #

def _estimate_bytes_per_tile(tile_w: int, tile_h: int) -> int:
    """Rough JPEG size estimate: raw pixels × compression factor + label overhead."""
    return int(tile_w * tile_h * 3 * _JPEG_FACTOR) + _LABEL_BYTES


def _estimate_process_gb(
    resolutions: Dict[Tuple[int, int], int],
    tile_w: int,
    tile_h: int,
    n_folds: int,
    geoms: List[Tuple[Tuple[int, int], int, GeometryParams]],
) -> float:
    """
    In k-fold every image is written n_folds times (n-1 as train + 1 as val).
    total_bytes = Σ image_count × tiles_per_image × n_folds × bytes_per_tile
    """
    bytes_per_tile = _estimate_bytes_per_tile(tile_w, tile_h)
    total = sum(
        count * g.tiles_per_image * n_folds * bytes_per_tile
        for (_, _), count, g in geoms
    )
    return total / 1e9


def _free_gb(path: str) -> float:
    anchor = path
    while not os.path.exists(anchor):
        anchor = os.path.dirname(anchor)
        if anchor in ("", "/"):
            anchor = "."
            break
    return shutil.disk_usage(anchor).free / 1e9


# ------------------------------------------------------------------ #
# Preview                                                              #
# ------------------------------------------------------------------ #

def _print_process_preview(proc: ProcessConfig) -> float:
    """Prints config + geometry table. Returns estimated GB for this process."""
    s   = proc.slicing
    d   = proc.dataset
    cf  = proc.crossfolds
    inf = proc.inference

    print(f"\n{'━' * 64}")
    print(f"  Processo {proc.index}  —  modo: {s.slicing_mode.upper()}")
    print(f"{'━' * 64}")

    print(f"  Dataset")
    print(f"    input   {d.input_path}")
    print(f"    output  {d.output_path}")

    print(f"\n  Slicing")
    print(f"    mode          {s.slicing_mode}")
    print(f"    tile_size     {s.tile_size[0]}×{s.tile_size[1]} px")
    print(f"    overlap       {s.overlap_ratio * 100:.0f}%")

    print(f"\n  K-Fold")
    print(f"    n_folds       {cf.n_folds}")
    print(f"    seed          {cf.seed}")
    print(f"    ioa_threshold {cf.ioa_threshold}")

    print(f"\n  Inferência")
    print(f"    suppression   {inf.suppression}")
    print(f"    conf          {inf.conf_threshold}  |  iou {inf.iou_threshold}")
    print(f"    batch_size    {inf.batch_size}")

    coco_path = _find_coco(d.input_path)
    if coco_path is None:
        print(f"\n  [!] COCO JSON não encontrado em '{d.input_path}' — geometria indisponível")
        return 0.0

    resolutions = _resolutions_from_coco(coco_path)
    if not resolutions:
        print(f"\n  [!] Nenhuma imagem com width/height no JSON — execute o preprocessor primeiro")
        return 0.0

    slicer = make_slicer(s.slicing_mode, s.overlap_ratio)
    geoms: List[Tuple[Tuple[int, int], int, GeometryParams]] = [
        ((w, h), count, compute_geometry(slicer, w, h))
        for (w, h), count in sorted(resolutions.items())
    ]

    # Geometry table
    print(f"\n  Geometria por resolução")
    print(f"  {_SEP}")
    print(
        f"  {'Resolução':<14} {'Imgs':>5} {'Tile p':>7} "
        f"{'Cols':>5} {'Rows':>5} {'Tiles/img':>10} {'Redundância':>12}"
    )
    print(f"  {_SEP}")
    for (w, h), count, g in geoms:
        print(
            f"  {f'{w}×{h}':<14} {count:>5} {g.tile_size_p:>7} "
            f"{g.cols:>5} {g.rows:>5} {g.tiles_per_image:>10} {g.redundancy_pct:>11.1f}%"
        )
    print(f"  {_SEP}")

    # Disk estimate
    estimated_gb = _estimate_process_gb(
        resolutions, s.tile_size[0], s.tile_size[1], cf.n_folds, geoms
    )
    free_gb = _free_gb(d.output_path)
    flag = "  [!] ESPAÇO INSUFICIENTE" if estimated_gb > free_gb else ""
    print(f"\n  Disco")
    print(f"    estimado para recorte  ~{estimated_gb:.1f} GB")
    print(f"    livre em output        {free_gb:.1f} GB{flag}")

    return estimated_gb


# ------------------------------------------------------------------ #
# Pipeline                                                             #
# ------------------------------------------------------------------ #

def _run_process(proc: ProcessConfig):
    d  = proc.dataset
    s  = proc.slicing
    cf = proc.crossfolds

    print(f"\n{'━' * 64}")
    print(f"  [Processo {proc.index}] {s.slicing_mode.upper()} — iniciando pipeline")
    print(f"{'━' * 64}")

    # Step 1 — preprocess
    print(f"\n  [1/2] Pré-processando dataset...")
    report = DatasetPreprocessor(d.input_path).run()
    imgs = report["images"]
    anns = report["annotations"]
    print(
        f"        imagens: {imgs['kept']} mantidas  "
        f"(removidas: {imgs['removed_no_file']} sem arquivo"
        f" + {imgs['removed_no_annotation']} sem anotação)"
    )
    if imgs["dimensions_filled_from_disk"]:
        print(f"        width/height preenchidos do disco: {imgs['dimensions_filled_from_disk']}")
    print(
        f"        anotações: {anns['kept']} mantidas  "
        f"(removidas: {anns['removed_degenerate_after_clamp']} degeneradas"
        f" + {anns['removed_unknown_category']} categoria inválida"
        f" + {anns['removed_malformed_bbox']} bbox malformado)"
    )
    if anns["clamped_to_bounds"]:
        print(f"        clampadas ao limite da imagem: {anns['clamped_to_bounds']}")
    if anns["area_recalculated"]:
        print(f"        área recalculada: {anns['area_recalculated']}")

    # Step 2 — kfold generation
    print(f"\n  [2/2] Gerando {cf.n_folds} dobras ({s.slicing_mode})...")
    slicing_cfg = SlicingConfig(
        slicing_mode=s.slicing_mode,
        tile_size=s.tile_size,
        overlap_ratio=s.overlap_ratio,
    )
    validator = AsahiKFoldValidator(
        dataset_path=d.input_path,
        slicing_config=slicing_cfg,
        n_splits=cf.n_folds,
        output_root=d.output_path,
        seed=cf.seed,
        ioa_threshold=cf.ioa_threshold,
        empty_tile_ratio=cf.empty_tile_ratio,
        val_ratio=cf.val_ratio,
    )
    validator.run()
    print(f"\n  Processo {proc.index} concluído.")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    try:
        loader = ConfigLoader(cfg_path)
    except Exception as exc:
        print(f"[ERRO] Falha ao carregar config.yaml: {exc}")
        sys.exit(1)

    processes = loader.processes

    print(f"\n{'━' * 64}")
    print(f"  Slice Inference API  —  {len(processes)} processo(s) configurado(s)")
    print(f"{'━' * 64}")

    total_estimated_gb = 0.0
    for proc in processes:
        total_estimated_gb += _print_process_preview(proc)

    # Global disk summary
    if total_estimated_gb > 0:
        # Use the first output path as reference for total free space
        ref_path = processes[0].dataset.output_path
        free_gb = _free_gb(ref_path)
        ok = total_estimated_gb <= free_gb
        print(f"\n{'━' * 64}")
        print(f"  Resumo de disco (todos os processos)")
        print(f"    total estimado   ~{total_estimated_gb:.1f} GB")
        print(f"    livre            {free_gb:.1f} GB")
        if not ok:
            print(f"    [!] Espaço insuficiente — libere pelo menos "
                  f"{total_estimated_gb - free_gb:.1f} GB antes de continuar")

    print(f"\n{'━' * 64}")
    try:
        answer = input("  Confirmar recorte e geração das dobras? [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelado.")
        return

    if answer not in ("s", "sim", "y", "yes"):
        print("  Cancelado.")
        return

    for proc in processes:
        _run_process(proc)

    print(f"\n{'━' * 64}")
    print(f"  Todos os processos concluídos.")
    print(f"{'━' * 64}\n")


if __name__ == "__main__":
    main()
