import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from collections import Counter
from typing import Dict, List, Optional, Tuple

from config.config_loader import ConfigLoader, ProcessConfig
from config.settings import SlicingConfig
from dataset.kfold_generator import AsahiKFoldValidator, GeometryParams, compute_geometry
from dataset.preprocessor import DatasetPreprocessor
from slicing.service import make_slicer

_COCO_CLEAN = "_annotations_clean.coco.json"
_COCO_RAW   = "_annotations.coco.json"
_SEP        = "─" * 62


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
    """Returns {(w, h): image_count} for entries that have width/height."""
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)
    counter: Counter = Counter()
    for img in coco.get("images", []):
        w, h = img.get("width", 0), img.get("height", 0)
        if w and h:
            counter[(w, h)] += 1
    return dict(counter)


# ------------------------------------------------------------------ #
# Preview                                                              #
# ------------------------------------------------------------------ #

def _print_process_preview(proc: ProcessConfig):
    s  = proc.slicing
    d  = proc.dataset
    cf = proc.crossfolds
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
    print(f"    min_coverage  {s.min_object_coverage * 100:.0f}%")

    print(f"\n  K-Fold")
    print(f"    n_folds       {cf.n_folds}")
    print(f"    seed          {cf.seed}")
    print(f"    ioa_threshold {cf.ioa_threshold}")

    print(f"\n  Inferência")
    print(f"    suppression   {inf.suppression}")
    print(f"    conf          {inf.conf_threshold}  |  iou {inf.iou_threshold}")
    print(f"    batch_size    {inf.batch_size}  |  workers {inf.num_workers}")

    coco_path = _find_coco(d.input_path)
    if coco_path is None:
        print(f"\n  [!] COCO JSON não encontrado em '{d.input_path}' — geometria indisponível")
        return

    resolutions = _resolutions_from_coco(coco_path)
    if not resolutions:
        print(f"\n  [!] Nenhuma imagem com width/height no JSON — execute o preprocessor primeiro")
        return

    slicer = make_slicer(s.slicing_mode, s.overlap_ratio)
    rows: List[Tuple[Tuple[int, int], int, GeometryParams]] = [
        ((w, h), count, compute_geometry(slicer, w, h))
        for (w, h), count in sorted(resolutions.items())
    ]

    print(f"\n  Geometria por resolução")
    print(f"  {_SEP}")
    print(
        f"  {'Resolução':<14} {'Imgs':>5} {'Tile p':>7} "
        f"{'Cols':>5} {'Rows':>5} {'Tiles/img':>10} {'Redundância':>12}"
    )
    print(f"  {_SEP}")
    for (w, h), count, g in rows:
        print(
            f"  {f'{w}×{h}':<14} {count:>5} {g.tile_size_p:>7} "
            f"{g.cols:>5} {g.rows:>5} {g.tiles_per_image:>10} {g.redundancy_pct:>11.1f}%"
        )
    print(f"  {_SEP}")


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
        min_object_coverage=s.min_object_coverage,
    )
    validator = AsahiKFoldValidator(
        dataset_path=d.input_path,
        slicing_config=slicing_cfg,
        n_splits=cf.n_folds,
        output_root=d.output_path,
        seed=cf.seed,
        ioa_threshold=cf.ioa_threshold,
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

    for proc in processes:
        _print_process_preview(proc)

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
