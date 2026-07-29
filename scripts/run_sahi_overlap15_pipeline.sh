#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--yes" ]]; then
  cat <<'USAGE'
Usage:
  scripts/run_sahi_overlap15_pipeline.sh --yes

This archives the current SAHI dataset/checkpoints/manifests/results, sets
configs/sahi.yaml overlap_ratio to 0.15, regenerates dataset/sahi, trains
YOLOV8/Faster/Detr, links manifests, and evaluates SAHI into timestamped CSVs.

Environment overrides:
  SLICING_ENV= slicing conda env used for dataset generation/evaluation
  TRAIN_ENV=detectores conda env used for model training
  MODELS_TO_RUN=YOLOV8,Faster,Detr
USAGE
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/sahi_overlap15_${STAMP}"
mkdir -p "$LOG_DIR"

SLICING_ENV="${SLICING_ENV:-slicing}"
TRAIN_ENV="${TRAIN_ENV:-detectores}"
MODELS="${MODELS_TO_RUN:-YOLOV8,Faster,Detr}"

log_run() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%F %T')] $name"
  echo "  $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

archive_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    local archived="${path}.before_sahi_overlap15_${STAMP}"
    echo "Archiving $path -> $archived"
    mv "$path" "$archived"
  fi
}

echo "Run stamp: $STAMP"
echo "Logs: $LOG_DIR"
echo "Slicing env: $SLICING_ENV"
echo "Training env: $TRAIN_ENV"
echo "Models: $MODELS"

archive_path "dataset/sahi"
archive_path "pesos/sahi"
archive_path "models/sahi"
archive_path "results/sahi"

python - <<'PY'
from pathlib import Path
path = Path("configs/sahi.yaml")
text = path.read_text(encoding="utf-8")
old = "  overlap_ratio: 0.1"
new = "  overlap_ratio: 0.15"
if new in text:
    print("configs/sahi.yaml already uses overlap_ratio 0.15")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("updated configs/sahi.yaml overlap_ratio: 0.1 -> 0.15")
else:
    raise SystemExit("Could not find expected SAHI overlap_ratio line in configs/sahi.yaml")
PY

log_run generate_sahi conda run --no-capture-output -n "$SLICING_ENV" \
  python main.py --setup sahi --yes

log_run validate_sahi conda run --no-capture-output -n "$TRAIN_ENV" \
  python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/sahi

log_run train_sahi conda run --no-capture-output -n "$TRAIN_ENV" \
  env DATASET_ROOT=dataset/sahi \
      MODEL_CHECKPOINTS_ROOT=pesos/sahi/model_checkpoints \
      EVAL_MODELS_ROOT=models \
      MODELS_TO_RUN="$MODELS" \
      TILING_MODE=basic \
      python train_model/compara_detectores_torch/src/main.py --no-eval

python - <<'PY'
from pathlib import Path
root = Path("pesos/sahi/model_checkpoints")
for fold in range(1, 6):
    fold_dir = root / f"fold_{fold}"
    src = fold_dir / "YOLOV8"
    dst = fold_dir / "YOLOV8N"
    if src.exists() and not dst.exists():
        src.rename(dst)
        print(f"renamed {src} -> {dst}")
    elif dst.exists():
        print(f"kept existing {dst}")
PY

log_run link_manifests conda run --no-capture-output -n "$SLICING_ENV" \
  python scripts/link_pesos_checkpoints.py

log_run evaluate_sahi conda run --no-capture-output -n "$SLICING_ENV" \
  python geraResultados.py \
    --setup sahi \
    --results-csv "results/sahi_overlap15_results_${STAMP}.csv" \
    --counting-csv "results/sahi_overlap15_counting_${STAMP}.csv"

echo
echo "Done."
echo "Results CSV: results/sahi_overlap15_results_${STAMP}.csv"
echo "Counting CSV: results/sahi_overlap15_counting_${STAMP}.csv"
echo "Logs: $LOG_DIR"
