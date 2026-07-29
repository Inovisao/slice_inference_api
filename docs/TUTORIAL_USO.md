# Tutorial de uso do pipeline

Este tutorial descreve o fluxo operacional do experimento. A ordem de
configuração é sempre:

```text
config.yaml -> setups_to_run -> configs/<recorte>.yaml
```

O `config.yaml` escolhe quais recortes entram no pipeline. Cada
`configs/<recorte>.yaml` define dataset, slicing, folds, inferência e modelos.

## 1. Preparar o dataset original

Coloque as imagens e o COCO bruto em:

```text
dataset/all/
├── _annotations.coco.json
├── imagem_01.jpg
└── ...
```

Cuidados:

- `images[].file_name` no COCO deve apontar para arquivos existentes.
- As caixas devem estar em COCO absoluto `[x, y, width, height]`.
- Os stems dos arquivos devem ser únicos.
- A divisão em folds é feita no nível da imagem original antes do recorte.

## 2. Configurar os recortes

Edite `config.yaml` para selecionar os setups:

```yaml
setups_to_run:
  - sahi
  - asahi
  - asahi_rect
  - all_640
```

Edite cada arquivo em `configs/` conforme necessário:

```yaml
slicing:
  mode: asahi
  tile_size: [640, 640]
  overlap_ratio: 0.15

crossfolds:
  n_folds: 5
  seed: 42
  split_strategy: kfold_holdout
  val_ratio: 0.15

models: [DETR, Faster, YOLOV8N]
```

No ASAHI, `tile_size: [640, 640]` define a dimensão limitante `r=640` usada no
threshold dependente da resolução:

```text
T = r * (4 - 3 * mu) + 1
```

## 3. Gerar os datasets

Gerar todos os setups selecionados:

```bash
conda run --no-capture-output -n slicing python main.py --yes
```

Gerar apenas um setup:

```bash
conda run --no-capture-output -n slicing python main.py --setup sahi --yes
conda run --no-capture-output -n slicing python main.py --setup asahi --yes
conda run --no-capture-output -n slicing python main.py --setup asahi_rect --yes
```

Gerar o baseline `all_640`:

```bash
conda run --no-capture-output -n slicing python scripts/build_baseline_all_640.py \
  --source dataset/all \
  --fold-source dataset/asahi_rect \
  --output dataset/all_640 \
  --size 640 \
  --seed 42 \
  --overwrite
```

## 4. Validar antes de treinar

Rode a validação para cada dataset que será treinado:

```bash
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/sahi
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/asahi
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/asahi_rect
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/all_640
```

Se a validação falhar, não treine. Corrija o dataset primeiro.

## 5. Smoke test do treinamento

Rode um teste curto antes do treinamento real:

```bash
DATASET_ROOT=dataset/asahi_rect \
MODEL_CHECKPOINTS_ROOT=pesos/asahi_rect_smoke/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8,Faster,Detr \
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/src/main.py --smoke-test --no-eval
```

Use uma pasta com `_smoke` para não misturar com pesos oficiais.

## 6. Treinamento real

Exemplo para SAHI:

```bash
DATASET_ROOT=dataset/sahi \
MODEL_CHECKPOINTS_ROOT=pesos/sahi/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8,Faster,Detr \
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/src/main.py
```

Exemplo para ASAHI:

```bash
DATASET_ROOT=dataset/asahi \
MODEL_CHECKPOINTS_ROOT=pesos/asahi/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8,Faster,Detr \
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/src/main.py
```

Exemplo para ASAHI-Rect:

```bash
DATASET_ROOT=dataset/asahi_rect \
MODEL_CHECKPOINTS_ROOT=pesos/asahi_rect/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8,Faster,Detr \
conda run --no-capture-output -n detectores python train_model/compara_detectores_torch/src/main.py
```

Depois do treino, padronize a estrutura de pesos e manifestos:

```bash
conda run --no-capture-output -n slicing python scripts/link_pesos_checkpoints.py
```

## 7. Avaliação

Avaliar tudo que está em `config.yaml`:

```bash
conda run --no-capture-output -n slicing python geraResultados.py
```

Avaliar apenas um caso:

```bash
conda run --no-capture-output -n slicing python geraResultados.py --setup asahi --model DETR --folds 1,2,3,4,5
```

Rodar diagnóstico sem contaminar o CSV oficial:

```bash
conda run --no-capture-output -n slicing python geraResultados.py \
  --setup asahi \
  --model DETR \
  --results-csv results/diagnostics/asahi_detr_results.csv \
  --counting-csv results/diagnostics/asahi_detr_counting.csv
```

## 8. Visualizações e análise

Visualizar recortes em uma imagem:

```bash
conda run --no-capture-output -n slicing python scripts/visualize_slicing.py \
  dataset/all/imagem_01.jpg \
  results/visualizations/slicing_visualizations.png \
  dataset/all/_annotations_clean.coco.json
```

Gerar análise estatística e gráficos:

```bash
Rscript geraGraficos.R
```

## 9. Checklist antes de considerar um resultado válido

- `config.yaml` seleciona o setup correto.
- `configs/<recorte>.yaml` tem `overlap_ratio`, `tile_size`, `seed` e `models` corretos.
- `validate_dataset_contract.py` passou.
- Pesos oficiais estão em `pesos/<recorte>/model_checkpoints/fold_N/<MODELO>/`.
- Manifestos existem em `models/<recorte>/fold_N/<MODELO>/manifest.json`.
- Avaliação foi feita nos 5 folds.
- Execuções diagnósticas usaram CSV alternativo.
