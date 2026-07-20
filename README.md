# Slice Inference API

Pipeline para preparar datasets de imagens em alta resolução, gerar recortes
SAHI/ASAHI em validação cruzada, treinar detectores e avaliar inferência
fatiada para detecção de pequenos objetos.

O fluxo principal é:

```text
dataset/all
    │
    ▼
limpeza COCO + geração de folds/tiles        python main.py
    │
    ▼
dataset/sahi|asahi|asahi_rect
    │
    ▼
treinamento por fold                         train_model/compara_detectores_torch/src/main.py
    │
    ▼
pesos/<modo>/model_checkpoints/fold_N/<Modelo>/...
models/<modo>/fold_N/<arquitetura>/manifest.json
    │
    ▼
avaliação e visualizações                    python geraResultados.py
```

## Instalação

Requer Python 3.10 ou mais recente para o pipeline de recorte/API.

O módulo de treinamento `train_model/compara_detectores_torch` é um submódulo git.
Ao clonar, inclua-o:

```bash
git clone --recurse-submodules <repo-url>
# ou, em um clone já existente:
git submodule update --init --recursive
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O módulo `train_model/compara_detectores_torch` possui dependências próprias para
treinamento de YOLO, Faster R-CNN, DETR e outros detectores. Consulte
[train_model/compara_detectores_torch/README.md](train_model/compara_detectores_torch/README.md).

## 1. Recorte Das Imagens

### Onde colocar as imagens

Coloque as imagens originais e o COCO bruto em `dataset/all/`:

```text
dataset/all/
├── _annotations.coco.json
├── imagem_01.jpg
├── imagem_02.jpg
└── ...
```

O arquivo `_annotations.coco.json` deve conter `images`, `annotations` e
`categories`. O arquivo original não é sobrescrito. A limpeza gera:

```text
dataset/all/_annotations_clean.coco.json
```

Cuidados antes de gerar os recortes:

- Os nomes de imagem devem existir fisicamente em `dataset/all/`.
- Os stems dos arquivos devem ser únicos, porque os labels YOLO usam o stem.
- As anotações devem estar em COCO absoluto `[x, y, width, height]`.
- Caixas degeneradas, malformadas, fora da imagem ou com categoria inválida são removidas/ajustadas pelo preprocessador.
- Imagens sem anotação sobrevivente são removidas do COCO limpo.

### Configuração dos recortes

Os processos ficam em [config.yaml](config.yaml):

```yaml
paths:
  source_dataset: ./dataset/all
  generated_datasets: ./output
  models: ./models
  results: ./results

processes:
  - index: 1
    dataset:
      input_path: ./dataset/all
      output_path: ./dataset/sahi
    slicing:
      mode: sahi
      tile_size: [640, 640]
      overlap_ratio: 0.1
    crossfolds:
      n_folds: 5
      seed: 42
      ioa_threshold: 0.4
      split_strategy: kfold_holdout
      val_ratio: 0.15
      empty_tile_ratio: 0.08
```

Modos suportados:

- `sahi`: grade de tiles fixos 640x640 com overlap configurado.
- `asahi`: tile quadrado adaptativo por resolução.
- `asahi_rect`: tile retangular adaptativo, com largura/altura resolvidas por eixo.

Parâmetros importantes:

- `n_folds`: número de folds determinísticos gerados.
- `seed`: embaralhamento determinístico das imagens.
- `split_strategy`: protocolo de divisão. Use `kfold_holdout` para o protocolo atual dos resultados já treinados ou `fixed_ratios` para proporções globais explícitas.
- `val_ratio`: em `kfold_holdout`, fração retirada do conjunto não-teste; com `n_folds=5` e `val_ratio=0.15`, a proporção efetiva fica aproximadamente `68/12/20`. Em `fixed_ratios`, é a fração global de validação.
- `test_ratio`: obrigatório apenas em `fixed_ratios`; para `80/10/10`, use `val_ratio: 0.10` e `test_ratio: 0.10`.
- `ioa_threshold`: cobertura mínima da anotação original para manter uma caixa no tile.
- `empty_tile_ratio`: proporção máxima de tiles vazios mantidos em relação aos tiles anotados.

### Gerar os datasets recortados

Gerar todos os processos:

```bash
python main.py --yes
```

Gerar apenas um processo:

```bash
python main.py --process 3 --yes
```

Saída esperada:

```text
dataset/asahi_rect/
├── filesJSON/
│   ├── fold_1_train.json
│   ├── fold_1_val.json
│   ├── fold_1_test.json
│   └── ...
├── filesJSON_infos/
│   ├── fold_1.yaml
│   ├── fold_1_stats.json
│   └── ...
├── fold_1/
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── summary_report.json
├── resolution_groups.csv
└── per_image_metrics.csv
```

Regra metodológica: somente `train` recebe imagem inteira letterbox + tiles. Os
splits `val` e `test` preservam imagens originais com labels YOLO normalizados.

Valide o contrato antes do treinamento:

```bash
python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/asahi_rect
```

## 2. Configurações Do Treinamento

O treinamento integrado fica em:

```text
train_model/compara_detectores_torch/
```

Esse módulo consome obrigatoriamente o contrato cross-fold gerado por este
projeto. Ele não deve consumir `dataset/all` nem pastas achatadas
`dataset/<modo>/train`, `val`, `test`.

### Variáveis principais

Execute a partir da raiz deste repositório:

```bash
DATASET_ROOT=dataset/asahi_rect \
MODEL_CHECKPOINTS_ROOT=pesos/asahi_rect/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8,Faster,Detr \
python train_model/compara_detectores_torch/src/main.py
```

Significado:

- `DATASET_ROOT`: dataset recortado a treinar, por exemplo `dataset/sahi`, `dataset/asahi` ou `dataset/asahi_rect`. Obrigatório.
- `MODEL_CHECKPOINTS_ROOT`: pasta dos pesos/checkpoints da execução de treino. Use `pesos/<modo>/model_checkpoints`.
- `EVAL_MODELS_ROOT`: raiz onde serão gravados apenas os manifestos compatíveis com `geraResultados.py`.
- `MODELS_TO_RUN`: modelos a treinar, separados por vírgula.
- `TILING_MODE`: modo de avaliação; para estes datasets use `basic` ou omita.

Modelos aceitos pelo módulo de treino:

```text
YOLOV8, YOLOV11, YOLO26, YOLOV5_TPH, Faster, RetinaNet, Detr, SSDLite, ViT
```

O avaliador principal deste repositório consome diretamente:

```text
YOLOV8 -> models/<modo>/fold_N/yolo/manifest.json
Faster -> models/<modo>/fold_N/faster_rcnn/manifest.json
Detr   -> models/<modo>/fold_N/detr/manifest.json
```

### Onde configurar hiperparâmetros

Cada detector mantém seus hiperparâmetros em:

```text
train_model/compara_detectores_torch/src/Detectors/<Modelo>/config.py
```

Exemplos:

- `Detectors/YOLOV8/config.py`
- `Detectors/YOLO26/config.py`
- `Detectors/FasterRCNN/config.py`
- `Detectors/Detr/config.py`
- `Detectors/RetinaNet/config.py`

Vários detectores também aceitam sobrescrita por variáveis de ambiente, como
`YOLOV8_EPOCHS`, `YOLO26_BATCH`, `YOLOV11_WEIGHTS`, `RETINANET_LR` e similares.

### Smoke test recomendado

Antes de rodar todos os folds/modelos, rode um único modelo:

```bash
DATASET_ROOT=dataset/asahi_rect \
MODEL_CHECKPOINTS_ROOT=pesos/asahi_rect/model_checkpoints \
EVAL_MODELS_ROOT=models \
MODELS_TO_RUN=YOLOV8 \
python train_model/compara_detectores_torch/src/main.py
```

O treinamento deve gerar pelo menos:

```text
pesos/asahi_rect/model_checkpoints/fold_1/YOLOV8/train/weights/best.pt
models/asahi_rect/fold_1/yolo/manifest.json
```

O manifesto aponta para o checkpoint real e registra dataset, fold e JSONs COCO
usados no treino/validação/teste.

## 3. Avaliação Dos Resultados E Visualizações

Depois do treinamento, rode:

```bash
python geraResultados.py
```

O script:

- lê os processos em `config.yaml`;
- percorre `modo x fold x arquitetura`;
- resolve checkpoints via `models/<modo>/fold_N/<arquitetura>/manifest.json`;
- carrega as imagens originais do split `test`;
- executa inferência na imagem inteira e nos tiles;
- reprojeta predições para o espaço normalizado da imagem original;
- aplica supressão configurada (`nms` ou `cluster_diou_nms` nos processos atuais);
- calcula `mAP50`, `mAP75`, `mAP`, precisão, recall, F1, MAE, RMSE e correlação de Pearson;
- salva CSVs e visualizações em `results/`.

Arquivos principais:

```text
results/results.csv
results/counting.csv
results/<modo>/<arquitetura>/fold_N/*_eval.jpg
```

Gráficos e análise estatística:

```bash
Rscript geraGraficos.R
```

Esse script espera `results/results.csv`, `results/counting.csv` e, se usado,
resultados de baseline em `results/baseline/results.csv`.

### Dashboard de análise de threshold

Dashboard HTML interativo para escolher o threshold de confiança por modo de recorte
e detector, com boxplots por fold, curvas P/R/F1, curva Precision-Recall e dispersão de
contagem — todos reagindo a um slider. As métricas são pré-computadas offline (filtro →
supressão do framework → matching IoU@0,5), então o slider só faz lookup. Protocolo sem
vazamento: o threshold é escolhido no `val` e reportado no `test`.

```bash
# 1. Coleta detecções brutas (pré-supressão, conf>=0.1) — usa GPU
python scripts/collect_raw_detections.py
# 2. Pré-computa métricas por threshold (CPU) -> results/threshold_analysis/dashboard_data.json
python scripts/threshold_precompute.py
# 3. Sirva a raiz do projeto e abra o dashboard
python -m http.server 8080
# http://localhost:8080/dashboards/threshold_dashboard.html
```

O HTML fica versionado em [dashboards/](dashboards/); os dados gerados vivem em
`results/threshold_analysis/` (ignorado pelo git, regenerável). Detalhes em
[dashboards/README.md](dashboards/README.md).

### API de inferência operacional

A API é separada do treinamento e carrega checkpoints YOLO:

```bash
bash run.sh
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

Exemplo:

```bash
curl -X POST http://localhost:8000/inference/single_image \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./pesos/sahi/model_checkpoints/fold_1/YOLOV8/train/weights/best.pt",
    "image_name": "imagem_01.jpg",
    "slicing_mode": "sahi",
    "overlap_ratio": 0.1,
    "suppression": "nms",
    "conf": 0.5,
    "iou_thr": 0.5,
    "batch_size": 32,
    "include_full_image": false,
    "device": "cpu"
  }'
```

## 4. Projeto, Arquitetura E Organização

### Visão geral

```text
.
├── main.py                         # limpeza COCO + geração dos datasets recortados
├── config.yaml                     # processos de recorte, folds e avaliação
├── src/
│   ├── dataset/                    # preprocessamento COCO e folds determinísticos
│   ├── slicing/                    # SAHI, ASAHI e ASAHI retangular
│   ├── suppression/                # NMS, WBF, Cluster-DIoU-NMS etc.
│   ├── inference/                  # motores e pipeline de inferência fatiada
│   ├── evaluation/                 # loader, matcher e métricas
│   ├── train/                      # treino YOLO simples local
│   └── config/                     # loader de config.yaml
├── api/                            # FastAPI operacional
├── train_model/compara_detectores_torch/   # submódulo git — treino multi-arquitetura
│   └── src/Detectors/
├── scripts/                        # utilitários (coleta, pré-computo, layout de pesos)
├── dashboards/                     # HTML de visualização versionado (dashboard de threshold)
├── geraResultados.py               # avaliação experimental cross-fold
├── geraGraficos.R                  # gráficos e ANOVA/Tukey
├── dataset/
│   ├── all/                        # imagens originais e COCO bruto/limpo
│   ├── sahi/
│   ├── asahi/
│   ├── asahi_rect/
│   └── all_640/                    # baseline sem tiling (resize direto 640×640)
├── pesos/                          # checkpoints pesados dos treinamentos
├── models/                         # somente manifestos consumidos pela avaliação
└── results/                        # CSVs, visualizações e dados de dashboard (gerados)
```

### Responsabilidades dos módulos

- `src/dataset/preprocessor.py`: valida e normaliza COCO bruto.
- `src/dataset/kfold_generator.py`: cria splits, tiles, labels YOLO, JSONs COCO por split e relatórios.
- `src/slicing/sahi.py`: recorte em grade fixa.
- `src/slicing/asahi.py`: recorte quadrado adaptativo.
- `src/slicing/asahi_rect.py`: recorte retangular adaptativo.
- `train_model/compara_detectores_torch/src/main.py`: orquestra treinamento multi-modelo por fold.
- `train_model/compara_detectores_torch/src/dataset_contract.py`: valida e resolve o contrato cross-fold.
- `train_model/compara_detectores_torch/src/Detectors/*`: conversão de labels, treino e inferência de cada detector.
- `src/inference/engine.py`: motores YOLO, Faster R-CNN e DETR para avaliação/inferência.
- `src/inference/pipeline.py`: inferência fatiada, passagem inteira opcional, supressão e visualização.
- `src/evaluation/*`: carregamento de GT, matching por IoU e cálculo de métricas.
- `api/routers/*`: endpoints para dataset, slicing, inferência e reconstrução.

### Contratos importantes

Dataset recortado:

```text
dataset/<modo>/filesJSON/fold_N_split.json
dataset/<modo>/fold_N/split/images
dataset/<modo>/fold_N/split/labels
```

Checkpoint para avaliação:

```text
pesos/<modo>/model_checkpoints/fold_N/YOLOV8/train/weights/best.pt
pesos/<modo>/model_checkpoints/fold_N/Faster/best.pth
pesos/<modo>/model_checkpoints/fold_N/Detr/training/best_model.pth
models/<modo>/fold_N/yolo/manifest.json
models/<modo>/fold_N/faster_rcnn/manifest.json
models/<modo>/fold_N/detr/manifest.json
```

Não use pastas achatadas `dataset/<modo>/train`, `val`, `test`. Nesta branch, o
treinamento deve falhar se o contrato cross-fold não existir.

## Testes E Validações

Validação do contrato:

```bash
python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/sahi
python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/asahi
python train_model/compara_detectores_torch/scripts/validate_dataset_contract.py --root dataset/asahi_rect
```

Testes do pipeline principal:

```bash
pytest
```

Preflight com ativos externos:

```bash
pytest -m integration
```

## Referências

- ASAHI: <https://arxiv.org/abs/2604.19233>
- SAHI: <https://ieeexplore.ieee.org/document/9897990>
