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
pesos/<recorte>/model_checkpoints/fold_N/<modelo>/...
models/<recorte>/fold_N/<modelo>/manifest.json
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

A configuração é dividida em duas camadas. O [config.yaml](config.yaml) é um
**índice de setups** — lista os recortes disponíveis e seleciona quais rodar:

```yaml
paths:
  source_dataset: ./dataset/all
  generated_datasets: ./output
  models: ./models
  results: ./results

# Quais recortes rodar em geraResultados.py.
setups_to_run:
  - asahi_rect

# Catálogo de recortes; cada um aponta para um arquivo em configs/.
setups:
  - { name: sahi,       config: configs/sahi.yaml }
  - { name: asahi,      config: configs/asahi.yaml }
  - { name: asahi_rect, config: configs/asahi_rect.yaml }
  - { name: all_640,    config: configs/all_640.yaml }
```

Cada `configs/<recorte>.yaml` tem o bloco de configuração do recorte **e a lista
de modelos a avaliar**:

```yaml
# configs/asahi_rect.yaml
dataset:
  input_path: ./dataset/all
  output_path: ./dataset/asahi_rect     # o basename ('asahi_rect') nomeia o recorte
slicing:
  mode: asahi_rect
  tile_size: [640, 640]
  overlap_ratio: 0.15
crossfolds:
  n_folds: 5
  seed: 42
  ioa_threshold: 0.4
  split_strategy: kfold_holdout
  val_ratio: 0.15
  empty_tile_ratio: 0.08
inference:
  suppression: cluster_diou_nms
  conf_threshold: 0.5
  iou_threshold: 0.5
  batch_size: 32
# Modelos a avaliar. Troque aqui para rodar outro modelo — NÃO crie um setup por
# modelo. O prefixo do nome define o engine (YOLO* -> Ultralytics, FASTER ->
# faster_rcnn, DETR -> detr) e o nome é a pasta de pesos correspondente.
models: [YOLOV8L]
```

**Contrato de saída (fixo).** A avaliação grava por modelo em
`results/<recorte>/<modelo>/fold_N/` e o rótulo no CSV é `<RECORTE>_<MODELO>`
(ex.: `ASAHI_RECT_YOLOV8L`). Assim, avaliar a YOLOv8n e a YOLOv8l no mesmo
recorte não se sobrescreve — vão para pastas e rótulos distintos. Para adicionar
um modelo novo, inclua o nome em `models:` e coloque os pesos em
`pesos/<recorte>/model_checkpoints/fold_N/<modelo>/`.

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

`main.py` gera os datasets dos recortes em `setups_to_run`. Gerar todos os
recortes selecionados:

```bash
python main.py --yes
```

`--process N` restringe pelo índice (1..N) na ordem de `setups_to_run`:

```bash
python main.py --process 1 --yes
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

> **Variantes YOLO (ex.: nano vs large).** O treinador sempre grava a pasta
> `YOLOV8`. Para conviver com múltiplas variantes no mesmo recorte, renomeie a
> pasta de pesos após o treino para o nome do modelo que a avaliação vai usar —
> por exemplo `YOLOV8` → `YOLOV8N`, e coloque a versão large em `YOLOV8L`:
>
> ```text
> pesos/asahi_rect/model_checkpoints/fold_N/YOLOV8N/   # treino nano (renomeado)
> pesos/asahi_rect/model_checkpoints/fold_N/YOLOV8L/   # treino large
> ```
>
> Depois rode `python scripts/link_pesos_checkpoints.py` e liste o nome
> correspondente em `models:` no `configs/<recorte>.yaml`.

O avaliador principal deste repositório resolve checkpoints por
`models/<recorte>/fold_N/<MODELO>/manifest.json`, onde `<MODELO>` é o nome livre
usado em `models:` (o prefixo define o engine). Gere esses manifestos a partir
das pastas em `pesos/` com:

```bash
python scripts/link_pesos_checkpoints.py
```

O script varre `pesos/<recorte>/model_checkpoints/fold_N/<MODELO>/`, infere o
engine pelo prefixo do nome e escreve o manifesto na posição canônica. Exemplos:

```text
pesos/asahi_rect/model_checkpoints/fold_1/YOLOV8N/... -> models/asahi_rect/fold_1/YOLOV8N/manifest.json
pesos/asahi_rect/model_checkpoints/fold_1/YOLOV8L/... -> models/asahi_rect/fold_1/YOLOV8L/manifest.json
pesos/asahi_rect/model_checkpoints/fold_1/Faster/...  -> models/asahi_rect/fold_1/Faster/manifest.json
pesos/asahi_rect/model_checkpoints/fold_1/Detr/...    -> models/asahi_rect/fold_1/Detr/manifest.json
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
```

Renomeie a pasta para a variante desejada (ex.: `YOLOV8N`) e gere o manifesto
com `python scripts/link_pesos_checkpoints.py`:

```text
models/asahi_rect/fold_1/YOLOV8N/manifest.json
```

O manifesto aponta para o checkpoint real e registra recorte, fold e engine.

## 3. Avaliação Dos Resultados E Visualizações

Depois do treinamento, rode:

```bash
python geraResultados.py
```

O script:

- lê os recortes em `setups_to_run` e os `models:` de cada `configs/<recorte>.yaml`;
- percorre `recorte x fold x modelo`;
- resolve checkpoints via `models/<recorte>/fold_N/<modelo>/manifest.json`;
- carrega as imagens originais do split `test`;
- executa inferência na imagem inteira e nos tiles;
- reprojeta predições para o espaço normalizado da imagem original;
- aplica a supressão configurada (`nms` ou `cluster_diou_nms`);
- calcula `mAP50`, `mAP75`, `mAP`, precisão, recall, F1, MAE, RMSE e correlação de Pearson;
- **acrescenta** (append) as linhas em `results/results.csv` e grava visualizações.

O CSV é append-only: cada execução soma linhas com o rótulo `<RECORTE>_<MODELO>`,
sem apagar resultados anteriores. Se rodar o mesmo recorte/modelo duas vezes, as
linhas duplicam — remova as antigas manualmente se necessário.

Arquivos principais:

```text
results/results.csv                       # rótulo: <RECORTE>_<MODELO>
results/counting.csv
results/<recorte>/<modelo>/fold_N/*_eval.jpg
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
    "model_path": "./pesos/sahi/model_checkpoints/fold_1/YOLOV8N/train/weights/best.pt",
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
├── config.yaml                     # índice de setups (setups_to_run + catálogo)
├── configs/                        # um <recorte>.yaml por recorte (config + models)
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

Checkpoint para avaliação — `<MODELO>` é o nome livre usado em `models:`
(o prefixo define o engine e o arquivo de peso esperado):

```text
pesos/<recorte>/model_checkpoints/fold_N/<YOLO*>/train/weights/best.pt   # ex. YOLOV8N, YOLOV8L
pesos/<recorte>/model_checkpoints/fold_N/<FASTER*>/best.pth
pesos/<recorte>/model_checkpoints/fold_N/<DETR*>/training/best_model.pth
models/<recorte>/fold_N/<MODELO>/manifest.json                            # gerado por link_pesos_checkpoints.py
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
