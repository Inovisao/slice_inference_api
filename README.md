# Slice Inference API

Pipeline de detecção de pequenos insetos em imagens agrícolas de alta resolução (4032×2268) usando **Slicing-based Inference** com YOLOv8.

Fazer resize global da imagem para 640×640 destrói a escala dos insetos. A solução é fatiar a imagem em tiles e inferir sobre cada tile — recontextualizando as detecções na imagem original.

---

## Início Rápido

**1. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**2. Organize o dataset:**
```
dataset/
├── imagem_01.jpg
├── imagem_02.jpg
└── ...

models/
└── best.pt       # modelo YOLOv8 treinado
```

**3. Configure o `config.yaml`** conforme o modo desejado (`sahi` ou `asahi`) e ajuste `overlap_ratio`.

**4. Suba a API:**
```bash
bash run.sh
# equivalente a:
# PYTHONPATH=src uvicorn api.api:app --reload
```

A API estará disponível em `http://localhost:8000`.  
Documentação interativa (Swagger): `http://localhost:8000/docs`.

**5. Execute uma inferência:**
```bash
curl -X POST http://localhost:8000/inference/single_image \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/best.pt",
    "slicing_mode": "asahi",
    "overlap_ratio": 0.2,
    "suppression": "wbf",
    "conf": 0.25,
    "device": "cpu"
  }'
```

**6. Fatiamento standalone** (sem inferência, para inspeção dos tiles):
```bash
curl -X POST http://localhost:8000/slicing/single_image \
  -H "Content-Type: application/json" \
  -d '{"slicing_mode": "asahi", "overlap_ratio": 0.2}'
```

Os tiles e o JSON de metadados são salvos em `output/<id_image>/`.

---

## Arquitetura (Clean Room)

O pipeline é dividido em três módulos estritos, sem acoplamento entre eles:

```
Imagem Original (4032×2268)
        │
        ▼
┌───────────────┐
│    Slicer     │  Geometria pura. Usa yield — nunca carrega tudo na RAM.
│  SAHI / ASAHI │  Retorna (tile, {x, y, width, height, ...}) por demanda.
└───────┬───────┘
        │ generator
        ▼
┌───────────────┐
│ TileInference │  Acumula tiles em batches de 32 → 1 roundtrip GPU.
│    Engine     │  model.predict(batch, imgsz=640) → xyxy no espaço do tile.
└───────┬───────┘
        │ raw_boxes (coordenadas normalizadas [0,1] na imagem original)
        ▼
┌───────────────┐
│  Suppression  │  NMS / BWS / NMS-IoA / WBF / Cluster-DIoU-NMS
│  + Visualizer │  Elimina duplicatas entre tiles sobrepostos.
└───────────────┘
```

---

## Modos de Fatiamento

### SAHI — Static Adaptive Head Inference (Fixed Slicing)

Tile fixo (640×640) com stride calculado a partir de `overlap_ratio`.

```
stride_x = tile_w - int(tile_w × overlap_ratio)
stride_y = tile_h - int(tile_h × overlap_ratio)
```

A grade é preenchida com ancoragem na borda direita/inferior, o que cria um **pico de sobreposição** nas margens — efeito inevitável quando a matemática não fecha com as dimensões da imagem.

**Uso típico:** imagens menores ou quando a grade regular é requisito.

---

### ASAHI — Adaptive SAHI (Adaptive Slicing)

Tile adaptativo derivado das dimensões da imagem (Equações 1–4 do paper). O stride é um **número de ponto flutuante** distribuído uniformemente — sem `if` de borda, sem picos.

**Cálculo do tile size `p`** para uma imagem W×H com overlap `l`:

```
ls = 640 × (4 - 3l) + 1

se max(W, H) ≤ ls:
    p = max(W / (3 - 2l) + 1,  H / (2 - l) + 1)
senão:
    p = max(W / (4 - 3l) + 1,  H / (3 - 2l) + 1)

p = ceil(p)
```

**Cálculo da grade `a×b`:**

```
a = ceil((W - p·l) / (p·(1-l)))     ← colunas
b = ceil((H - p·l) / (p·(1-l)))     ← linhas
```

**Posições dos tiles** (stride float, arredondado só no momento do corte):

```
x[i] = round(i × (W - p) / (a - 1))    para i in 0..a-1
y[j] = round(j × (H - p) / (b - 1))    para j in 0..b-1
```

**Exemplo — 4032×2268 com `overlap_ratio=0.2`:**

| Variável | Valor |
|---|---|
| `p` | 1187 px |
| Grade | 4 × 3 = **12 tiles** |
| `stride_x` (float) | 948.33 px |
| `stride_y` (float) | 540.5 px |
| Batches GPU | **1** (12 < batch_size=32) |

O YOLO recebe tiles de 1187×1187, faz o resize `1187→640` internamente na GPU e devolve as `xyxy` no espaço do tile original. A reprojeção para a imagem global é `(bx + x_off) / img_w`.

**Uso típico:** imagens agrícolas de alta resolução onde a redução de tiles é crítica para throughput.

---

## Reprojeção de Coordenadas

As boxes retornadas pelo YOLO estão no espaço do tile (e.g. 1187×1187 px). A reprojeção para a imagem original:

```python
gx1 = max(0.0, min(1.0, (bx1 + x_off) / img_w))
gy1 = max(0.0, min(1.0, (by1 + y_off) / img_h))
gx2 = max(0.0, min(1.0, (bx2 + x_off) / img_w))
gy2 = max(0.0, min(1.0, (by2 + y_off) / img_h))
```

As coordenadas resultantes são normalizadas `[0, 1]` relativas à imagem original.

---

## Métodos de Supressão

| Método | Descrição |
|---|---|
| `nms` | Non-Maximum Suppression clássico |
| `bws` | Box-Weighted Suppression |
| `nms_ioa` | NMS com IoA (Intersection over Area) — favorece caixas pequenas |
| `wbf` | Weighted Boxes Fusion — media ponderada das caixas sobrepostas |
| `cluster_diou_nms` | Cluster + DIoU-NMS — usa distância euclidiana além de IoU |

---

## Configuração

```yaml
# config.yaml

dataset:
  input_path: ./dataset
  output_path: ./output

slicing:
  mode: asahi              # sahi | asahi
  tile_size: [640, 640]    # usado apenas no modo sahi
  overlap_ratio: 0.2       # float em (0, 1) — usado nos dois modos
  min_object_coverage: 0.5

inference:
  suppression: wbf         # nms | bws | nms_ioa | wbf | cluster_diou_nms
  models_path: ./models
  output_results_path: ./output
  conf_threshold: 0.25
  iou_threshold: 0.45
  batch_size: 32
  num_workers: 4
  save_original_annotations: true

crossfolds:
  n_folds: 5
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
```

---

## API Endpoints

### Slicing

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/slicing/single_image` | Fatiamento de 1 imagem aleatória do dataset |
| `POST` | `/slicing/dataset` | Fatiamento de todo o dataset |
| `POST` | `/slicing/dataset/crossFolds` | Fatiamento com cross-validation |

**Payload (`SlicingRequest`):**
```json
{
  "slicing_mode": "asahi",
  "overlap_ratio": 0.2
}
```

**Payload (`CrossFoldsRequest`):**
```json
{
  "slicing_mode": "asahi",
  "n_folds": 5,
  "overlap_ratio": 0.2
}
```

---

### Inference

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/inference/single_image` | Inferência em 1 imagem aleatória do dataset |
| `POST` | `/inference/dataset` | Inferência em todo o dataset |

**Payload (`InferenceRequest`):**
```json
{
  "model_path": "./models/best.pt",
  "slicing_mode": "asahi",
  "conf": 0.25,
  "iou_thr": 0.45,
  "suppression": "wbf",
  "overlap_ratio": 0.2,
  "device": "cuda"
}
```

**Resposta:**
```json
{
  "id_image": 42,
  "image_name": "campo_01.jpg",
  "slicing_mode": "asahi",
  "detections": 17,
  "raw_detections": 31,
  "duplicates_removed": 14,
  "scores": { "min": 0.261, "max": 0.941, "mean": 0.612 },
  "output_path": "./output/42/campo_01_resultado.jpg"
}
```

---

## Estrutura do Projeto

```
slice_inference_api/
├── api/
│   ├── routers/
│   │   ├── inference.py     # Endpoints de inferência
│   │   └── slicing.py       # Endpoints de fatiamento
│   └── helpers.py
├── src/
│   ├── config/
│   │   ├── settings.py      # DataClasses: SlicingConfig, DataInferenceConfig, ...
│   │   └── config_loader.py # Leitura e validação do config.yaml
│   ├── slicing/
│   │   ├── sahi.py          # Motor geométrico SAHI (tile fixo, borda heurística)
│   │   ├── asahi.py         # Motor geométrico ASAHI (tile adaptativo, stride float)
│   │   └── service.py       # make_slicer(), slice_image(), save_slicing_config()
│   ├── inference/
│   │   ├── engine.py        # TileInferenceEngine — batching + reprojeção
│   │   ├── pipeline.py      # InferencePipeline — orquestra engine + supressão
│   │   ├── service.py       # make_inference_pipeline()
│   │   └── visualizer.py    # draw_detections()
│   └── suppression/
│       ├── nms.py
│       ├── bws.py
│       ├── nms_ioa.py
│       ├── wbf.py
│       └── cluster_diou_nms.py
├── config.yaml
└── README.md
```

---

## Decisões de Design

**Por que `imgsz=640` no engine e não `imgsz=p`?**
O YOLO foi treinado em 640×640. Passar `imgsz=p` (e.g. 1187) forçaria a rede a operar em resolução diferente da de treino, degradando precisão. Com `imgsz=640`, o ultralytics faz o resize `p→640` na GPU antes da forward pass e devolve as boxes já remapeadas para o espaço do tile original — sem custo extra para o pipeline.

**Por que não usar capping `min(p, 640)`?**
Cappar `p` em 640 transforma o ASAHI em um SAHI com grade maior — perde a invariante de stride uniforme e multiplica o número de tiles, aumentando o custo de inferência desnecessariamente. O ASAHI é projetado para ter **menos tiles, maiores** — o downscale interno do YOLO é a abstração correta para esse tradeoff.

**Por que generators no Slicer?**
Para imagens 4032×2268 com tiles de 640×640, uma grade pode ter 40+ tiles. Materializar todos na RAM antes da inferência cria pico de memória. O generator entrega tiles sob demanda, permitindo que o engine os acumule em batches de forma streaming.


### Referencias e Agradecimentos

Asahi: https://arxiv.org/abs/2604.19233
Sahi: https://ieeexplore.ieee.org/document/9897990