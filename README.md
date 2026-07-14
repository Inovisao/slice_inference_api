# Slice Inference API

Ferramentas para preparar datasets e detectar pequenos objetos em imagens de alta
resolução por inferência fatiada. O repositório cobre quatro etapas diferentes:

```text
imagens + anotações COCO
          │
          ▼
limpeza, recorte e geração de folds YOLO       python main.py
          │
          ▼
treinamento por fold                           python -m train.train
          │
          ▼
checkpoint treinado
          ├── inferência operacional           API FastAPI
          └── avaliação cross-fold             geraResultados.py
```

O modelo não é treinado pela API. Primeiro se gera o dataset, depois se treina um
modelo com cada `fold_N.yaml`; o checkpoint resultante é então usado na API ou na
avaliação.

## Instalação

Requer Python 3.10 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para desenvolver ou executar testes:

```bash
pip install -r requirements-dev.txt
```

## 1. Dataset de entrada

A preparação exige imagens e um arquivo COCO. O arquivo original nunca é
sobrescrito.

```text
dataset/
├── _annotations.coco.json
├── imagem_01.jpg
├── imagem_02.jpg
└── ...
```

Durante o preprocessamento são removidas referências inválidas e caixas
degeneradas, caixas que ultrapassam a imagem são limitadas às suas dimensões e o
resultado é salvo como `dataset/_annotations_clean.coco.json`.

Para usar apenas a inferência, `dataset/` pode conter somente imagens.

## 2. Configuração

`config.yaml` possui caminhos globais e uma lista de processos. O arquivo atual
gera experimentos SAHI, ASAHI quadrado e ASAHI retangular separadamente. Os três
usam o mesmo dataset, seed e divisão das imagens originais.

```yaml
paths:
  source_dataset: ./dataset
  generated_datasets: ./output
  models: ./models
  results: ./results

processes:
  - index: 1
    dataset:
      input_path: ./dataset
      output_path: ./output/sahi
    slicing:
      mode: sahi
      tile_size: [640, 640]
      overlap_ratio: 0.1
    crossfolds:
      n_folds: 5
      seed: 42
      ioa_threshold: 0.4
      val_ratio: 0.15
      empty_tile_ratio: 0.08
    inference:
      suppression: nms
      conf_threshold: 0.5
      iou_threshold: 0.5
      batch_size: 32
```

`val_ratio` é a fração retirada do conjunto que sobra depois da seleção do teste.
No K-fold, a fração de teste é `1 / n_folds`; portanto ela não é configurada
separadamente. `ioa_threshold` é a cobertura mínima da anotação original exigida
para manter uma caixa cortada. `empty_tile_ratio` limita a quantidade de tiles
sem objetos em relação aos exemplos anotados.

Os caminhos globais são usados pela API, treinamento e avaliação. Cada processo
mantém seu próprio `dataset.output_path`, pois os tiles produzidos por cada modo
são distintos. O processo `index: 3` usa `mode: asahi_rect` e grava em
`./output/asahi_rect`.

## 3. Gerar os datasets treináveis

```bash
python main.py
```

Para gerar somente o experimento retangular sem refazer os demais:

```bash
python main.py --process 3 --yes
```

Antes de escrever, o comando mostra a geometria estimada, espaço em disco e pede
confirmação. Para cada fold ele produz:

```text
output/sahi/
├── fold_1/
│   ├── train/images/    # imagem inteira letterbox + tiles anotados/amostrados
│   ├── train/labels/    # labels YOLO correspondentes
│   ├── val/images/      # imagens originais, sem recorte
│   ├── val/labels/
│   ├── test/images/     # imagens originais, sem recorte
│   └── test/labels/
├── fold_1.yaml
├── fold_1_stats.json
├── summary_report.json
├── resolution_groups.csv
└── per_image_metrics.csv
```

Somente o treino é materializado com tiles. Validação e teste preservam a
resolução original e o framework de treinamento aplica seu próprio letterbox.
Executar novamente um fold substitui a saída física daquele fold.

## 4. Treinar

Treinar um fold com o modelo inicial incluído no repositório:

```bash
PYTHONPATH=src python -m train.train \
  --data output/sahi/fold_1.yaml \
  --mode sahi \
  --model src/train/yolo26n.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0
```

Treinar todos os folds configurados de um modo:

```bash
PYTHONPATH=src python -m train.train \
  --all-folds \
  --mode asahi_rect \
  --epochs 100 \
  --device 0
```

Smoke test curto em CPU antes do treinamento completo:

```bash
PYTHONPATH=src python -m train.train \
  --data output/asahi_rect/fold_1.yaml \
  --mode asahi_rect --epochs 1 --batch 2 --workers 0 \
  --fraction 0.01 --no-val --device cpu
```

A saída segue este contrato:

```text
models/sahi/fold_1/yolo/
├── manifest.json
└── train/weights/best.pt
```

O manifesto registra dataset, checkpoint e hiperparâmetros. Avaliações de Faster
R-CNN e DETR também devem fornecer um manifesto na mesma estrutura, usando as
pastas `faster_rcnn` e `detr` respectivamente.

## 5. Inferência pela API

```bash
bash run.sh
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

Inferência em uma imagem específica do dataset configurado:

```bash
curl -X POST http://localhost:8000/inference/single_image \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/sahi/fold_1/yolo/train/weights/best.pt",
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

Se `image_name` for omitido, uma imagem é escolhida aleatoriamente. Se
`include_full_image` for omitido, o padrão é `false` para SAHI e `true` para
ASAHI. A inferência sobre os tiles sempre acontece; a passagem pela imagem
inteira, quando habilitada, acontece exatamente uma vez.

Resultados visuais são escritos em `paths.generated_datasets/<id>/`. Em
`POST /inference/dataset`, o modelo é carregado uma única vez e cada falha é
retornada em `failed_images` com sua causa.

### Endpoints principais

| Método | Endpoint | Finalidade |
|---|---|---|
| `GET` | `/dataset/validate` | Validar imagens e COCO |
| `POST` | `/dataset/clean` | Gerar o COCO normalizado |
| `POST` | `/slicing/single_image` | Inspecionar tiles de uma imagem aleatória |
| `POST` | `/slicing/dataset` | Inspecionar tiles de todas as imagens |
| `POST` | `/inference/single_image` | Inferir uma imagem |
| `POST` | `/inference/dataset` | Inferir todo o diretório configurado |
| `POST` | `/reconstruct/single_image` | Reconstruir tiles salvos |
| `POST` | `/reconstruct/validate` | Inferir usando a configuração de um recorte salvo |

A API operacional carrega checkpoints compatíveis com Ultralytics YOLO. Faster
R-CNN e DETR são suportados no pipeline experimental de avaliação, não nesses
endpoints.

## Como a inferência funciona

```text
imagem original
   ├── passagem inteira opcional ───────────────┐
   └── SAHI/ASAHI → tiles → batches do modelo ─┤
                                                ▼
                              reprojeção para [0, 1]
                                                ▼
                         supressão por classe + visualização
```

Uma caixa do tile é reprojetada para a imagem original por:

```python
gx1 = (tile_x1 + x_offset) / image_width
gy1 = (tile_y1 + y_offset) / image_height
gx2 = (tile_x2 + x_offset) / image_width
gy2 = (tile_y2 + y_offset) / image_height
```

As supressões disponíveis são `nms`, `bws`, `nms_ioa`, `wbf` e
`cluster_diou_nms`. Métodos originalmente agnósticos a classe são aplicados
separadamente por classe.

## SAHI e ASAHI

SAHI significa *Slicing Aided Hyper Inference*. Neste projeto, o modo `sahi`
implementa uma grade de tiles fixos, normalmente 640×640, com stride calculado a
partir de `overlap_ratio`.

O modo `asahi` calcula um tile quadrado adaptativo a partir da resolução da
imagem e distribui as posições uniformemente até as bordas. No dataset de treino,
esses tiles são redimensionados para o `tile_size` configurado antes de serem
gravados. Na inferência YOLO, o tile adaptativo é entregue ao Ultralytics com
`imgsz=640`, que realiza o resize e devolve caixas no espaço do tile recebido.

O modo experimental `asahi_rect` preserva a quantidade de colunas sugerida pelo
ASAHI, calcula a quantidade de linhas pela proporção da imagem e resolve largura
e altura do tile independentemente para manter o overlap em ambos os eixos. Para
4032×2268 e overlap de 15%, ele gera uma grade 4×2 de tiles 1136×1226: 8 tiles e
aproximadamente 21,8% de redundância, contra 12 tiles e 69,6% no ASAHI quadrado.
No treino, cada tile retangular recebe letterbox para 640×640 e as anotações são
transformadas com o mesmo scale e padding, sem distorcer a imagem.

## Avaliação cross-fold 

```bash
python geraResultados.py
```

O script avalia a imagem inteira e os tiles para YOLO, Faster R-CNN e DETR,
aplica a supressão configurada e grava métricas e visualizações em `paths.results`.
Um checkpoint ausente é relatado e o par arquitetura/fold é ignorado.

O preflight com ativos externos é executado separadamente:

```bash
pytest -m integration
```

## Testes

```bash
pytest
```

A suíte padrão não exige GPU, datasets reais ou checkpoints externos. Ela cobre
geometria, geração de folds, formato de labels, inferência mockada, supressão,
métricas e configuração.

## Referências

- ASAHI: <https://arxiv.org/abs/2604.19233>
- SAHI: <https://ieeexplore.ieee.org/document/9897990>
