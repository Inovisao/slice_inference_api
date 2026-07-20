# Threshold Analysis Dashboard

Análise de sensibilidade ao threshold de confiança para os detectores, por modo de
recorte (SAHI / ASAHI / ASAHI_RECT / all_640) e arquitetura (YOLO / Faster R-CNN /
DETR), com protocolo sem vazamento (ajuste no `val`, leitura no `test`).

O HTML (`threshold_dashboard.html`) é versionado aqui; os dados que ele carrega são
gerados em `results/threshold_analysis/` (ignorado pelo git, regenerável pelos scripts).

## Como abrir o dashboard

`threshold_dashboard.html` carrega `results/threshold_analysis/dashboard_data.json` via
`fetch`, que o navegador bloqueia sob `file://`. Sirva a **raiz do projeto** por HTTP
(o caminho do fetch é relativo à raiz):

```bash
# a partir da raiz do repositório
python -m http.server 8080
```

Abra <http://localhost:8080/dashboards/threshold_dashboard.html>.

Controles: slider de threshold, split (val/test), modelo, métrica do boxplot, e quais
modos exibir. O botão "melhor F1 no val" posiciona o slider no threshold que o `val`
elegeria — o número que se reporta no artigo.

## Pipeline de regeneração

```bash
# 1. Coleta detecções brutas (pré-supressão, conf>=0.1) — usa GPU, ~1-2h
python scripts/collect_raw_detections.py

# 2. Pré-computa as métricas por threshold (CPU, ~15 min) -> dashboard_data.json
python scripts/threshold_precompute.py

# 3. Sirva a raiz e abra o dashboard (acima)
```

Passo 1 grava caches reutilizáveis em `results/threshold_analysis/raw/`; passo 2 só lê
esses caches (nunca dispara GPU). Para varrer outros thresholds ou métricas, basta
reeditar `METRIC_THRESHOLDS` em `scripts/threshold_precompute.py` e rerodar o passo 2.

## Arquivos

| Arquivo | Versionado? | Conteúdo |
|---|---|---|
| `dashboards/threshold_dashboard.html` | sim | Dashboard interativo (standalone, só depende do JSON) |
| `results/threshold_analysis/dashboard_data.json` | não (gerado) | Métricas pré-computadas por (split, modo, modelo, threshold, fold) |
| `results/threshold_analysis/raw/*.pkl` | não (gerado) | Detecções brutas pré-supressão (piso conf 0.1), por arquitetura/split |
| `scripts/collect_raw_detections.py` | sim | Coleta GPU das detecções brutas |
| `scripts/threshold_precompute.py` | sim | Sweep offline → dashboard_data.json |

## Método

- **Filtro antes da supressão.** A confiança corta as detecções antes de NMS/Cluster-DIoU;
  filtrar depois não reproduz o efeito real do threshold.
- **Supressão é passo do framework**, fixa por modo: `nms` (SAHI, all_640),
  `cluster_diou_nms` (ASAHI, ASAHI_RECT). Não é variável de ablação aqui.
- **Sem vazamento.** O threshold ótimo é escolhido pela curva do `val`; a métrica final
  é lida no `test` com esse threshold fixo. O dashboard mostra ambos os splits e marca o
  threshold que o val elegeria.
- **mAP não é truncado.** mAP percorre a curva precisão-recall inteira; reportá-lo num
  threshold alto o subestima. Use o mAP no piso de coleta como métrica de ranking.
