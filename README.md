# PoC · Planogram compliance con Ultralytics YOLO26

Prueba de concepto basada en el tutorial de Ultralytics
[*Using Ultralytics YOLO26 for planogram compliance detection*](https://www.ultralytics.com/blog/using-ultralytics-yolo26-for-planogram-compliance-detection)
y en el sistema de 7-Eleven Taiwán ([Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-27773-5)).

```
foto de la góndola ──► YOLO26 (productos y huecos) ──► estantes / posiciones ──► comparación con el planograma ──► reporte
```

## Estructura

```
data/
├── datasets/
│   └── shelf-generic/       dataset YOLO: product + gap (data.yaml, manifest.csv, train/valid/test)
└── planograms/              góndolas demo: granos, lácteos, licores
models/
├── README.md                qué es cada modelo, métricas y umbrales
├── pretrained/yolo26m.pt    pesos oficiales (COCO) para entrenar
└── shelf-generic-yolo26m/   modelo principal: best.pt + metadata.json
runs/
└── shelf-generic-yolo26m/   entrenamiento (train/), evaluaciones (test/, test-1024/) y train.log
notebooks/                   (vacío por ahora)
src/planogram/
├── detection.py             inferencia YOLO26
├── layout.py                agrupar detecciones en estantes; descartar el mueble vecino
├── compliance.py            comparación con el planograma (ok / misplaced / wrong_shelf / missing / extra)
└── visualization.py         reporte visual: planograma | detecciones | reporte
```

Las carpetas `data/datasets/`, `models/`, `runs/` y `outputs/` no van a git (pesan GB).

## Dataset recomendado: SKU-110K

Para entrenar el detector de productos se recomienda
[SKU-110K](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)
([docs](https://docs.ultralytics.com/datasets/detect/sku-110k/)): fotos reales de góndolas retail con productos muy juntos.
Ultralytics lo descarga automáticamente la primera vez que se entrena con él (~13.6 GB):

```python
from ultralytics import YOLO

model = YOLO("models/pretrained/yolo26m.pt")
model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
```

SKU-110K solo anota productos (una única clase, sin `gap`). Para detectar huecos hay que
complementarlo con un dataset propio como `shelf-generic`.

## Planogramas

Cada góndola es una matriz de **módulos** (columnas) × **estantes** (niveles), y cada estante tiene posiciones con SKU, ubicación y frentes:

```json
{
  "gondola_id": "gondola_granos",
  "productos": {"arroz_costeno_1kg": {"nombre": "Arroz Costeño 1 kg", "ancho_cm": 11, "alto_cm": 22}},
  "modulos": [
    {"modulo": 1, "ancho_cm": 100, "estantes": [
      {"nivel": 1, "posiciones": [{"sku": "arroz_paisana_5kg", "x_cm": 0, "ancho_cm": 24, "frentes": 4}]}
    ]}
  ]
}
```

`nivel` 1 es el estante de abajo; `x_cm` es la distancia desde el borde izquierdo del módulo.

## Uso

```bash
uv sync
```

```python
from planogram import detect, load_model

model = load_model("models/shelf-generic-yolo26m/best.pt")
detections = detect(model, "foto.jpg", conf=0.25)   # imgsz=1024 por defecto
products = [d for d in detections if d.sku == "product" and d.conf >= 0.40]
gaps = [d for d in detections if d.sku == "gap" and d.conf >= 0.26]
```

Umbrales por clase y métricas: `models/README.md`.
