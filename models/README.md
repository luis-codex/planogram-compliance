# Modelos

| carpeta | qué detecta | uso |
|---|---|---|
| `shelf-generic-yolo26m/` | `product` (cualquier producto en góndola) y `gap` (espacio vacío) | **modelo principal**: detectar productos en la góndola |
| `pretrained/yolo26m.pt` | 80 clases de COCO (oficial de Ultralytics) | punto de partida para entrenar |

## shelf-generic-yolo26m

- **Pesos:** `best.pt`. **Umbrales y métricas:** `metadata.json`.
- **Entrenamiento:** YOLO26m, 60 épocas, `imgsz=1024`. Curvas y matrices en `runs/shelf-generic-yolo26m/train/`.
- **Dataset:** `data/datasets/shelf-generic/` (14 944 train / 1 573 val / 3 436 test), que combina:
  - SKU-110K: 11 743 fotos de góndolas con productos etiquetados.
  - Datasets de huecos: 6 861 fotos con espacios vacíos etiquetados.
  - shelf-product: 1 349 fotos con productos y huecos.
  - A cada fuente se le completó por pseudo-etiquetado la clase que no traía (ver `manifest.csv`).
- **Métricas en test** (`runs/shelf-generic-yolo26m/test-1024/`), con el umbral de mejor F1 por clase:

| clase | conf | precision | recall |
|---|---|---|---|
| product | 0.40 | 0.91 | 0.91 |
| gap | 0.26 | 0.78 | 0.76 |

```python
from planogram import detect, load_model

model = load_model("models/shelf-generic-yolo26m/best.pt")
detections = detect(model, "foto.jpg", conf=0.25, imgsz=1024)
products = [d for d in detections if d.sku == "product" and d.conf >= 0.40]
gaps = [d for d in detections if d.sku == "gap" and d.conf >= 0.26]
```
