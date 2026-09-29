"""Inferencia con YOLO26 y conversión a una estructura simple."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class Detection:
    sku: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 en píxeles

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def bottom(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def load_model(weights: str | Path) -> YOLO:
    return YOLO(str(weights))


def detect(model: YOLO, image: str | Path | Image.Image, conf: float = 0.4, imgsz: int = 1024) -> list[Detection]:
    """`imgsz` debe coincidir con el tamaño usado al entrenar (shelf-generic-yolo26m: 1024)."""
    result = next(iter(model.predict(image, conf=conf, imgsz=imgsz, verbose=False)))
    if not isinstance(result, Results):  # predict() está tipado como Results | Tensor
        raise TypeError(f"Se esperaba Results, llegó {type(result).__name__}")
    boxes = result.boxes
    if boxes is None:
        return []
    return [
        Detection(result.names[int(c)], float(p), (x1, y1, x2, y2))
        for (x1, y1, x2, y2), c, p in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
    ]
