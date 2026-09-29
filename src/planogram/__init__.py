"""PoC de planogram compliance con Ultralytics YOLO26."""

from PIL import Image

from .compliance import ComplianceReport, check
from .detection import Detection, detect, load_model
from .layout import drop_outside_shelf, group_rows


def realogram(detections: list[Detection], image_width: int) -> list[list[Detection]]:
    """Detecciones -> filas del estante real (sin productos del mueble vecino)."""
    return drop_outside_shelf(group_rows(detections), image_width)


def run(model, image: Image.Image, planogram: dict, conf: float = 0.4) -> ComplianceReport:
    """Pipeline completo: detectar -> filas -> comparar con el planograma."""
    return check(planogram, realogram(detect(model, image, conf=conf), image.width))


__all__ = ["ComplianceReport", "Detection", "check", "detect", "drop_outside_shelf", "group_rows", "load_model", "realogram", "run"]
