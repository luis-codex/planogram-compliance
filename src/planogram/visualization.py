"""Dibuja el resultado de compliance sobre la imagen del estante."""

from PIL import Image, ImageDraw, ImageFont

from .compliance import EXTRA, MISPLACED, MISSING, OK, WRONG_SHELF, ComplianceReport
from .detection import Detection
from .stockout import Facing

# mismos colores que la figura del paper / blog: correct, wrong location, out of stock, wrong shelf
COLORS = {OK: (40, 200, 70), MISPLACED: (150, 80, 60), MISSING: (230, 30, 30), WRONG_SHELF: (240, 200, 40), EXTRA: (0, 160, 255)}


def draw_detections(image: Image.Image, detections: list[Detection], font_size: int = 13) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size)
    for det in detections:
        d.rectangle(det.box, outline=(0, 255, 255), width=2)
        d.text((det.box[0] + 2, det.box[1] - font_size - 2), f"{det.sku} {det.conf:.2f}", fill=(0, 255, 255), font=font)
    return img


def draw_report(image: Image.Image, report: ComplianceReport, font_size: int = 14) -> Image.Image:
    img = image.convert("RGB").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size)

    for s in report.slots:
        if s.box is None:
            continue
        color = COLORS[s.status]
        if s.status != OK:
            o.rectangle(s.box, fill=(*color, 70))
        d.rectangle(s.box, outline=color, width=2 if s.status == OK else 4)
        if s.status == MISPLACED:
            label = f"hay {s.found}, va {s.expected}"
        elif s.status == MISSING:
            label = f"falta {s.expected}"
        elif s.status == WRONG_SHELF:
            label = f"otra repisa: {s.found}"
        elif s.status == EXTRA:
            label = f"extra {s.found}"
        else:
            continue
        x, y = s.box[0], max(0, s.box[1] - font_size - 3)
        tb = d.textbbox((x, y), label, font=font)
        d.rectangle(tb, fill=color)
        d.text((x, y), label, fill=(255, 255, 255), font=font)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)

    # faltantes sin hueco visible: se anotan al final de la fila, sin inventar una posición
    for row, missing in report.unlocated_missing().items():
        if row not in report.row_boxes:
            continue
        _, y1, x2, y2 = report.row_boxes[row]
        label = "faltan " + ", ".join(f"{n}x {sku}" for sku, n in missing.items())
        tb = d.textbbox((0, 0), label, font=font)
        x = min(x2 + 6, img.width - (tb[2] - tb[0]) - 4)
        y = (y1 + y2) / 2
        d.rectangle(d.textbbox((x, y), label, font=font), fill=COLORS[MISSING])
        d.text((x, y), label, fill=(255, 255, 255), font=font)
    banner = f"{report.shelf_id}  compliance {report.score * 100:.1f}%"
    big = ImageFont.load_default(size=font_size * 2)
    d.rectangle(d.textbbox((10, 6), banner, font=big), fill=(0, 0, 0))
    d.text((10, 6), banner, fill=(255, 255, 255), font=big)
    return img


STATUS_LABELS = {
    OK: "correcto",
    MISPLACED: "SKU equivocado",
    WRONG_SHELF: "de otra repisa",
    MISSING: "falta",
    EXTRA: "sobra",
}


def draw_planogram(
    planogram: dict, report: ComplianceReport | None = None, width: int = 900, height: int = 600
) -> Image.Image:
    """Dibuja el planograma como grilla. Si se pasa un reporte, colorea cada posición según su estado."""
    status = {(s.row, s.slot): s.status for s in report.slots if s.slot is not None} if report else {}
    visible = {r for r, _ in report.row_matches} if report else None
    rows = planogram["rows"]
    pad = 10
    cell_h = min(160, (height - pad) / max(len(rows), 1) - pad)
    img = Image.new("RGB", (width, height), (245, 245, 245))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    for r, row in enumerate(rows):
        y = pad + r * (cell_h + pad)
        cell_w = (width - 2 * pad) / max(len(row), 1)
        for i, sku in enumerate(row):
            if visible is not None and r not in visible:
                color = (190, 190, 190)  # fila fuera de la foto
            else:
                color = COLORS[status.get((r, i), OK)] if report else (225, 225, 225)
            x = pad + i * cell_w
            d.rectangle((x + 1, y, x + cell_w - 1, y + cell_h), fill=color, outline=(80, 80, 80))
            d.text((x + cell_w / 2, y + cell_h / 2), sku.replace("_", "\n"), fill=(0, 0, 0), font=font, anchor="mm", align="center")
    return img


def _titled(img: Image.Image, title: str, height: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((10_000, height))
    out = Image.new("RGB", (img.width, img.height + 44), (255, 255, 255))
    out.paste(img, (0, 44))
    ImageDraw.Draw(out).text((8, 8), title, fill=(0, 0, 0), font=ImageFont.load_default(size=26))
    return out


def draw_comparison(
    image: Image.Image, planogram: dict, detections: list[Detection], report: ComplianceReport, height: int = 900
) -> Image.Image:
    """Tres paneles como la figura del paper: planograma | lo que ve el modelo | reporte de compliance."""
    font_size = max(12, image.height // 60)
    panels = [
        _titled(draw_planogram(planogram, report, width=int(height * 0.9), height=height), "1. Planograma (lo esperado)", height),
        _titled(draw_detections(image, detections, font_size=font_size), "2. Lo que detecta YOLO26", height),
        _titled(draw_report(image, report, font_size=font_size), "3. Reporte de compliance", height),
    ]
    legend_h = 50
    out = Image.new("RGB", (sum(p.width for p in panels) + 20 * (len(panels) - 1), height + 44 + legend_h), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 20
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=22)
    x, y = 10, height + 44 + 12
    for status, label in STATUS_LABELS.items():
        d.rectangle((x, y, x + 26, y + 26), fill=COLORS[status])
        d.text((x + 34, y), label, fill=(0, 0, 0), font=font)
        x += 60 + int(d.textlength(label, font=font))
    missing = report.unlocated_missing()
    if missing:
        note = "Sin hueco visible (hay menos facings de los esperados): " + "; ".join(
            f"fila {r}: " + ", ".join(f"{n}x {sku}" for sku, n in c.items()) for r, c in missing.items()
        )
        d.text((x + 20, y), note, fill=COLORS[MISSING], font=font)
    return out


def draw_stockouts(image: Image.Image, facings: list[Facing]) -> Image.Image:
    """Frentes con producto en verde; frentes vacíos (quiebres) en rojo con el SKU que falta.

    Cada frente lleva un número en el orden de `facings`: de izquierda a derecha,
    empezando por el estante de abajo.
    """
    img = image.convert("RGB").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o, d = ImageDraw.Draw(overlay), ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, img.height // 50))
    small = ImageFont.load_default(size=max(10, img.height // 70))
    for n, f in enumerate(facings, start=1):
        if f.detection is not None:
            box, color = f.detection.box, COLORS[OK]
            d.rectangle(box, outline=color, width=2)
        else:
            box, color = f.box, COLORS[MISSING]
            o.rectangle(box, fill=(*color, 90))
            d.rectangle(box, outline=color, width=4)
            # SKU en dos líneas y centrado, para que no invada los frentes vecinos
            half = f.sku.count("-") // 2 + 1
            sku = "-".join(f.sku.split("-")[:half]) + "-\n" + "-".join(f.sku.split("-")[half:])
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            d.multiline_text(center, sku, fill=(255, 255, 255), font=small, anchor="mm", align="center",
                             stroke_width=2, stroke_fill=color)
        x, y = box[0] + 3, box[1] + 3
        d.rectangle(d.textbbox((x, y), str(n), font=font), fill=color)
        d.text((x, y), str(n), fill=(255, 255, 255), font=font)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
