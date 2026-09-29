"""Quiebres de stock: cruzar los productos detectados con la distribución del planograma.

El planograma solo dice, por estante, qué grupos de productos van y cuántos frentes
tiene cada uno, de izquierda a derecha (sin posiciones exactas). Suponemos que los
frentes de cada estante ocupan todo el ancho del módulo, así que el frente *i*
"debería" estar en una x proporcional a su lugar en la secuencia.

Cada producto detectado se empareja con el frente esperado más cercano, respetando
el orden (programación dinámica). Los frentes que quedan sin producto son quiebres.
El ancho del módulo se toma de todos los estantes juntos: si en un estante falta el
producto del borde, los otros estantes igual marcan dónde empieza y termina.
"""

import statistics
from dataclasses import dataclass

import pandas as pd

from .detection import Detection

Box = tuple[float, float, float, float]

SKIP_COST = 0.6  # en anchos de frente: más barato dejar un frente vacío que emparejar a más de 0.6 frentes de distancia


@dataclass
class Facing:
    nivel: int
    grupo: int  # índice del grupo dentro del estante (el mismo SKU puede aparecer en dos grupos)
    sku: str
    box: Box  # dónde debería estar según el planograma
    detection: Detection | None  # producto emparejado, o None si falta

    @property
    def missing(self) -> bool:
        return self.detection is None


def _match(centers: list[float], width: float, products: list[Detection]) -> list[int | None]:
    """Empareja frentes esperados (centros x) con productos ordenados por x. Devuelve el producto de cada frente."""
    n, m = len(centers), len(products)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * SKIP_COST
    for j in range(1, m + 1):
        cost[0][j] = j * SKIP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(products[j - 1].cx - centers[i - 1]) / width
            cost[i][j] = min(cost[i - 1][j - 1] + dist, cost[i - 1][j] + SKIP_COST, cost[i][j - 1] + SKIP_COST)

    assigned: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        dist = abs(products[j - 1].cx - centers[i - 1]) / width
        if abs(cost[i][j] - (cost[i - 1][j - 1] + dist)) < 1e-9:
            assigned[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(cost[i][j] - (cost[i - 1][j] + SKIP_COST)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return assigned


def find_stockouts(planogram: dict, shelves: list[list[Detection]], modulo: int = 1) -> list[Facing]:
    """Frentes esperados del módulo, cada uno con su producto detectado o marcado como faltante.

    `shelves`: productos detectados agrupados por estante, **de abajo hacia arriba**
    (el primero es el nivel 1), cada estante ordenado de izquierda a derecha.
    """
    estantes = sorted(planogram["modulos"][modulo - 1]["estantes"], key=lambda e: e["nivel"])
    widths = {sku: p.get("ancho_cm", 1.0) for sku, p in planogram["productos"].items()}
    all_products = [d for shelf in shelves for d in shelf]
    left = min(d.box[0] for d in all_products)
    right = max(d.box[2] for d in all_products)

    facings: list[Facing] = []
    for estante, products in zip(estantes, shelves, strict=False):
        expected = [(g, grupo["sku"]) for g, grupo in enumerate(estante["grupos"]) for _ in range(grupo["frentes"])]
        total = sum(widths[sku] for _, sku in expected)
        scale = (right - left) / total
        y1 = statistics.median(d.box[1] for d in products)
        y2 = statistics.median(d.box[3] for d in products)

        boxes, x = [], left
        for _, sku in expected:
            boxes.append((x, y1, x + widths[sku] * scale, y2))
            x += widths[sku] * scale
        centers = [(b[0] + b[2]) / 2 for b in boxes]
        avg_width = (right - left) / len(expected)

        for (g, sku), box, k in zip(expected, boxes, _match(centers, avg_width, products), strict=True):
            facings.append(Facing(estante["nivel"], g, sku, box, products[k] if k is not None else None))
    return facings


def stockout_table(facings: list[Facing]) -> pd.DataFrame:
    """Una fila por grupo del planograma: frentes esperados, detectados y estado."""
    df = pd.DataFrame([{"nivel": f.nivel, "grupo": f.grupo, "sku": f.sku, "missing": f.missing} for f in facings])
    table = df.groupby(["nivel", "grupo", "sku"], sort=True).agg(esperados=("missing", "size"), faltan=("missing", "sum"))
    table["detectados"] = table.esperados - table.faltan
    table["estado"] = "ok"
    table.loc[table.faltan > 0, "estado"] = "quiebre parcial"
    table.loc[table.detectados == 0, "estado"] = "quiebre"
    return table.reset_index().drop(columns="grupo")[["nivel", "sku", "esperados", "detectados", "faltan", "estado"]]
