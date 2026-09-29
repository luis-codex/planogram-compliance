"""Convierte detecciones sueltas en una grilla de filas ordenadas.

Los productos de una misma repisa están apoyados sobre la misma tabla, así que
su borde inferior (y2) es casi el mismo aunque tengan alturas distintas. Por eso
agrupamos por y2 en vez de por el centro.
"""

import statistics

from .detection import Detection


def group_rows(detections: list[Detection], tol: float = 0.35) -> list[list[Detection]]:
    """Agrupa en filas (de arriba hacia abajo) y ordena cada fila de izquierda a derecha.

    `tol` es la distancia máxima entre bordes inferiores, como fracción de la
    altura mediana de los productos, para considerar que están en la misma fila.
    """
    if not detections:
        return []
    median_h = statistics.median(d.height for d in detections)
    rows: list[list[Detection]] = []
    for d in sorted(detections, key=lambda d: d.bottom):
        if rows and abs(d.bottom - statistics.mean(x.bottom for x in rows[-1])) < tol * median_h:
            rows[-1].append(d)
        else:
            rows.append([d])
    return [sorted(row, key=lambda d: d.cx) for row in rows]


def drop_outside_shelf(rows: list[list[Detection]], image_width: int, max_gap: float = 2.5) -> list[list[Detection]]:
    """Quita de los extremos de cada fila productos que probablemente no son de este mueble.

    - Producto cortado por el borde de la foto (toca el borde y es más angosto que
      lo normal): suele ser del mueble vecino y además no se puede identificar bien.
    - Grupo de 1-2 productos separado del resto por más de `max_gap` anchos de producto.
    """

    def cut(d: Detection, w: float) -> bool:
        touches = d.box[0] <= 2 or d.box[2] >= image_width - 2
        return touches and d.width < 0.8 * w

    cleaned = []
    for row in rows:
        row = list(row)
        while len(row) > 2:
            w = statistics.median(d.width for d in row)
            if cut(row[-1], w):
                row.pop()
            elif cut(row[0], w):
                row.pop(0)
            elif row[-1].box[0] - row[-2].box[2] > max_gap * w:
                row.pop()
            elif row[1].box[0] - row[0].box[2] > max_gap * w:
                row.pop(0)
            else:
                break
        cleaned.append(row)
    return cleaned
