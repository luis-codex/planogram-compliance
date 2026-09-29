"""Comparación entre el planograma esperado y lo detectado en el estante.

Cada fila se compara con un alineamiento de secuencias (Needleman-Wunsch). Así un
producto faltante no "corre" todas las posiciones siguientes, como pasaría si
comparáramos posición por posición:

    esperado:  A A B B C
    detectado: A B B C        -> 1 faltante (A), no 4 errores

Además, los huecos físicos entre productos se convierten en "espacios vacíos"
explícitos antes de alinear. Así, si falta una lata en medio de `A A A`, se
marca la posición correcta (la del hueco) y no cualquiera de las tres.

Las filas también se alinean entre sí: una foto puede mostrar solo parte del
estante o una repisa que no está en el planograma, así que la fila 0 detectada no
necesariamente es la fila 0 del planograma.

Estados (los mismos que usa el sistema de 7-Eleven, Sci. Rep. 2025):

| estado        | significado                                                        |
|---------------|--------------------------------------------------------------------|
| `ok`          | el SKU esperado está en su lugar                                   |
| `misplaced`   | hay otro SKU en esa posición (wrong location)                      |
| `wrong_shelf` | hay un SKU que según el planograma va en otra repisa               |
| `missing`     | no hay producto donde se espera uno (out of stock)                 |
| `extra`       | producto que no corresponde a ninguna posición del planograma      |
"""

import statistics
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

import pandas as pd

from .detection import Detection

OK, MISPLACED, WRONG_SHELF, MISSING, EXTRA = "ok", "misplaced", "wrong_shelf", "missing", "extra"
STATUSES = (OK, MISPLACED, WRONG_SHELF, MISSING, EXTRA)

Box = tuple[float, float, float, float]


@dataclass
class SlotResult:
    row: int
    slot: int | None  # posición en el planograma (None si es un producto extra)
    status: str
    expected: str | None
    found: str | None
    box: tuple[float, float, float, float] | None  # caja detectada, o estimada si falta
    conf: float | None = None


@dataclass
class ComplianceReport:
    shelf_id: str
    slots: list[SlotResult]
    rows_expected: int
    rows_detected: int
    row_matches: list[tuple[int, int]]  # (fila planograma, fila detectada)
    row_boxes: dict[int, Box]  # caja que envuelve cada fila detectada, por fila del planograma
    facings: pd.DataFrame = field(repr=False)

    def unlocated_missing(self) -> dict[int, Counter]:
        """Faltantes sin hueco visible, por fila: el estante tiene menos facings de los esperados."""
        out: dict[int, Counter] = {}
        for s in self.slots:
            if s.status == MISSING and s.box is None:
                out.setdefault(s.row, Counter())[s.expected] += 1
        return out

    @property
    def rows_not_visible(self) -> list[int]:
        """Filas del planograma que no aparecen en la foto (no cuentan para el score)."""
        matched = {r for r, _ in self.row_matches}
        return [r for r in range(self.rows_expected) if r not in matched]

    @property
    def expected_slots(self) -> int:
        return sum(1 for s in self.slots if s.slot is not None)

    @property
    def score(self) -> float:
        """Porcentaje de posiciones del planograma con el SKU correcto."""
        ok = sum(1 for s in self.slots if s.status == OK)
        return ok / self.expected_slots if self.expected_slots else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in s.__dict__.items() if k != "box"} for s in self.slots])

    def issues(self) -> pd.DataFrame:
        df = self.to_dataframe()
        return df[df.status != OK].reset_index(drop=True)

    def summary(self) -> dict:
        counts = Counter(s.status for s in self.slots)
        return {
            "shelf_id": self.shelf_id,
            "compliance": round(self.score * 100, 1),
            "rows_expected": self.rows_expected,
            "rows_detected": self.rows_detected,
            "rows_matched": len(self.row_matches),
            **{k: counts.get(k, 0) for k in STATUSES},
        }


GAP_MATCH = 0.5  # costo de asignar un producto esperado a un hueco físico (< 1 = faltante normal)
GAP_SKIP_EDGE = 0.0  # ignorar un hueco al inicio/fin de la fila (puede ser solo que la fila es más corta)
GAP_SKIP_INNER = 0.8  # ignorar un hueco entre dos productos (casi siempre es un faltante real)
POS_WEIGHT = 0.4  # penaliza emparejar posiciones relativas muy distintas (desempata casos ambiguos)


def align(
    expected: list[str], found: list[str | None], gap_skip: list[float] | None = None
) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch: devuelve pares (índice esperado, índice detectado).

    `found` puede tener `None`, que representa un hueco físico en el estante, y
    `gap_skip[j]` es el costo de ignorar ese hueco (0 por defecto).
    Costos: coincidencia 0, SKU distinto 1, faltante/extra 1, esperado↔hueco
    `GAP_MATCH`. Con estos costos un SKU equivocado en
    su lugar se prefiere a "falta uno + sobra otro", y un faltante se ubica
    preferentemente donde hay un hueco. Cada emparejamiento suma además
    `POS_WEIGHT` × la diferencia entre su posición relativa en la fila esperada y
    en la detectada, para preferir emparejar cosas que están en el mismo lugar.
    """
    n, m = len(expected), len(found)

    def pair_cost(i: int, j: int) -> float:
        pos = POS_WEIGHT * abs((i + 0.5) / n - (j + 0.5) / m)
        if found[j] is None:
            return GAP_MATCH + pos
        return (0.0 if expected[i] == found[j] else 1.0) + pos

    def skip_found(j: int) -> float:
        if found[j] is None:
            return gap_skip[j] if gap_skip else 0.0
        return 1.0

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + skip_found(j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + pair_cost(i - 1, j - 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + skip_found(j - 1),
            )

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    eps = 1e-9
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i][j] - cost[i - 1][j - 1] - pair_cost(i - 1, j - 1)) < eps:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(cost[i][j] - cost[i - 1][j] - 1) < eps:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]




@dataclass
class Gap:
    box: Box
    edge: bool  # True si está al inicio o al final de la fila


def find_gaps(
    row: list[Detection], shelf_left: float, shelf_right: float, n_expected: int, min_frac: float = 0.5
) -> list[Detection | Gap]:
    """Intercala en la fila huecos donde hay espacio para uno o más productos.

    Devuelve una lista mezclando `Detection` y `Gap`, ordenada de izquierda
    a derecha. `shelf_left`/`shelf_right` son los bordes del estante (para
    detectar faltantes al inicio y al final de la fila). Como las filas pueden
    ser más cortas que el estante, los huecos del final solo se agregan si a la
    fila le faltan productos respecto a los `n_expected` del planograma.
    """
    if not row:
        return []
    w = statistics.median(d.width for d in row)
    spaces = [b.box[0] - a.box[2] for a, b in pairwise(row)]
    spacing = max(0.0, statistics.median(spaces)) if spaces else 0.0
    y1 = statistics.median(d.box[1] for d in row)
    y2 = statistics.median(d.box[3] for d in row)

    def gaps(left: float, right: float, edge: bool) -> list[Gap]:
        free = right - left - spacing
        if free < min_frac * w:
            return []
        n = max(1, round(free / (w + spacing)))
        step = (right - left) / n
        return [Gap((left + k * step, y1, left + (k + 1) * step, y2), edge) for k in range(n)]

    out: list[Detection | Gap] = list(gaps(shelf_left - spacing, row[0].box[0], edge=True))
    for a, b in pairwise(row):
        out.append(a)
        out += gaps(a.box[2], b.box[0], edge=False)
    out.append(row[-1])
    deficit = n_expected - len(out)
    if deficit > 0:
        out += gaps(row[-1].box[2], shelf_right + spacing, edge=True)[:deficit]
    return out


ROW_SKIP = 0.6  # costo por producto de dejar una fila sin emparejar (fuera de vista / fuera del planograma)


def _check_row(
    r: int, exp: list[str], dets: list[Detection], shelf_left: float, shelf_right: float
) -> list[SlotResult]:
    tokens = find_gaps(dets, shelf_left, shelf_right, len(exp))
    found_tokens = [t.sku if isinstance(t, Detection) else None for t in tokens]
    gap_skip = [GAP_SKIP_EDGE if isinstance(t, Gap) and t.edge else GAP_SKIP_INNER for t in tokens]

    results = []
    for ei, fi in align(exp, found_tokens, gap_skip):
        t = tokens[fi] if fi is not None else None
        d = t if isinstance(t, Detection) else None
        if ei is None:
            if d is None:
                continue  # hueco que no corresponde a ningún producto esperado
            status = EXTRA
        elif d is None:
            status = MISSING
        else:
            status = OK if d.sku == exp[ei] else MISPLACED
        results.append(
            SlotResult(
                row=r,
                slot=ei,
                status=status,
                expected=exp[ei] if ei is not None else None,
                found=d.sku if d else None,
                box=d.box if d else _gap_box(t),
                conf=d.conf if d else None,
            )
        )
    return results


def _gap_box(t: Detection | Gap | None) -> Box | None:
    """Un faltante solo se ubica en la imagen si cayó en un hueco *entre* dos productos.

    Los huecos al inicio/fin de fila no son evidencia confiable (con perspectiva
    cada repisa empieza en una x distinta), y si no hay hueco físico el faltante
    significa "hay menos facings de los esperados", sin una posición concreta.
    """
    return t.box if isinstance(t, Gap) and not t.edge else None


def _match_rows(pairwise: list[list[float]], n_exp: list[int], n_det: list[int]) -> list[tuple[int, int]]:
    """Alinea filas del planograma con filas detectadas (en orden, de arriba hacia abajo)."""
    n, m = len(n_exp), len(n_det)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i and j and cost[i - 1][j - 1] + pairwise[i - 1][j - 1] < cost[i][j]:
                cost[i][j], back[i][j] = cost[i - 1][j - 1] + pairwise[i - 1][j - 1], "match"
            if i and cost[i - 1][j] + ROW_SKIP * n_exp[i - 1] < cost[i][j]:
                cost[i][j], back[i][j] = cost[i - 1][j] + ROW_SKIP * n_exp[i - 1], "skip_exp"
            if j and cost[i][j - 1] + ROW_SKIP * n_det[j - 1] < cost[i][j]:
                cost[i][j], back[i][j] = cost[i][j - 1] + ROW_SKIP * n_det[j - 1], "skip_det"
    pairs, i, j = [], n, m
    while i or j:
        move = back[i][j]
        if move == "match":
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "skip_exp":
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def check(planogram: dict, rows: list[list[Detection]]) -> ComplianceReport:
    """Compara el planograma (`{"shelf_id", "rows"}`) contra las filas detectadas."""
    expected_rows: list[list[str]] = planogram["rows"]
    shelf_left = min((row[0].box[0] for row in rows if row), default=0.0)
    shelf_right = max((row[-1].box[2] for row in rows if row), default=0.0)

    # 1) comparar cada fila esperada contra cada fila detectada y quedarnos con el mejor emparejamiento
    results = [[_check_row(i, exp, dets, shelf_left, shelf_right) for dets in rows] for i, exp in enumerate(expected_rows)]
    row_costs = [[float(sum(s.status != OK for s in res)) for res in row_res] for row_res in results]
    matches = _match_rows(row_costs, [len(e) for e in expected_rows], [len(d) for d in rows])

    slots: list[SlotResult] = [s for i, j in matches for s in results[i][j]]

    # 2) wrong shelf: el SKU encontrado no va en esta repisa, pero sí en otra
    skus_by_row = [set(e) for e in expected_rows]
    for s in slots:
        in_other_row = any(s.found in skus for k, skus in enumerate(skus_by_row) if k != s.row)
        if s.status in (MISPLACED, EXTRA) and s.found not in skus_by_row[s.row] and in_other_row:
            s.status = WRONG_SHELF

    # 3) facings por SKU y fila
    facings = []
    for i, j in matches:
        exp_count, found_count = Counter(expected_rows[i]), Counter(d.sku for d in rows[j])
        for sku in sorted(set(exp_count) | set(found_count)):
            facings.append({"row": i, "sku": sku, "expected": exp_count[sku], "found": found_count[sku]})
    facings_df = pd.DataFrame(facings)
    if not facings_df.empty:
        facings_df["diff"] = facings_df.found - facings_df.expected

    row_boxes = {
        i: (
            min(d.box[0] for d in rows[j]),
            statistics.median(d.box[1] for d in rows[j]),
            max(d.box[2] for d in rows[j]),
            statistics.median(d.box[3] for d in rows[j]),
        )
        for i, j in matches
        if rows[j]
    }
    return ComplianceReport(
        planogram.get("shelf_id", "?"), slots, len(expected_rows), len(rows), matches, row_boxes, facings_df
    )
