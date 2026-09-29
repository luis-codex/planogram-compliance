"""Reporte de quiebres de stock en PDF, con estilo de paper LaTeX (Times).

Un solo reporte para todas las fotos analizadas. Cada foto aporta los mismos datos:
su nombre, la imagen y los frentes que devuelve `find_stockouts`; el resto sale del
planograma.

Requiere `pdflatex` instalado (TeX Live).
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from PIL import Image

from .compliance import MISSING, OK
from .stockout import Facing, stockout_table
from .visualization import COLORS, draw_stockouts

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex(text) -> str:
    return "".join(_LATEX_SPECIAL.get(c, c) for c in str(text))


@dataclass
class Shot:
    """Una foto analizada."""

    name: str
    image: Image.Image
    facings: list[Facing]

    @property
    def expected(self) -> int:
        return len(self.facings)

    @property
    def missing(self) -> int:
        return sum(f.missing for f in self.facings)

    @property
    def availability(self) -> float:
        return 1 - self.missing / self.expected if self.expected else 1.0

    def issues(self) -> list[dict]:
        """Grupos con frentes vacíos, del más afectado al menos, con los números de frente de la foto."""
        positions: dict[tuple[int, str], list[int]] = {}
        for n, f in enumerate(self.facings, start=1):
            if f.missing:
                positions.setdefault((f.nivel, f.sku), []).append(n)
        table = stockout_table(self.facings)
        table = table[table.estado != "ok"].sort_values(["faltan", "nivel"], ascending=[False, True])
        return [
            {"nivel": int(r["nivel"]), "sku": str(r["sku"]), "esperados": int(r["esperados"]),
             "faltan": int(r["faltan"]), "estado": str(r["estado"]),
             "pos": positions.get((int(r["nivel"]), str(r["sku"])), [])}
            for r in table.to_dict("records")
        ]


# Cómo se obtuvieron los datos. Se puede reemplazar con `metodologia=` si los frentes vienen de otra fuente.
# `<<n_fotos_texto>>` se reemplaza por "N fotos".
METODOLOGIA = r"""Se tomaron <<n_fotos_texto>> de la góndola. Un modelo de visión computacional (YOLO) detecta cada
producto visible en la imagen y lo ubica en su estante. Luego se compara con el planograma: cada \emph{frente}
(espacio de un producto, de izquierda a derecha) se empareja con el producto detectado más cercano, y los frentes que
quedan sin producto se cuentan como \emph{quiebres}."""


def _n(n: int, singular: str, plural: str) -> str:
    """"1 frente vacío" / "3 frentes vacíos"."""
    return f"{n} {singular if n == 1 else plural}"


def _pct(x: float) -> str:
    return f"{x * 100:g}\\,\\%"


def _verdict(availability: float, meta: float, critico: float) -> str:
    if availability >= meta:
        return f"dentro de la norma (meta: {_pct(meta)})"
    if availability >= critico:
        return f"por debajo de la meta del {_pct(meta)} y requiere atención"
    return f"en nivel crítico, por debajo del {_pct(critico)} (meta: {_pct(meta)})"


TEMPLATE = r"""
\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.9cm,columnsep=0.75cm,bottom=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands,es-tabla,es-noindentfirst]{babel}
\usepackage{mathptmx}
\renewcommand{\ttdefault}{ptm}  % una sola fuente: Times también para nombres de archivo
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{float}
\usepackage{xcolor}
% mismos colores que las cajas de la foto (visualization.COLORS)
\definecolor{frenteok}{RGB}{<<color_ok>>}
\definecolor{frentevacio}{RGB}{<<color_vacio>>}
\newcommand{\muestra}[1]{\textcolor{#1}{\rule[-0.1ex]{0.8em}{0.8em}}}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage[font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false]{caption}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{fancyhdr}

% estilo tipo SAGE: títulos sin numerar; todo en Times
\titleformat{\section}{\bfseries\large}{}{0pt}{}
\titleformat{\subsection}{\itshape\normalsize}{}{0pt}{}
\titlespacing*{\section}{0pt}{12pt}{5pt}
\titlespacing*{\subsection}{0pt}{9pt}{4pt}
\setlist{nosep,leftmargin=1.4em}
\renewcommand{\arraystretch}{1.1}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\scriptsize\itshape Reporte interno de ejecución en punto de venta -- <<tienda>>}
\fancyfoot[R]{\scriptsize\thepage}

\begin{document}

\twocolumn[{%
\noindent\rule{\textwidth}{0.5pt}\par\vspace{6pt}
\noindent\begin{minipage}[t]{0.66\textwidth}
  \vspace{0pt}
  {\bfseries\LARGE Auditoría de disponibilidad\\[2pt] en góndola: \textit{<<categoria>>}\par}
  \vspace{16pt}
  {\bfseries\large <<autor>>\par}
\end{minipage}\hfill
\begin{minipage}[t]{0.3\textwidth}
  \vspace{0pt}\scriptsize\raggedright
  <<tienda>>\\
  Categoría: <<categoria>>\\
  Góndola: <<gondola>>\\
  Planograma vigente desde <<vigente>>\\
  Fecha de auditoría: <<fecha>>\\[4pt]
  \fbox{\footnotesize USO INTERNO}
\end{minipage}
\par\vspace{22pt}
{\bfseries Resumen\par}\vspace{2pt}
{ <<resumen>>\par}
\vspace{10pt}
{\bfseries Palabras clave\par}\vspace{2pt}
{ Disponibilidad en góndola, quiebres de stock, planograma, visión computacional, \textit{<<categoria>>}\par}
\vspace{22pt}
}]

\section{Introducción}
Un producto que no está en la góndola es una venta que no ocurre. Este reporte mide cuántos de los espacios que el
planograma vigente (desde el <<vigente>>) asigna a la categoría <<categoria>> tienen efectivamente producto a la vista
del cliente, e identifica qué productos deben reponerse y en qué estante.

\section{Metodología}
<<metodologia>>

La métrica principal es la \textbf{disponibilidad en góndola}: el porcentaje de frentes del planograma que tienen
producto visible. Un \emph{quiebre total} indica que el producto no aparece en ninguno de sus frentes del estante;
un \emph{quiebre parcial}, que falta en algunos de ellos. La meta de disponibilidad es del <<meta>>; por debajo
del <<critico>> la situación se considera crítica.

\section{Resultados}
La Tabla~\ref{tab:resumen} resume la disponibilidad por foto. En conjunto, se encontraron <<faltan_texto>>
de <<esperados>> esperados, lo que da una disponibilidad de \textbf{<<disponibilidad>>\,\%}.

\begin{table}[H]
  \centering\small
  \caption{Disponibilidad en góndola por foto analizada.}
  \label{tab:resumen}
  \begin{tabularx}{\columnwidth}{X r r r}
    \toprule
    \textbf{Foto} & \textbf{Frentes} & \textbf{Vacíos} & \textbf{Disp.} \\
    \midrule
<<filas_resumen>>
    \midrule
    \textbf{Total} & \textbf{<<esperados>>} & \textbf{<<faltan>>} & \textbf{<<disponibilidad>>\,\%} \\
    \bottomrule
  \end{tabularx}
\end{table}

<<detalle_fotos>>

\section{Recomendaciones}
<<acciones>>

\section{Conclusión}
<<conclusion>>

\onecolumn
\appendix
\section*{Anexo A. Evidencia fotográfica}
<<anexo>>

\end{document}
"""


def _shot_figure(i: int, shot: Shot) -> str:
    """Foto anotada a todo el ancho, para el anexo."""
    return "\n".join([
        r"\begin{figure}[H]",
        r"  \centering",
        rf"  \includegraphics[width=\textwidth,height=0.4\textheight,keepaspectratio]{{foto{i}.png}}",
        rf"  \caption{{Foto {i} (\texttt{{{_tex(shot.name)}}}). \muestra{{frenteok}}\ Verde: frente con producto. \muestra{{frentevacio}}\ Rojo: frente vacío, "
        r"con el producto que debería estar. Los números corresponden a la columna N.\textsuperscript{o} "
        r"de las tablas de quiebres.}",
        rf"  \label{{fig:foto{i}}}",
        r"\end{figure}",
    ])


def _shot_section(i: int, shot: Shot, name) -> str:
    """Subsección de resultados de una foto: texto y tabla de quiebres (la foto va en el anexo)."""
    issues = shot.issues()
    title = f"Foto {i}: \\texttt{{{_tex(shot.name)}}}"
    if not issues:
        text = (rf"Los {shot.expected} frentes del planograma tienen producto: no se detectaron quiebres "
                rf"(Figura~\ref{{fig:foto{i}}}, en el anexo).")
        return "\n".join([rf"\subsection{{{title}}}", text])

    worst = issues[0]
    text = (
        rf"Hay {_n(shot.missing, 'frente vacío', 'frentes vacíos')} de {shot.expected} "
        rf"(disponibilidad del {shot.availability * 100:.1f}\,\%). "
        rf"El producto más afectado es {_tex(name(worst['sku']))}, en el estante {worst['nivel']}, "
        rf"con {worst['faltan']} de {worst['esperados']} frentes {'vacío' if worst['faltan'] == 1 else 'vacíos'}. "
        rf"La Tabla~\ref{{tab:foto{i}}} lista los quiebres y la Figura~\ref{{fig:foto{i}}} (anexo) los muestra en la foto."
    )
    rows = [
        rf"    {_tex(name(r['sku']))} & {r['nivel']} & {', '.join(map(str, r['pos']))} & "
        rf"{r['esperados'] - r['faltan']}/{r['esperados']} & {'Total' if r['estado'] == 'quiebre' else 'Parcial'} \\"
        for r in issues
    ]
    table = "\n".join([
        r"\begin{table}[H]",
        r"  \centering\small",
        rf"  \caption{{Quiebres detectados en la foto {i}. \textit{{Frentes}}: con producto / esperados.}}",
        rf"  \label{{tab:foto{i}}}",
        r"  \begin{tabularx}{\columnwidth}{X c c c l}",
        r"    \toprule",
        r"    \textbf{Producto} & \textbf{Est.} & \textbf{N.\textsuperscript{o}} & \textbf{Frentes} & \textbf{Quiebre} \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join([rf"\subsection{{{title}}}", text, "", table])


def render_tex(
    shots: list[Shot], planogram: dict, autor: str = "Área de Trade Marketing", fecha: date | None = None,
    meta: float = 0.95, critico: float = 0.85, metodologia: str = METODOLOGIA,
) -> str:
    """Arma el .tex. Las fotos anotadas deben quedar junto al .tex como `foto1.png`, `foto2.png`, ...

    `meta` y `critico`: umbrales de disponibilidad (0-1) para el veredicto.
    `metodologia`: texto LaTeX de cómo se obtuvieron los datos.
    """
    products = planogram["productos"]

    def name(sku: str) -> str:
        return products.get(sku, {}).get("nombre", sku)

    fecha = fecha or datetime.now().astimezone().date()
    esperados = sum(s.expected for s in shots)
    faltan = sum(s.missing for s in shots)
    availability = 1 - faltan / esperados if esperados else 1.0
    n = len(shots)

    # quiebres de todas las fotos, sumados por producto y estante
    total: dict[tuple[str, int], dict] = {}
    for i, shot in enumerate(shots, start=1):
        for r in shot.issues():
            t = total.setdefault((r["sku"], r["nivel"]), {"faltan": 0, "total": False, "fotos": []})
            t["faltan"] += r["faltan"]
            t["total"] |= r["estado"] == "quiebre"
            t["fotos"].append(str(i))
    ranking = sorted(total.items(), key=lambda kv: (not kv[1]["total"], -kv[1]["faltan"]))

    if ranking:
        (top_sku, top_nivel), top = ranking[0]
        resumen = (
            f"Se analizaron {n} foto{'s' if n != 1 else ''} de la góndola de {_tex(planogram.get('categoria', ''))} "
            f"en {_tex(planogram.get('tienda', ''))} y se compararon con el planograma vigente. La disponibilidad "
            f"en góndola es del \\textbf{{{availability * 100:.1f}\\,\\%}} ({esperados - faltan} de {esperados} frentes "
            f"con producto), {_verdict(availability, meta, critico)}. Se detectaron {_n(faltan, 'frente vacío', 'frentes vacíos')} en "
            f"{_n(len(ranking), 'producto', 'productos')}; la prioridad de reposición es {_tex(name(top_sku))} "
            f"en el estante {top_nivel}."
        )
        items = [
            rf"\item \textbf{{{_tex(name(sku))}}}, estante {nivel}: reponer {t['faltan']} "
            rf"{'unidad' if t['faltan'] == 1 else 'unidades'} (foto{'s' if len(t['fotos']) > 1 else ''} "
            rf"{', '.join(t['fotos'])})" + (r". \textit{Producto ausente: prioridad alta.}" if t["total"] else ".")
            for (sku, nivel), t in ranking
        ]
        acciones = "\n".join([
            "Ordenadas por prioridad (primero los productos que no aparecen en su estante):",
            r"\begin{enumerate}", *items, r"\end{enumerate}",
            r"\vspace{4pt}Si no hay inventario en bodega para alguno de estos productos, escalar al área comercial "
            r"para gestionar el pedido con el proveedor.",
        ])
        conclusion = (
            f"La góndola está {_verdict(availability, meta, critico)}. Reponer {'el frente listado' if faltan == 1 else f'los {faltan} frentes listados'} en la sección de "
            f"recomendaciones devolvería la disponibilidad al 100\\,\\% del planograma. Se sugiere repetir la "
            f"auditoría después de la reposición para confirmar."
        )
    else:
        resumen = (
            f"Se analizaron {n} foto{'s' if n != 1 else ''} de la góndola de {_tex(planogram.get('categoria', ''))} "
            f"en {_tex(planogram.get('tienda', ''))}. Los {esperados} frentes del planograma tienen producto: "
            f"la disponibilidad en góndola es del \\textbf{{100\\,\\%}}."
        )
        acciones = "Mantener la frecuencia de reposición actual. No se requieren acciones inmediatas."
        conclusion = "La góndola cumple con el planograma en todas las fotos analizadas."

    filas_resumen = "\n".join(
        rf"    {i}.\ \texttt{{{_tex(s.name)}}} & {s.expected} & {s.missing} & {s.availability * 100:.1f}\,\% \\"
        for i, s in enumerate(shots, start=1)
    )
    values = {
        "metodologia": metodologia,  # primero: puede traer otros <<campos>>
        "meta": _pct(meta),
        "critico": _pct(critico),
        "categoria": _tex(planogram.get("categoria", "")),
        "tienda": _tex(planogram.get("tienda", "")),
        "gondola": _tex(planogram.get("gondola_id", "")),
        "vigente": _tex(planogram.get("vigente_desde", "")),
        "autor": _tex(autor),
        "fecha": f"{fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}",
        "n_fotos_texto": f"{n} foto{'s' if n != 1 else ''}",
        "esperados": str(esperados),
        "faltan": str(faltan),
        "faltan_texto": _n(faltan, "frente vacío", "frentes vacíos"),
        "disponibilidad": f"{availability * 100:.1f}",
        "resumen": resumen,
        "filas_resumen": filas_resumen,
        "detalle_fotos": "\n\n".join(_shot_section(i, s, name) for i, s in enumerate(shots, start=1)),
        "anexo": "\n\n".join(_shot_figure(i, s) for i, s in enumerate(shots, start=1)),
        "acciones": acciones,
        "color_ok": ",".join(map(str, COLORS[OK])),
        "color_vacio": ",".join(map(str, COLORS[MISSING])),
        "conclusion": conclusion,
    }
    tex = TEMPLATE
    for key, value in values.items():
        tex = tex.replace(f"<<{key}>>", value)
    return tex


def stockout_report(
    shots: list[Shot], planogram: dict, out_pdf: Path, autor: str = "Área de Trade Marketing", fecha: date | None = None,
    meta: float = 0.95, critico: float = 0.85, metodologia: str = METODOLOGIA,
) -> Path:
    """Compila el reporte y lo guarda en `out_pdf` (también deja el .tex al lado, para abrirlo en Overleaf)."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tex = render_tex(shots, planogram, autor=autor, fecha=fecha, meta=meta, critico=critico, metodologia=metodologia)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, shot in enumerate(shots, start=1):
            draw_stockouts(shot.image, shot.facings).save(tmp / f"foto{i}.png")
        (tmp / "reporte.tex").write_text(tex)
        for _ in range(2):  # dos pasadas para que resuelvan las referencias a figuras y tablas
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "reporte.tex"],
                cwd=tmp, capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError("pdflatex falló:\n" + result.stdout[-3000:])
        shutil.move(tmp / "reporte.pdf", out_pdf)  # /tmp puede estar en otro disco: Path.replace falla
    out_pdf.with_suffix(".tex").write_text(tex)
    return out_pdf
