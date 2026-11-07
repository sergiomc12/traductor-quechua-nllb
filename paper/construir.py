"""Genera paper/Paper.tex + Paper.pdf (IEEEtran, como Plantilla_Paper_IEEE) y paper/Paper.docx (dos columnas, Times 10 pt).

  python -X utf8 paper/construir.py
Requiere: pdflatex (MiKTeX) y python-docx.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import contenido as C  # noqa: E402
import resultados as R  # noqa: E402

AQUI = Path(__file__).parent
ROMANOS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def bloques() -> list[tuple]:
    return C.secciones_previas() + R.bloques() + R.contribuciones()


# ---------------------------------------------------------------- LaTeX
def esc(t: str) -> str:
    # Con babel-spanish, la comilla recta " es un atajo (ª, ñ...) y rompe el texto: se convierte a comillas de LaTeX.
    t = re.sub(r'"([^"]*)"', lambda m: "“" + m.group(1) + "”", t)
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")):
        t = t.replace(a, b)
    t = re.sub(r"(\d)·10⁻⁵", r"\1\\cdot 10^{-5}", t)
    t = t.replace("3\\cdot 10^{-5}", "$3\\cdot 10^{-5}$")
    return t.replace("≥", r"$\geq$").replace("±", r"$\pm$").replace("→", r"$\rightarrow$")


def cita(t: str) -> str:
    return re.sub(r"\[(\d+(?:\]\s*,\s*\[\d+)*)\]", lambda m: "\\cite{" + ",".join(f"r{n}" for n in re.findall(r"\d+", m.group(0))) + "}", t)


def tex_bloque(b: tuple) -> str:
    k = b[0]
    if k == "sec":
        return f"\\section{{{esc(b[1])}}}\n"
    if k == "sub":
        return f"\\subsection{{{esc(b[1])}}}\n"
    if k == "p":
        return cita(esc(b[1])) + "\n\n"
    if k == "bullets":
        return "\\begin{itemize}\n" + "".join(f"\\item {cita(esc(x))}\n" for x in b[1]) + "\\end{itemize}\n"
    if k == "eq":
        return f"\\begin{{equation}}\n{b[1]}\n\\end{{equation}}\n"
    if k == "table":
        _, etq, tit, cab, filas = b
        col = "l" + "c" * (len(cab) - 1)
        cuerpo = "\n".join(" & ".join(esc(str(c)) for c in f) + r" \\" for f in filas)
        tabla = (f"\\begin{{tabular}}{{{col}}}\n\\toprule\n{' & '.join(esc(c) for c in cab)} \\\\\n\\midrule\n{cuerpo}\n\\bottomrule\n\\end{{tabular}}")
        ancha = len(cab) >= 7  # 7 o más columnas: ocupa las dos columnas de la página
        if len(cab) > 3:  # tablas anchas: se escalan al ancho disponible para que no se superpongan
            tabla = f"\\adjustbox{{max width={'\\textwidth' if ancha else '\\columnwidth'}}}{{{tabla}}}"
        env = "table*" if ancha else "table"
        return f"\\begin{{{env}}}[!t]\n\\caption{{{esc(tit)}}}\n\\label{{{etq}}}\n\\centering\\small\n{tabla}\n\\end{{{env}}}\n"
    if k == "fig":
        _, ruta, tit, etq = b
        return f"\\begin{{figure}}[!t]\n\\centering\n\\includegraphics[width=\\columnwidth]{{{Path(ruta).name}}}\n\\caption{{{esc(tit)}}}\n\\label{{{etq}}}\n\\end{{figure}}\n"
    if k == "pend":
        return f"\\textbf{{[PENDIENTE: {esc(b[1])}]}}\n\n"
    if k == "resumen":
        return ""
    if k == "contrib":
        filas = "\n".join(f"{esc(n)} & {esc(t)} \\\\" for n, t in b[1])
        return ("\\section*{Contribuciones}\n\\begin{tabular}{@{}p{0.32\\columnwidth}p{0.6\\columnwidth}@{}}\n\\toprule\nIntegrante & Tareas en las que participó \\\\\n\\midrule\n"
                f"{filas}\n\\bottomrule\n\\end{{tabular}}\n\n")
    if k == "disp":
        return f"\\section*{{Disponibilidad de código y uso de herramientas de IA}}\n{esc(b[1])}\n\n"
    raise ValueError(k)


def construir_tex(bl: list[tuple]) -> str:
    resumen = next((b[1] for b in bl if b[0] == "resumen"), None)
    autores = "\n\\and\n".join(
        f"\\IEEEauthorblockN{{{esc(n)}}}\n\\IEEEauthorblockA{{{' \\\\ '.join(esc(x) for x in C.AFILIACION[:2])}\\\\{esc(C.AFILIACION[2])}\\\\{c}}}" for n, c in C.AUTORES
    )
    cuerpo = "".join(tex_bloque(b) for b in bl)
    refs = "\n".join(f"\\bibitem{{r{i}}} {esc(r)}" for i, r in enumerate(C.REFERENCIAS, 1))
    pre = r"""\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-tabla]{babel}
\usepackage{amsmath,booktabs,graphicx,cite,adjustbox}
\usepackage{microtype}
\hyphenation{aglu-ti-nan-te}
\renewcommand{\IEEEkeywordsname}{Palabras clave}
\begin{document}
\title{""" + esc(C.TITULO) + r"""}
\author{
""" + autores + r"""
}
\maketitle
\begin{abstract}
""" + (cita(esc(resumen)) if resumen else r"\textbf{[PENDIENTE: resumen con los resultados finales]}") + r"""
\end{abstract}
\begin{IEEEkeywords}
""" + esc(C.PALABRAS_CLAVE) + r"""
\end{IEEEkeywords}
"""
    return pre + cuerpo + "\\begin{thebibliography}{99}\n" + refs + "\n\\end{thebibliography}\n\\end{document}\n"


def compilar_pdf() -> None:
    if not shutil.which("pdflatex"):
        raise SystemExit("pdflatex no encontrado")
    for fig in (AQUI.parent / "results").glob("fig_*.png"):
        shutil.copy(fig, AQUI / fig.name)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "Paper.tex"], cwd=AQUI, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex fallo")


# ---------------------------------------------------------------- Word
def construir_docx(bl: list[tuple], destino: Path) -> None:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.59), Cm(27.94)  # carta
    s.left_margin = s.right_margin = Cm(1.78)
    s.top_margin, s.bottom_margin = Cm(1.9), Cm(2.3)
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(0)

    def par(texto="", just=WD_ALIGN_PARAGRAPH.JUSTIFY, tam=None, negrita=False, cursiva=False, versal=False, antes=0, despues=4, sangria=True):
        p = d.add_paragraph()
        p.alignment = just
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(antes), Pt(despues)
        if sangria and just == WD_ALIGN_PARAGRAPH.JUSTIFY:
            p.paragraph_format.first_line_indent = Cm(0.5)
        r = p.add_run(texto)
        r.bold, r.italic, r.font.small_caps = negrita, cursiva, versal
        if tam:
            r.font.size = Pt(tam)
        return p

    def columnas(seccion, n, espacio_cm=0.63):
        cols = seccion._sectPr.xpath("./w:cols")
        e = cols[0] if cols else OxmlElement("w:cols")
        e.set(qn("w:num"), str(n))
        e.set(qn("w:space"), str(int(espacio_cm * 567)))
        if not cols:
            seccion._sectPr.append(e)

    # Cabecera a una columna
    par(C.TITULO, WD_ALIGN_PARAGRAPH.CENTER, tam=22, despues=10)
    tabla = d.add_table(rows=2, cols=2)
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (n, c) in enumerate(C.AUTORES):
        cel = tabla.cell(i // 2, i % 2)
        cel.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cel.paragraphs[0].add_run(n).font.size = Pt(11)
        for linea, it in ((C.AFILIACION[0], True), (" – ".join(C.AFILIACION[1:]), False), (c, False)):
            p = cel.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(linea)
            r.italic, r.font.size = it, Pt(9)
        cel.add_paragraph().paragraph_format.space_after = Pt(4)

    cuerpo = d.add_section(WD_SECTION.CONTINUOUS)
    columnas(cuerpo, 2)

    resumen = next((b[1] for b in bl if b[0] == "resumen"), "[PENDIENTE: resumen con los resultados finales]")
    p = par("", antes=4, despues=2)
    r = p.add_run("Resumen—"); r.bold = r.italic = True; r.font.size = Pt(9)
    r = p.add_run(resumen); r.bold = True; r.font.size = Pt(9)
    p = par("", despues=6)
    r = p.add_run("Palabras clave—"); r.bold = r.italic = True; r.font.size = Pt(9)
    r = p.add_run(C.PALABRAS_CLAVE); r.bold = True; r.font.size = Pt(9)

    n_sec, n_sub, n_tab, n_fig, n_eq = 0, 0, 0, 0, 0
    for b in bl:
        k = b[0]
        if k == "sec":
            n_sec += 1
            n_sub = 0
            par(f"{ROMANOS[n_sec - 1]}. {b[1]}", WD_ALIGN_PARAGRAPH.CENTER, versal=True, antes=8, despues=3)
        elif k == "sub":
            n_sub += 1
            par(f"{chr(64 + n_sub)}. {b[1]}", WD_ALIGN_PARAGRAPH.LEFT, cursiva=True, antes=4, despues=2)
        elif k == "p":
            par(b[1])
        elif k == "bullets":
            for x in b[1]:
                p = par("", sangria=False, despues=2)
                p.paragraph_format.left_indent = Cm(0.6)
                p.paragraph_format.first_line_indent = Cm(-0.4)
                p.add_run("▪ " + x)
        elif k == "eq":
            n_eq += 1
            p = par(f"{b[2]}    ({n_eq})", WD_ALIGN_PARAGRAPH.CENTER, cursiva=True, antes=2, despues=4)
        elif k == "table":
            n_tab += 1
            par(f"Tabla {ROMANOS[n_tab - 1]}", WD_ALIGN_PARAGRAPH.CENTER, tam=8, antes=4, despues=0)
            par(b[2].upper(), WD_ALIGN_PARAGRAPH.CENTER, tam=8, despues=2)
            t = d.add_table(rows=1 + len(b[4]), cols=len(b[3]))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False
            primera = Cm(3.4) if len(b[3]) > 3 else Cm(2.8)
            resto = int((Cm(8.6) - primera) / max(len(b[3]) - 1, 1))
            for fila_t in t.rows:
                for j, cel in enumerate(fila_t.cells):
                    cel.width = primera if j == 0 else resto
            for j, h in enumerate(b[3]):
                t.cell(0, j).paragraphs[0].add_run(h).bold = True
            for i, f in enumerate(b[4], 1):
                for j, v in enumerate(f):
                    t.cell(i, j).paragraphs[0].add_run(str(v))
            for row in t.rows:
                for j, cel in enumerate(row.cells):
                    for pp in cel.paragraphs:
                        pp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                        for rr in pp.runs:
                            rr.font.size = Pt(8)
            par("", despues=2)
        elif k == "fig":
            n_fig += 1
            d.add_picture(str(b[1]), width=Cm(8.6))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            par(f"Figura {n_fig}. {b[2]}", WD_ALIGN_PARAGRAPH.CENTER, tam=8, despues=4)
        elif k == "pend":
            par(f"[PENDIENTE: {b[1]}]", negrita=True)
        elif k == "contrib":
            par("Contribuciones", WD_ALIGN_PARAGRAPH.CENTER, versal=True, antes=8, despues=3)
            t = d.add_table(rows=1 + len(b[1]), cols=2)
            t.cell(0, 0).paragraphs[0].add_run("Integrante").bold = True
            t.cell(0, 1).paragraphs[0].add_run("Tareas en las que participó").bold = True
            for i, (nom, tar) in enumerate(b[1], 1):
                t.cell(i, 0).paragraphs[0].add_run(nom)
                t.cell(i, 1).paragraphs[0].add_run(tar)
            for row in t.rows:
                for cel in row.cells:
                    for rr in cel.paragraphs[0].runs:
                        rr.font.size = Pt(9)
            par("", despues=2)
        elif k == "disp":
            par("Disponibilidad de código y uso de herramientas de IA", WD_ALIGN_PARAGRAPH.CENTER, versal=True, antes=8, despues=3)
            par(b[1])
        elif k == "resumen":
            pass
    par("Referencias", WD_ALIGN_PARAGRAPH.CENTER, versal=True, antes=8, despues=3)
    for i, r in enumerate(C.REFERENCIAS, 1):
        p = par("", sangria=False, despues=2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        run = p.add_run(f"[{i}]  {r}")
        run.font.size = Pt(8)
    d.save(destino)


if __name__ == "__main__":
    bl = bloques()
    (AQUI / "Paper.tex").write_text(construir_tex(bl), encoding="utf-8")
    compilar_pdf()
    construir_docx(bl, AQUI / "Paper.docx")
    print("Paper.tex, Paper.pdf y Paper.docx generados en", AQUI)
