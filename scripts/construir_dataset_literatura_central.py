"""Construye pares oracion a oracion quechua<->castellano desde los dos CSV de literatura Quechua Central.

Entradas (mal formadas para Excel: cp1252 y delimitadores distintos):
  data/dataset_literatura_quechua_central.csv    tabuladores, 5 columnas
  data/dataset_literatura_quechua_centralv2.csv  punto y coma, 5 columnas
Salidas:
  data/auto/extra_literatura_central.tsv   (es, qu) entrenamiento; formato de --extra-archivo de finetune.py
  data/auto/central_test.tsv               (es, qu) textos reservados para evaluar Quechua Central
  data/literatura_central_pares.csv        todos los pares con metadatos (UTF-8) para revisar a mano

Alineacion: si un texto tiene igual numero de oraciones en ambos idiomas se alinea 1 a 1 ("exacta").
Si no, se intenta un alineador por longitud ("dp") y esos pares NO entran a entrenamiento salvo
--incluir-dp. La division train/test es por texto (hash del titulo), nunca por oracion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FUENTES = [("v1", DATA / "dataset_literatura_quechua_central.csv", "\t", 0, 1),
           ("v2", DATA / "dataset_literatura_quechua_centralv2.csv", ";", 1, 2)]
FRACCION_TEST = 5  # 1 de cada 5 textos (por hash) va a test
RATIO_CARACTERES = (0.5, 2.0)  # largo quechua / largo castellano
MIN_PALABRAS = 2
_FIN_ORACION = re.compile(r"(?<=[.!?…])\s+(?=[«\"“¿¡A-ZÑÁÉÍÓÚ0-9])")
_PUNTOS_FINALES = re.compile(r"\s*(\.\.\.|…)+\s*$")


def limpiar(t: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", t)).strip()


def oraciones(texto: str) -> list[str]:
    return [limpiar(s) for s in _FIN_ORACION.split(texto.strip()) if limpiar(s)]


def alinear_dp(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Alineador por longitud (tipo Gale-Church simplificado): cuentas 1:1, 1:2 y 2:1."""
    n, m = len(a), len(b)
    la, lb = [len(x) for x in a], [len(x) for x in b]
    inf = float("inf")
    costo = [[inf] * (m + 1) for _ in range(n + 1)]
    atras = [[None] * (m + 1) for _ in range(n + 1)]
    costo[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if costo[i][j] == inf:
                continue
            for di, dj in ((1, 1), (1, 2), (2, 1)):
                if i + di > n or j + dj > m:
                    continue
                x, y = sum(la[i:i + di]), sum(lb[j:j + dj])
                c = costo[i][j] + abs(x - 1.07 * y) / max(x, y, 1) + (0.15 if (di, dj) != (1, 1) else 0)
                if c < costo[i + di][j + dj]:
                    costo[i + di][j + dj], atras[i + di][j + dj] = c, (i, j)
    if costo[n][m] == inf:
        return []
    pares, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj = atras[i][j]
        pares.append((" ".join(a[pi:i]), " ".join(b[pj:j])))
        i, j = pi, pj
    return pares[::-1]


def es_test(titulo: str) -> bool:
    return int(hashlib.sha1(titulo.encode("utf-8")).hexdigest(), 16) % FRACCION_TEST == 0


def leer_fuente(ruta: Path, delim: str):
    texto = ruta.read_bytes().decode("cp1252")
    filas = list(csv.reader(io.StringIO(texto, newline=""), delimiter=delim))
    return filas[1:]


def pares_de_texto(qu: str, es: str):
    a, b = oraciones(qu), oraciones(es)
    if len(a) == len(b):
        return "exacta", list(zip(a, b))
    return "dp", alinear_dp(a, b)


def valido(qu: str, es: str) -> bool:
    if len(qu.split()) < MIN_PALABRAS or len(es.split()) < MIN_PALABRAS:
        return False
    r = len(qu) / max(len(es), 1)
    return RATIO_CARACTERES[0] <= r <= RATIO_CARACTERES[1]


def construir():
    filas, pid = [], 0
    for fuente, ruta, delim, i_tit, i_cat in FUENTES:
        for r in leer_fuente(ruta, delim):
            titulo, categoria, qu, es = limpiar(r[i_tit]), limpiar(r[i_cat]), r[3], r[4]
            modo, pares = pares_de_texto(qu, es)
            for a, b in pares:
                a, b = _PUNTOS_FINALES.sub("", a), _PUNTOS_FINALES.sub("", b)
                pid += 1
                ok = valido(a, b)
                filas.append({"id": pid, "fuente": fuente, "titulo": titulo, "categoria": categoria,
                              "split": "test" if es_test(titulo) else "train", "alineacion": modo,
                              "valido": int(ok), "es": b, "qu": a})
    return filas


def escribir_tsv(ruta: Path, filas) -> int:
    vistos, n = set(), 0
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["es", "qu"])
        for f in filas:
            clave = (f["es"].lower(), f["qu"].lower())
            if clave in vistos:
                continue
            vistos.add(clave)
            w.writerow([f["es"], f["qu"]])
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--incluir-dp", action="store_true", help="incluye pares alineados por longitud en entrenamiento")
    a = p.parse_args()
    filas = construir()
    usables = [f for f in filas if f["valido"] and (f["alineacion"] == "exacta" or a.incluir_dp)]
    n_tr = escribir_tsv(DATA / "auto" / "extra_literatura_central.tsv", [f for f in usables if f["split"] == "train"])
    n_te = escribir_tsv(DATA / "auto" / "central_test.tsv", [f for f in usables if f["split"] == "test"])
    with (DATA / "literatura_central_pares.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"pares totales {len(filas)} | validos {sum(f['valido'] for f in filas)} | "
          f"exactos {sum(f['alineacion'] == 'exacta' for f in filas)} | dp {sum(f['alineacion'] == 'dp' for f in filas)}")
    print(f"train {n_tr} | test {n_te}")


if __name__ == "__main__":
    main()
