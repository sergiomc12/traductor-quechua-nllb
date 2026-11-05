"""Limpia y reparticiona los corpus es<->qu descargados (Ayacucho y Cusco).

Pasos por variedad (no se mezclan variedades):
  1. Normaliza (NFC, espacios) y descarta vacias, cortas, es==qu, muy largas
     y pares con razon de longitud absurda.
  2. Quita duplicados exactos.
  3. Une todos los splits originales (train/val/test; el `eval` de Cusco se
     descarta porque ya estaba 100% en train) y los REPARTICIONA por grupos
     para que ninguna frase espanola ni quechua cruce de un split a otro.
  4. Marca cada fila con `biblico` para poder medir dominio por separado.

Uso:  uv run --with pandas --with pyarrow python -X utf8 scripts/limpiar_datos.py
Salida: data/limpio/<variedad>/{train,valid,test}.tsv + reporte_limpieza.json
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "limpio"

MIN_CHARS = 3
MAX_CHARS = 400          # por encima son parrafos sin segmentar
RATIO_LONG_MAX = 4.0     # max(len)/min(len) entre es y qu
PROP_VALID = 0.05
PROP_TEST = 0.05
RE_BIBLICO = re.compile(r"\b(?:Dios|Jehov[aá]|Biblia|b[ií]blic\w*|Jes[uú]s|Cristo)\b", re.I)

FUENTES = {
    "ayacucho": {
        "dir": RAIZ / "data" / "hf_somosnlp_spanish_to_quechua" / "data",
        "cols": ("es", "qu"),
        "archivos": ["train-00000-of-00001", "validation-00000-of-00001", "test-00000-of-00001"],
    },
    "cusco": {
        "dir": RAIZ / "data" / "hf_pollitoconpapass_cuzco" / "data",
        "cols": ("spa", "quz"),
        # `eval` excluido a proposito: 100% de sus frases ya estan en train.
        "archivos": ["train-00000-of-00001", "validate-00000-of-00001", "test-00000-of-00001"],
    },
}


def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(texto))).strip()


def clave(texto: str) -> str:
    """Clave de comparacion: minusculas y solo letras/numeros."""
    return re.sub(r"[\W_]+", " ", texto.lower()).strip()


def cargar(variedad: str) -> pd.DataFrame:
    cfg = FUENTES[variedad]
    es_col, qu_col = cfg["cols"]
    partes = [pd.read_parquet(cfg["dir"] / f"{a}.parquet") for a in cfg["archivos"]]
    df = pd.concat(partes, ignore_index=True)
    return pd.DataFrame({"es": df[es_col].map(normalizar), "qu": df[qu_col].map(normalizar)})


def filtrar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica filtros en cadena; devuelve el df limpio y cuantas filas quito cada uno."""
    descartes: dict[str, int] = {}

    def paso(nombre: str, mascara_conservar: pd.Series, actual: pd.DataFrame) -> pd.DataFrame:
        descartes[nombre] = int((~mascara_conservar).sum())
        return actual[mascara_conservar]

    le, lq = df["es"].str.len(), df["qu"].str.len()
    df = paso("vacia_o_corta", (le >= MIN_CHARS) & (lq >= MIN_CHARS), df)
    df = paso("es_igual_qu", df["es"].map(clave) != df["qu"].map(clave), df)
    le, lq = df["es"].str.len(), df["qu"].str.len()
    df = paso("muy_larga", (le <= MAX_CHARS) & (lq <= MAX_CHARS), df)
    le, lq = df["es"].str.len(), df["qu"].str.len()
    ratio = pd.concat([le, lq], axis=1).max(axis=1) / pd.concat([le, lq], axis=1).min(axis=1)
    df = paso("ratio_longitud", ratio <= RATIO_LONG_MAX, df)
    df = paso("sin_letras", df["es"].str.contains(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]") & df["qu"].str.contains(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]"), df)
    antes = len(df)
    df = df.drop_duplicates(subset=["es", "qu"])
    descartes["duplicado_exacto"] = antes - len(df)
    return df.reset_index(drop=True), descartes


def grupos(df: pd.DataFrame) -> pd.Series:
    """Union-find: filas que comparten clave es O clave qu quedan en el mismo grupo."""
    padre = list(range(len(df)))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for col in ("es", "qu"):
        primero: dict[str, int] = {}
        for i, k in enumerate(df[col].map(clave)):
            if k in primero:
                padre[raiz(i)] = raiz(primero[k])
            else:
                primero[k] = i
    return pd.Series([raiz(i) for i in range(len(df))], index=df.index)


def asignar_split(id_grupo: int) -> str:
    h = int(hashlib.md5(str(id_grupo).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < PROP_TEST:
        return "test"
    if h < PROP_TEST + PROP_VALID:
        return "valid"
    return "train"


def procesar(variedad: str) -> dict:
    bruto = cargar(variedad)
    limpio, descartes = filtrar(bruto)
    limpio = limpio.assign(biblico=limpio["es"].str.contains(RE_BIBLICO) | limpio["qu"].str.contains(RE_BIBLICO))
    limpio = limpio.assign(split=grupos(limpio).map(asignar_split))

    destino = SALIDA / variedad
    destino.mkdir(parents=True, exist_ok=True)
    conteos: dict[str, int] = {}
    for nombre in ("train", "valid", "test"):
        parte = limpio[limpio["split"] == nombre][["es", "qu", "biblico"]]
        parte.to_csv(destino / f"{nombre}.tsv", sep="\t", index=False)
        conteos[nombre] = len(parte)

    # Verificacion de fuga: ninguna clave es/qu compartida entre splits.
    fuga = {}
    for col in ("es", "qu"):
        claves = {n: set(limpio[limpio["split"] == n][col].map(clave)) for n in conteos}
        fuga[col] = {
            "train&valid": len(claves["train"] & claves["valid"]),
            "train&test": len(claves["train"] & claves["test"]),
            "valid&test": len(claves["valid"] & claves["test"]),
        }
    return {
        "filas_brutas": len(bruto),
        "descartes": descartes,
        "filas_limpias": len(limpio),
        "splits": conteos,
        "pct_biblico": round(float(limpio["biblico"].mean() * 100), 1),
        "fuga_entre_splits": fuga,
    }


def main() -> None:
    reporte = {v: procesar(v) for v in FUENTES}
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "reporte_limpieza.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(reporte, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
