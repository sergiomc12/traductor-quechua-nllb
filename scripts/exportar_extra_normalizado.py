"""Exporta del conjunto normalizado los pares (es, qu) que se añaden al entrenamiento.

Entrada:  data/dataset_quechua_espanol_normalizado.csv  (lo escribe scripts/normalizar_dataset_quechua.py)
Salidas:  data/auto/extra_norm15k.tsv          filas sin marca de revisión y sin prefijos dudosos
          data/auto/extra_norm2k.tsv           solo oraciones con verbo y léxico
          data/auto/extra_norm15k_central.tsv  15k + literatura quechua central repetida 3 veces

Reglas (decididas en la revisión del conjunto, ver el artículo, Sección III-F):
  - se descartan las filas con marca en la columna `revisar` (dudosas o sin sentido);
  - se descartan las frases con los prefijos Nisqanman, Cheqapta y Ancha, cuyo uso en contexto no se validó,
    pero se conservan las entradas sueltas "Nisqanman hina" y "Cheqapta", que sí tienen glosa confirmada;
  - se eliminan los pares repetidos conservando el orden.

  python -X utf8 scripts/exportar_extra_normalizado.py [--central]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "dataset_quechua_espanol_normalizado.csv"
SALIDA = RAIZ / "data" / "auto"
PREFIJOS_DUDOSOS = ("Nisqanman", "Cheqapta", "Ancha")
ENTRADAS_SUELTAS = {"Nisqanman hina", "Cheqapta"}
CATEGORIAS_2K = ("Oraci", "L")  # Oración presente/pasado/futuro y Léxico
REPETICIONES_CENTRAL = 3


def filtrar_15k(df: pd.DataFrame) -> pd.DataFrame:
    sin_marca = df[df["revisar"] == ""]
    dudosas = sin_marca["quechua"].str.startswith(PREFIJOS_DUDOSOS) & ~sin_marca["quechua"].isin(ENTRADAS_SUELTAS)
    return sin_marca[~dudosas]


def a_pares(df: pd.DataFrame) -> pd.DataFrame:
    pares = list(dict.fromkeys(zip(df["espanol"], df["quechua"])))
    return pd.DataFrame(pares, columns=["es", "qu"])


def escribir(df: pd.DataFrame, nombre: str) -> None:
    destino = SALIDA / f"{nombre}.tsv"
    df.to_csv(destino, sep="\t", index=False, lineterminator="\n")
    print(f"{destino.relative_to(RAIZ)}: {len(df)} pares")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--central", action="store_true", help="genera también extra_norm15k_central.tsv (necesita extra_literatura_central.tsv)")
    args = ap.parse_args()
    if not ENTRADA.exists():
        raise SystemExit(f"Falta {ENTRADA}: ejecuta antes scripts/normalizar_dataset_quechua.py")
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ENTRADA, encoding="utf-8-sig", keep_default_na=False)
    base = filtrar_15k(df)
    p15 = a_pares(base)
    escribir(p15, "extra_norm15k")
    escribir(a_pares(base[base["categoria"].str.startswith(CATEGORIAS_2K)]), "extra_norm2k")
    if args.central:
        central = pd.read_csv(SALIDA / "extra_literatura_central.tsv", sep="\t", keep_default_na=False)[["es", "qu"]]
        escribir(pd.concat([p15] + [central] * REPETICIONES_CENTRAL, ignore_index=True), "extra_norm15k_central")


if __name__ == "__main__":
    main()
