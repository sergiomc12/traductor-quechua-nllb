"""Deriva de data/limpio/ayacucho/ las particiones que usa el entrenamiento: solo texto NO religioso.

  data/limpio/ayacucho_general/{train,valid,test}.tsv   filas con religioso == 0, columnas es y qu
  data/limpio/ayacucho_general/train_aleatorio_15.tsv   (con --aleatorio) variante del experimento "sin 15 % al azar"

Antes se ejecutan src/limpiar_datos.py y src/marcar_religioso.py. Las tres particiones salen idénticas, fila por fila,
a las que se usaron en los experimentos del artículo (se verificó comparándolas).

El archivo train_aleatorio_15.tsv del experimento exp_aleatorio15_s0 se generó con código que no se conservó, por lo que
la semilla original se desconoce; esta opción produce un equivalente (mismo número de pares quitados, elegidos al azar entre los
de 5 o más palabras, como hace puntuar_pares.py con los de mayor pérdida) pero NO el mismo archivo.

  python -X utf8 src/preparar_particiones.py [--aleatorio]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = RAIZ / "data" / "limpio" / "ayacucho"
DESTINO = RAIZ / "data" / "limpio" / "ayacucho_general"
PALABRAS_MINIMAS = 5
SEMILLA_ALEATORIO = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aleatorio", action="store_true", help="genera también train_aleatorio_15.tsv (equivalente, no idéntico al usado)")
    args = ap.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)
    for particion in ("train", "valid", "test"):
        df = pd.read_csv(ORIGEN / f"{particion}.tsv", sep="\t", keep_default_na=False)
        general = df[df["religioso"] == 0][["es", "qu"]].reset_index(drop=True)
        general.to_csv(DESTINO / f"{particion}.tsv", sep="\t", index=False)
        print(f"{particion}: {len(general)} de {len(df)} pares")
    if args.aleatorio:
        train = pd.read_csv(DESTINO / "train.tsv", sep="\t", keep_default_na=False)
        referencia = pd.read_csv(DESTINO / "train_filtrado_15.tsv", sep="\t", keep_default_na=False) if (DESTINO / "train_filtrado_15.tsv").exists() else None
        n_quitar = len(train) - len(referencia) if referencia is not None else round(0.15 * (train["es"].str.split().str.len() >= PALABRAS_MINIMAS).sum())
        candidatos = train[train["es"].str.split().str.len() >= PALABRAS_MINIMAS]
        quitar = candidatos.sample(n=n_quitar, random_state=SEMILLA_ALEATORIO).index
        train.drop(quitar).to_csv(DESTINO / "train_aleatorio_15.tsv", sep="\t", index=False)
        print(f"train_aleatorio_15: se quitaron {n_quitar} pares al azar (semilla {SEMILLA_ALEATORIO})")


if __name__ == "__main__":
    main()
