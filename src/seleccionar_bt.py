"""Une los bloques de retrotraduccion y escribe los conjuntos filtrados por calidad de ida y vuelta.

  python -X utf8 src/seleccionar_bt.py --umbrales 30 40
Salida: data/bt/bt_rt30.tsv, data/bt/bt_rt40.tsv (columnas es, qu)
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SAL = RAIZ / "data" / "bt"


def repite(h: str) -> bool:
    w = h.lower().split()
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(re.search(r"\b(\S+)(?: \1){2,}", h.lower())) or any(tri.count(t) >= 2 for t in set(tri))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbrales", type=int, nargs="+", default=[30, 40])
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f, sep="\t", keep_default_na=False) for f in sorted(glob.glob(str(SAL / "bloque_*.tsv")))])
    print("pares generados:", len(d))
    d["largo_ok"] = d["es"].str.split().str.len().between(3, 60) & (d["es"].str.split().str.len() / d["qu"].str.split().str.len()).between(0.4, 3.0)
    d = d[d["largo_ok"] & ~d["es"].map(repite) & (d["es"].str.strip() != "")]
    print("tras control de longitud y repeticiones:", len(d))
    for u in a.umbrales:
        s = d[d["rt_chrf"] >= u][["es", "qu"]].drop_duplicates()
        s.to_csv(SAL / f"bt_rt{u}.tsv", sep="\t", index=False)
        print(f"ida y vuelta >= {u}: {len(s)} pares -> bt_rt{u}.tsv")


if __name__ == "__main__":
    main()
