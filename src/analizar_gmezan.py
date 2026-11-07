"""Analiza gmezan/quechua-spanish-parallel: fuentes, contenido religioso, solapamiento y pares NUEVOS no religiosos."""
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from marcar_religioso import RE  # noqa: E402

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
C = RAIZ / "data" / "candidatos"
g = pd.concat([pq.read_table(C / "gmezan" / f"{s}.parquet", columns=["id", "text_quy", "text_spa", "source"]).to_pandas().assign(split=s) for s in ("train", "validation", "test")])
print("filas:", len(g))
print(g["source"].value_counts().to_string())
g = g[(g.text_quy.str.len() > 1) & (g.text_spa.str.len() > 1)].copy()
print("con quechua y espanol:", len(g))
g["relig"] = g.text_spa.str.contains(RE) | g.text_quy.str.contains(RE)
print("marcadas religiosas por nuestro filtro:", int(g.relig.sum()), f"({100 * g.relig.mean():.0f} %)")
print(g.groupby("source").relig.mean().round(2).to_string())

L = RAIZ / "data" / "limpio"
A = pd.concat([pd.read_csv(L / "ayacucho" / f"{s}.tsv", sep="\t", keep_default_na=False) for s in ("train", "valid", "test")])
Aes, Aqu = set(map(norm, A.es)), set(map(norm, A.qu))
te = pd.read_csv(L / "ayacucho_general" / "test.tsv", sep="\t", keep_default_na=False)
Tes, Tqu = set(map(norm, te.es)), set(map(norm, te.qu))
va = pd.read_csv(L / "ayacucho_general" / "valid.tsv", sep="\t", keep_default_na=False)
Ves, Vqu = set(map(norm, va.es)), set(map(norm, va.qu))
FL = set(norm(x) for sp in ("dev", "devtest") for q in ("quy_Latn",) for x in open(RAIZ / "data" / "flores" / "flores200_dataset" / sp / f"{q}.{sp}", encoding="utf-8").read().splitlines())
FLes = set(norm(x) for sp in ("dev", "devtest") for x in open(RAIZ / "data" / "flores" / "flores200_dataset" / sp / f"spa_Latn.{sp}", encoding="utf-8").read().splitlines())

g["es_n"], g["qu_n"] = g.text_spa.map(norm), g.text_quy.map(norm)
g["en_nuestro"] = g.es_n.isin(Aes) | g.qu_n.isin(Aqu)
g["en_test_valid"] = g.es_n.isin(Tes | Ves) | g.qu_n.isin(Tqu | Vqu)
g["en_flores"] = g.qu_n.isin(FL) | g.es_n.isin(FLes)
print("\nya presentes en nuestro conjunto completo:", int(g.en_nuestro.sum()), f"({100 * g.en_nuestro.mean():.0f} %) | en nuestra prueba/validacion:", int(g.en_test_valid.sum()), "| en FLORES:", int(g.en_flores.sum()))
nuevo = g[(~g.relig) & (~g.en_nuestro) & (~g.en_test_valid) & (~g.en_flores)]
print("NUEVOS no religiosos y sin solapamiento:", len(nuevo))
print(nuevo["source"].value_counts().to_string())
for _, r in nuevo.sample(min(8, len(nuevo)), random_state=1).iterrows():
    print("  ES:", r.text_spa[:100], "\n  QU:", r.text_quy[:100])
nuevo[["text_spa", "text_quy", "source"]].rename(columns={"text_spa": "es", "text_quy": "qu"}).to_csv(C / "gmezan_nuevos_no_religiosos.tsv", sep="\t", index=False)
