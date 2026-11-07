"""Mide el apóstrofo DENTRO de palabras (eyectivas: p'unchay, ch'aki) en el quechua de los datos y en las salidas del modelo."""
import collections, re
import pandas as pd
RAIZ = __import__("pathlib").Path(__file__).resolve().parent.parent
APO = "'’‘´`ʼ"
INTERNO = re.compile(rf"(?<=\w)[{APO}](?=\w)")
test = pd.read_csv(RAIZ / "data/limpio/ayacucho_general/test.tsv", sep="\t", keep_default_na=False)
train = pd.read_csv(RAIZ / "data/limpio/ayacucho_general/train.tsv", sep="\t", keep_default_na=False)
pred = pd.read_csv(RAIZ / "results/exp_norm15k_s0/test_predicciones.tsv", sep="\t", keep_default_na=False)
def resumen(nombre, serie):
    frases = serie.tolist()
    con = [f for f in frases if INTERNO.search(f)]
    palabras = [w for f in frases for w in f.split()]
    con_w = [w for w in palabras if INTERNO.search(w)]
    formas = collections.Counter(m.group() for f in frases for m in INTERNO.finditer(f))
    print(f"{nombre}: {len(frases)} frases; {len(con)} con apóstrofo interno ({100*len(con)/len(frases):.1f} %); "
          f"{len(con_w)} de {len(palabras)} palabras ({100*len(con_w)/len(palabras):.1f} %); formas {dict(formas)}")
resumen("train qu", train["qu"]); resumen("test qu", test["qu"])
resumen("referencia es->qu", pred[pred.direccion == "es->qu"]["ref"])
resumen("salida es->qu (norm15k)", pred[pred.direccion == "es->qu"]["hip"])

import json
def cifras(serie):
    f = serie.tolist(); p = [w for x in f for w in x.split()]
    return {"frases": len(f), "frases_con": sum(bool(INTERNO.search(x)) for x in f),
            "palabras": len(p), "palabras_con": sum(bool(INTERNO.search(w)) for w in p),
            "formas": dict(collections.Counter(m.group() for x in f for m in INTERNO.finditer(x)))}
out = {"train_qu": cifras(train["qu"]), "test_qu": cifras(test["qu"]),
       "salida_es_qu": cifras(pred[pred.direccion == "es->qu"]["hip"]), "referencia_es_qu": cifras(pred[pred.direccion == "es->qu"]["ref"])}
(RAIZ / "results" / "apostrofo_datos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
