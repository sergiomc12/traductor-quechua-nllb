"""Examen de concordancia verbal: persona x tiempo con verbos que NO estan en el dataset normalizado.

Mide si un modelo conjuga bien (acierto = la forma verbal esperada aparece en la salida):
  es->qu: "Tu trabajaras"  -> debe contener "llamkanki"
  qu->es: "qam llamkanki"  -> debe contener "trabajas"
Las formas esperadas salen de las reglas de scripts/normalizar_dataset_quechua.py (paradigma
Cusco/Chanka). Por eso mide si el modelo APRENDIO ESE PARADIGMA, no si es el Ayacucho estandar.
Para 1.a persona plural el español lleva la anotacion "(inclusivo)/(exclusivo)" del dataset.

  python -X utf8 src/examen_concordancia.py --modelo zeroshot=facebook/nllb-200-distilled-600M \
      --modelo principal_s0=models/principal_s0/mejor
Salida: results/concordancia/<nombre>.json y tabla en pantalla.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
import finetune as ft  # noqa: E402
import normalizar_dataset_quechua as n  # noqa: E402

# Verbos fuera del dataset normalizado: raiz quechua -> infinitivo espanol (todos regulares)
VERBOS_NUEVOS = {"llamka": "trabajar", "waqa": "llorar", "sama": "descansar", "kawsa": "vivir", "qillqa": "escribir"}
_PALABRA = re.compile(r"[\wñáéíóúü']+")


def palabras(texto: str) -> set[str]:
    return set(_PALABRA.findall(texto.lower()))


def construir_items() -> list[dict]:
    items = []
    for raiz, inf in VERBOS_NUEVOS.items():
        for tiempo in n.TIEMPOS:
            for persona in n.PERSONAS:
                qu_verbo = n.verbo_quechua(raiz, tiempo, persona)
                es_verbo = n.verbo_espanol(inf, tiempo, persona)
                items.append({
                    "tiempo": tiempo, "persona": persona, "verbo": raiz,
                    "es": f"{n.PRON_ES[persona]} {es_verbo}".capitalize(),
                    "qu": f"{n.PRON_QU[persona]} {qu_verbo}",
                    "esperado_qu": qu_verbo, "esperado_es": es_verbo,
                    # futuro de 2.a persona se escribe igual que el presente: ambiguo, solo se evalua es->qu
                    "evaluar_qu_es": not (tiempo == "futuro" and persona in ("2s", "2p")),
                })
    return items


def evaluar(tok, modelo, items, dtipo) -> dict:
    es_qu = ft.traducir(tok, modelo, [i["es"] for i in items], "es", "qu", dtipo)
    sel = [i for i in items if i["evaluar_qu_es"]]
    qu_es = dict(zip(map(id, sel), ft.traducir(tok, modelo, [i["qu"] for i in sel], "qu", "es", dtipo)))
    acc = defaultdict(lambda: [0, 0])
    detalle = []
    for i, salida in zip(items, es_qu):
        ok = i["esperado_qu"].lower() in palabras(salida)
        for clave in ("es->qu", f"es->qu:{i['tiempo']}", f"es->qu:{i['persona']}"):
            acc[clave][0] += ok; acc[clave][1] += 1
        detalle.append({"dir": "es->qu", "entrada": i["es"], "esperado": i["esperado_qu"], "salida": salida, "ok": ok})
        if i["evaluar_qu_es"]:
            salida2 = qu_es[id(i)]
            ok2 = i["esperado_es"].lower() in palabras(salida2)
            for clave in ("qu->es", f"qu->es:{i['tiempo']}", f"qu->es:{i['persona']}"):
                acc[clave][0] += ok2; acc[clave][1] += 1
            detalle.append({"dir": "qu->es", "entrada": i["qu"], "esperado": i["esperado_es"], "salida": salida2, "ok": ok2})
    return {"acierto": {k: round(100 * a / t, 1) for k, (a, t) in acc.items()},
            "n": {k: t for k, (a, t) in acc.items()}, "detalle": detalle}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", action="append", required=True, help="nombre=ruta_o_id (repetible)")
    a = p.parse_args()
    items = construir_items()
    dtipo = ft.elegir_precision("auto")
    out = RAIZ / "results" / "concordancia"
    out.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for spec in a.modelo:
        nombre, ruta = spec.split("=", 1)
        carga = str(RAIZ / ruta) if (RAIZ / ruta).exists() else ruta
        tok = AutoTokenizer.from_pretrained(carga)
        modelo = AutoModelForSeq2SeqLM.from_pretrained(carga).to("cuda").eval()
        r = evaluar(tok, modelo, items, dtipo)
        (out / f"{nombre}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        resumen[nombre] = r["acierto"]
        del modelo
        torch.cuda.empty_cache()
    cols = ["es->qu", "qu->es"] + [f"es->qu:{t}" for t in n.TIEMPOS] + [f"es->qu:{p}" for p in n.PERSONAS]
    print(f"{'modelo':18}" + "".join(f"{c.replace('es->qu','eq').replace('qu->es','qe'):>13}" for c in cols))
    for nombre, acc in resumen.items():
        print(f"{nombre:18}" + "".join(f"{acc.get(c, float('nan')):13.1f}" for c in cols))


if __name__ == "__main__":
    main()
