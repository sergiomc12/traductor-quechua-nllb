"""Exporta las listas para corregir a mano el dataset quechua-espanol normalizado.

Genera en data/ (sin sobrescribir los que ya existan, para no pisar correcciones; usa --forzar):
  revision_adjetivo_sustantivo.csv   600 pares. Rellena `decision` con si/no; al volver a correr
                                      normalizar_dataset_quechua.py se aplican (no/si).
  revision_oraciones_dudosas.csv     Frases y oraciones que mis reglas marcan como dudosas.
  revision_palabras_en_contexto.csv  Como el dataset ORIGINAL anteponia Pasaq, Maypichus, etc.
                                      a frases, para que marques donde tienen sentido y donde no.
"""
import argparse
import csv
import random
from pathlib import Path

import normalizar_dataset_quechua as n

DATA = n.ROOT / "data"
# palabra: (glosa del usuario, glosa que le puso el dataset original)
PREFIJOS = {
    "Pasaq": ("Pasaré", "Ciertamente bastante"),
    "Maypichus": ("¿Dónde está?", "En donde"),
    "Allin puni": ("Muy bueno", "Definitivamente muy"),
    "Nisqanman hina": ("Según lo que dijo", "Según lo dicho"),
    "Cheqapta": ("Verdad", "Verdaderamente"),
    "Ancha": ("Muy", "Muy"),
}
CATEGORIAS_MUESTRA = ["Frase nominal", "Caso gramatical", "Oración presente", "Oración pasado", "Oración futuro"]
POR_CATEGORIA = 3


def _escribir(ruta: Path, campos, filas, forzar: bool) -> str:
    if ruta.exists() and not forzar:
        return f"{ruta.name}: ya existe, no se sobrescribe (usa --forzar)"
    with ruta.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    return f"{ruta.name}: {len(filas)} filas"


def pares(forzar: bool) -> str:
    filas = []
    for adj in n.ADJETIVOS:
        for sust in n.SUSTANTIVOS:
            q, e = n.frase_nominal(adj, sust, "base")
            if sust in n.ADJ_BLOQUEO.get(adj, set()):
                nivel, motivo = "bloqueado", "seco/mojado no se aplica a liquidos, astros ni conceptos"
            else:
                nivel, motivo = n.sugerencia_par(adj, sust)
            filas.append({"adjetivo": adj, "sustantivo": sust, "ejemplo_quechua": q, "ejemplo_espanol": e,
                          "sugerencia": nivel, "motivo": motivo, "decision": ""})
    orden = {"sin_sentido": 0, "dudoso": 1, "bloqueado": 2, "ok": 3}
    filas.sort(key=lambda f: orden[f["sugerencia"]])
    campos = ["adjetivo", "sustantivo", "ejemplo_quechua", "ejemplo_espanol", "sugerencia", "motivo", "decision"]
    return _escribir(DATA / "revision_adjetivo_sustantivo.csv", campos, filas, forzar)


def oraciones_dudosas(forzar: bool) -> str:
    filas = []
    for f in n.generar():
        es_nominal_simple = f["categoria"] == "Frase nominal" and not f["quechua"].startswith("Ancha ")
        if f["revisar"] and f["dialecto"] == "base" and (es_nominal_simple or f["categoria"].startswith("Oración")):
            filas.append({"categoria": f["categoria"], "quechua": f["quechua"], "espanol": f["espanol"],
                          "motivo": f["revisar"], "correccion_quechua": "", "correccion_espanol": ""})
    campos = ["categoria", "quechua", "espanol", "motivo", "correccion_quechua", "correccion_espanol"]
    return _escribir(DATA / "revision_oraciones_dudosas.csv", campos, filas, forzar)


def _minuscula(texto: str) -> str:
    return texto[0].lower() + texto[1:]


def palabras_en_contexto(forzar: bool) -> str:
    rng = random.Random(7)
    base = [f for f in n.generar() if f["dialecto"] == "base" and not f["revisar"]
            and not f["quechua"].startswith(("Ancha ", "Cheqapta ", "Nisqanman ", "Kunan ", "Qayna ", "Paqarin "))]
    filas = []
    for palabra, (glosa, glosa_original) in PREFIJOS.items():
        for cat in CATEGORIAS_MUESTRA:
            for f in rng.sample([b for b in base if b["categoria"] == cat], POR_CATEGORIA):
                filas.append({
                    "palabra": palabra, "glosa_del_usuario": glosa, "glosa_en_dataset_original": glosa_original,
                    "categoria": cat, "quechua_como_estaba": f"{palabra} {f['quechua']}",
                    "espanol_con_tu_glosa": f"{glosa} {_minuscula(f['espanol'])}",
                    "tiene_sentido": "", "correccion_quechua": "", "correccion_espanol": ""})
    campos = ["palabra", "glosa_del_usuario", "glosa_en_dataset_original", "categoria", "quechua_como_estaba",
              "espanol_con_tu_glosa", "tiene_sentido", "correccion_quechua", "correccion_espanol"]
    return _escribir(DATA / "revision_palabras_en_contexto.csv", campos, filas, forzar)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forzar", action="store_true", help="sobrescribe archivos existentes")
    a = p.parse_args()
    for exportar in (pares, oraciones_dudosas, palabras_en_contexto):
        print(exportar(a.forzar))


if __name__ == "__main__":
    main()
