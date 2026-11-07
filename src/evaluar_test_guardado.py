"""Evalua en el test general un modelo YA guardado (models/<nombre>/mejor), sin reentrenar.

Usa las mismas funciones de finetune.py (beams, precision, metricas), asi que los numeros son
comparables con results/<nombre>/test.json de los entrenamientos normales.

  python -X utf8 src/evaluar_test_guardado.py --nombre exp_norm15k_s0
Salida: results/<nombre>/test.json y test_predicciones.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune as ft  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--test-tsv", default="", help="TSV (es, qu) alternativo en vez del test general; ej. data/auto/central_test.tsv")
    p.add_argument("--etiqueta", default="", help="sufijo del archivo de salida cuando se usa --test-tsv")
    a = p.parse_args()
    ruta = ft.RAIZ / "models" / a.nombre / "mejor"
    if not (ruta / "model.safetensors").exists():
        raise SystemExit(f"No hay pesos en {ruta}")
    salida = ft.RAIZ / "results" / a.nombre
    salida.mkdir(parents=True, exist_ok=True)
    dtipo = ft.elegir_precision(a.precision)
    tok = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(ruta).to("cuda").eval()
    if a.test_tsv:
        import pandas as pd
        datos = pd.read_csv(ft.RAIZ / a.test_tsv, sep="	", keep_default_na=False)[["es", "qu"]]
        sub = salida / f"eval_{a.etiqueta or 'externo'}"
        sub.mkdir(exist_ok=True)
        resultado = ft.evaluar_test(tok, modelo, datos, dtipo, sub)
        (sub / "test.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    else:
        resultado = ft.evaluar_test(tok, modelo, ft.leer("test"), dtipo, salida)
        (salida / "test.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(json.dumps(resultado, indent=1), flush=True)
    del modelo
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
