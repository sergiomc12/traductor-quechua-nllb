"""Linea base ZERO-SHOT: NLLB-200-distilled-600M sin ajustar, sobre el MISMO test sin texto religioso
y con el mismo codigo de evaluacion que el modelo ajustado (src/finetune.py).

  python -X utf8 src/baseline_nllb.py
Salida: results/baseline_zeroshot/{test.json,test_predicciones.tsv}
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import finetune as F  # noqa: E402


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.3)
    test = F.leer("test")
    tok = AutoTokenizer.from_pretrained(F.MODELO)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(F.MODELO).to("cuda").eval()
    salida = F.RAIZ / "results" / "baseline_zeroshot"
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = F.evaluar_test(tok, modelo, test, F.elegir_precision("auto"), salida)
    (salida / "test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
