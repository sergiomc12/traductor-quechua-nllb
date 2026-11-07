"""Promedio de pesos entre principal_s0 y auto_it1 (mismo NLLB base) y evaluacion con el protocolo de auto_ciclo.
  python -X utf8 src/mezclar_pesos.py --alfas 25 50 75
alfa = % de auto_it1 en la mezcla. Salida: models/mezcla_<alfa>/mejor, results/auto/eval_mezcla_<alfa>.json
Solo se decide con valid + devcheck; devtest y test se registran."""
import argparse, shutil, sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import auto_ciclo as ac  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--alfas", type=int, nargs="+", default=[25, 50, 75]); a = ap.parse_args()
    a_dir, b_dir = RAIZ / "models/principal_s0/mejor", RAIZ / "models/auto_it1/mejor"
    A = AutoModelForSeq2SeqLM.from_pretrained(a_dir, torch_dtype=torch.float32)
    sb = AutoModelForSeq2SeqLM.from_pretrained(b_dir, torch_dtype=torch.float32).state_dict()
    sa = {k: v.clone() for k, v in A.state_dict().items()}
    for al in a.alfas:
        w = al / 100
        A.load_state_dict({k: (sa[k] * (1 - w) + sb[k] * w) if sa[k].is_floating_point() else sa[k] for k in sa})
        out = RAIZ / f"models/mezcla_{al}/mejor"; out.mkdir(parents=True, exist_ok=True)
        A.save_pretrained(out); AutoTokenizer.from_pretrained(b_dir).save_pretrained(out)
        r = ac.evaluar_modelo(str(out), f"mezcla_{al}", False)
        print(al, "seleccion", r["_seleccion"], "valid", r["_valid"],
              "devtest", r["devtest_es->qu"]["chrF++"], r["devtest_qu->es"]["chrF++"],
              "test", r["test_es->qu"]["chrF++"], r["test_qu->es"]["chrF++"], flush=True)
main()
