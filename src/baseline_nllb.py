"""Linea base ZERO-SHOT de NLLB-200 sobre el test limpio (sin ajuste fino).

Mide chrF++ y BLEU en es->quy y quy->es, separando frases biblicas y no biblicas.
Solo evalua, no entrena. Usar el python del venv con torch+transformers:
  D:/Backend/redesneuronales/.venv/Scripts/python.exe -X utf8 src/baseline_nllb.py
Requiere sacrebleu (instalar fuera del venv: ver AGENTS.md §7).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "facebook/nllb-200-distilled-600M"
N_MUESTRA = 400
BATCH = 32
CODIGOS = {"es": "spa_Latn", "qu": "quy_Latn"}


def traducir(tok, modelo, textos: list[str], origen: str, destino: str) -> list[str]:
    tok.src_lang = CODIGOS[origen]
    salida: list[str] = []
    for i in range(0, len(textos), BATCH):
        lote = tok(textos[i:i + BATCH], return_tensors="pt", padding=True, truncation=True, max_length=128).to(modelo.device)
        with torch.inference_mode():
            gen = modelo.generate(**lote, forced_bos_token_id=tok.convert_tokens_to_ids(CODIGOS[destino]), num_beams=4, max_new_tokens=128)
        salida.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return salida


def puntuar(hip: list[str], ref: list[str]) -> dict[str, float]:
    return {
        "chrF++": round(CHRF(word_order=2).corpus_score(hip, [ref]).score, 1),
        "BLEU": round(BLEU().corpus_score(hip, [ref]).score, 1),
        "n": len(ref),
    }


def main() -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(MODELO, torch_dtype=torch.float16 if dispositivo == "cuda" else torch.float32).to(dispositivo).eval()
    resultados: dict = {"modelo": MODELO, "dispositivo": dispositivo, "variedad_nllb": "quy_Latn (Ayacucho)"}

    test = pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho" / "test.tsv", sep="\t")
    for etiqueta, sub in (("no_biblico", test[~test["biblico"]]), ("biblico", test[test["biblico"]])):
        muestra = sub.sample(min(N_MUESTRA, len(sub)), random_state=0)
        for origen, destino in (("es", "qu"), ("qu", "es")):
            t0 = time.time()
            hip = traducir(tok, modelo, muestra[origen].tolist(), origen, destino)
            res = puntuar(hip, muestra[destino].tolist())
            res["seg_por_frase"] = round((time.time() - t0) / len(muestra), 3)
            res["ejemplo"] = {"src": muestra[origen].iloc[0], "ref": muestra[destino].iloc[0], "hip": hip[0]}
            resultados[f"{etiqueta}_{origen}->{destino}"] = res
            print(etiqueta, origen, "->", destino, res["chrF++"], res["BLEU"], flush=True)

    salida = RAIZ / "data" / "limpio" / "baseline_nllb600m_zeroshot.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultados, indent=2, ensure_ascii=False), file=sys.stdout)


if __name__ == "__main__":
    main()
