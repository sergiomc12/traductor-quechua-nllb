"""Evalua un modelo en FLORES-200 (es <-> quy_Latn) y permite probar variantes de decodificacion.

  python -X utf8 src/evaluar_flores.py --modelo models/principal_s0/mejor --nombre ajustado --split devtest
  python -X utf8 src/evaluar_flores.py --modelo facebook/nllb-200-distilled-600M --nombre zeroshot --split devtest

Datos: data/flores/flores200_dataset (https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz, CC BY-SA 4.0).
Salida: results/flores/<nombre>_<split>.json (+ predicciones .tsv)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
FLORES = RAIZ / "data" / "flores" / "flores200_dataset"
CODIGOS = {"es": "spa_Latn", "qu": "quy_Latn"}
DIRECCIONES = (("es", "qu"), ("qu", "es"))


def cargar(split: str) -> pd.DataFrame:
    leer = lambda c: (FLORES / split / f"{c}.{split}").read_text(encoding="utf-8").splitlines()
    return pd.DataFrame({"es": leer("spa_Latn"), "qu": leer("quy_Latn")})


@torch.inference_mode()
def traducir(tok, modelo, textos, origen, destino, beams=4, lote=16, **extra) -> list[str]:
    tok.src_lang = CODIGOS[origen]
    orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    salida = [""] * len(textos)
    for i in range(0, len(orden), lote):
        idx = orden[i:i + lote]
        enc = tok([textos[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(modelo.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = modelo.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODIGOS[destino]), num_beams=beams, max_new_tokens=256, **extra)
        for j, h in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            salida[j] = h
    return salida


def repite(h: str) -> bool:
    import re
    w = h.lower().split()
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(re.search(r"\b(\S+)(?: \1){3,}", h.lower())) or any(tri.count(t) >= 3 for t in set(tri))


def evaluar(tok, modelo, df: pd.DataFrame, **extra) -> tuple[dict, pd.DataFrame]:
    res, filas = {}, []
    for o, d in DIRECCIONES:
        hip = traducir(tok, modelo, df[o].tolist(), o, d, **extra)
        res[f"{o}->{d}"] = {
            "chrF++": round(CHRF(word_order=2).corpus_score(hip, [df[d].tolist()]).score, 1),
            "BLEU": round(BLEU().corpus_score(hip, [df[d].tolist()]).score, 1),
            "repeticiones": int(sum(repite(h) for h in hip)),
            "n": len(df),
        }
        filas.append(pd.DataFrame({"direccion": f"{o}->{d}", "src": df[o], "ref": df[d], "hip": hip}))
    return res, pd.concat(filas)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True)
    p.add_argument("--nombre", required=True)
    p.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--no-repeat", type=int, default=0, dest="no_repeat")
    p.add_argument("--rep-pen", type=float, default=1.0, dest="rep_pen")
    a = p.parse_args()
    extra = {}
    if a.no_repeat:
        extra["no_repeat_ngram_size"] = a.no_repeat
    if a.rep_pen != 1.0:
        extra["repetition_penalty"] = a.rep_pen
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(a.modelo).to("cuda").eval()
    t0 = time.time()
    res, pred = evaluar(tok, modelo, cargar(a.split), beams=a.beams, **extra)
    res["_config"] = {"modelo": a.modelo, "split": a.split, "beams": a.beams, **extra, "segundos": round(time.time() - t0)}
    carpeta = RAIZ / "results" / "flores"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{a.nombre}_{a.split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pred.to_csv(carpeta / f"{a.nombre}_{a.split}.tsv", sep="\t", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
