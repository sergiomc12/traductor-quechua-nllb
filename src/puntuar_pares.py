"""Puntua cada par del entrenamiento con la perdida del NLLB SIN ajustar (juez independiente: no vio nuestros datos).

Un par desalineado (quechua y español de oraciones distintas) tiene perdida alta en ambas direcciones.
Escribe data/limpio/ayacucho_general/train_puntaje.tsv con es, qu, nll_es_qu, nll_qu_es, nll (media).
Luego `--frac` crea train_filtrado_<p>.tsv sin el p % peor:

  python -X utf8 src/puntuar_pares.py            # puntua
  python -X utf8 src/puntuar_pares.py --quitar 15  # escribe train_filtrado_15.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "limpio" / "ayacucho_general"
MODELO = "facebook/nllb-200-distilled-600M"
COD = {"es": "spa_Latn", "qu": "quy_Latn"}
BATCH = 24


@torch.inference_mode()
def nll(tok, m, src, tgt, o, d) -> list[float]:
    tok.src_lang, tok.tgt_lang = COD[o], COD[d]
    orden = sorted(range(len(src)), key=lambda i: len(src[i]) + len(tgt[i]))
    out = [0.0] * len(src)
    for i in range(0, len(orden), BATCH):
        idx = orden[i:i + BATCH]
        enc = tok([src[j] for j in idx], text_target=[tgt[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        etiquetas = enc["labels"]
        dec_in = shift_tokens_right(etiquetas, m.config.pad_token_id, m.config.decoder_start_token_id)
        lab = etiquetas.masked_fill(etiquetas == tok.pad_token_id, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):  # sin pasar labels: asi el modelo no calcula su propia perdida (6 GB)
            logits = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec_in).logits
        perd = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), lab, ignore_index=-100, reduction="none")
        medias = perd.sum(1) / (lab != -100).sum(1).clamp(min=1)
        del logits, perd
        for j, v in zip(idx, medias.tolist()):
            out[j] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quitar", type=float, default=0, help="porcentaje peor a descartar (crea train_filtrado_<p>.tsv)")
    a = ap.parse_args()
    arch = CARPETA / "train_puntaje.tsv"
    if not arch.exists():
        torch.cuda.set_per_process_memory_fraction(0.5)
        df = pd.read_csv(CARPETA / "train.tsv", sep="\t", keep_default_na=False)
        tok = AutoTokenizer.from_pretrained(MODELO)
        m = AutoModelForSeq2SeqLM.from_pretrained(MODELO).to("cuda").eval()
        df["nll_es_qu"] = nll(tok, m, df["es"].tolist(), df["qu"].tolist(), "es", "qu")
        df["nll_qu_es"] = nll(tok, m, df["qu"].tolist(), df["es"].tolist(), "qu", "es")
        df["nll"] = (df["nll_es_qu"] + df["nll_qu_es"]) / 2
        df.to_csv(arch, sep="\t", index=False)
        print("puntaje escrito:", arch)
    df = pd.read_csv(arch, sep="\t", keep_default_na=False)
    print(df["nll"].describe().round(2).to_string())
    if a.quitar:
        # Las entradas cortas (palabras sueltas) siempre tienen perdida alta aunque esten bien: nunca se descartan.
        largo = df["es"].str.split().str.len() >= 5
        corte = df.loc[largo, "nll"].quantile(1 - a.quitar / 100)
        descartar = largo & (df["nll"] > corte)
        keep = df[~descartar][["es", "qu"]]
        salida = CARPETA / f"train_filtrado_{int(a.quitar)}.tsv"
        keep.to_csv(salida, sep="\t", index=False)
        print(f"pares de 5+ palabras: {int(largo.sum())} | corte nll <= {corte:.2f} | se conservan {len(keep)} de {len(df)} ({int(descartar.sum())} descartados) -> {salida.name}")
        peor = df[descartar].sample(8, random_state=1)
        for _, r in peor.iterrows():
            print(f"  [nll {r.nll:.1f}] ES: {r.es[:90]}\n               QU: {r.qu[:90]}")


if __name__ == "__main__":
    main()
