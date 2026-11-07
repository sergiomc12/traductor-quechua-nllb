"""Retrotraduccion (back-translation) con la Wikipedia en quechua.

1. Selecciona oraciones quechuas (Wikipedia qu, CC BY-SA 3.0) con vocabulario NUEVO pero plausible:
   sin rasgos del quechua central y con 10-45 % de palabras que nuestro entrenamiento no conoce.
2. Las traduce qu->es con el modelo ajustado (respaldo: se escribe un archivo por bloque y se retoma).
3. Control de calidad por ida y vuelta: traduce de nuevo es->qu y guarda el chrF++ frente al original.

  python -X utf8 src/preparar_bt.py --n 40000
Salida: data/bt/candidatas.tsv, data/bt/bloque_0000.tsv ... (qu, es, rt_chrf)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from evaluar_flores import traducir  # noqa: E402
from marcar_religioso import RE as RELIGIOSO  # noqa: E402

SALIDA = RAIZ / "data" / "bt"
TOK = lambda s: re.findall(r"[^\W\d_]+", s.lower())
NORM = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
# Rasgos del quechua central/otras variedades (locativo -chaw, ablativo -pita, 1.ª persona pl. -ntsik, etc.)
# Grafia de Cusco/Bolivia: ejectivas con apostrofe (q'illay) y aspiradas (qh, kh, ph, th, chh); no son del ayacuchano de nuestros datos (0.1 % y 0.7 %)
CUSCO = re.compile(r"[a-záéíóúñ][’'ʼ`´][a-záéíóúñ]|\w*(qh|kh|ph|th|chh)\w*", re.I)
CENTRAL = re.compile(r"\b\w+chaw\b|\b\w+pita\b|\b\w+ntsik\b|\bnoqa\b|\bnuqa\b|\bshu\w+|\b\w+shqa\b|\bkaq\b\s+\w+kuna\b.*chaw", re.I)


def candidatas(n: int, semilla: int = 0) -> pd.DataFrame:
    arch = SALIDA / "candidatas.tsv"
    if arch.exists():
        return pd.read_csv(arch, sep="\t", keep_default_na=False)
    SALIDA.mkdir(parents=True, exist_ok=True)
    wiki = "\n".join(gzip.open(f, "rt", encoding="utf-8", errors="ignore").read() for f in sorted(glob.glob(str(RAIZ / "data" / "candidatos" / "wiki" / "p*.gz"))))
    for extra in ("cc100-quechua.txt", "que_community_2017-sentences.txt"):  # Llamacha/monolingual-quechua-iic (Apache-2.0)
        wiki += "\n" + (RAIZ / "data" / "candidatos" / extra).read_text(encoding="utf-8", errors="ignore")
    oraciones = {s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", wiki)}
    train = pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho_general" / "train.tsv", sep="\t", keep_default_na=False)
    vocab = {w for s in train["qu"] for w in TOK(s)}
    excluir = set()
    for sp in ("dev", "devtest"):
        excluir |= {NORM(x) for x in (RAIZ / "data" / "flores" / "flores200_dataset" / sp / f"quy_Latn.{sp}").read_text(encoding="utf-8").splitlines()}
    for sp in ("valid", "test"):
        excluir |= set(map(NORM, pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho_general" / f"{sp}.tsv", sep="\t", keep_default_na=False)["qu"]))
    filas = []
    for s in oraciones:
        w = s.split()
        if not 6 <= len(w) <= 40 or CENTRAL.search(s) or CUSCO.search(s) or NORM(s) in excluir:
            continue
        # Mismo criterio del proyecto: sin texto religioso (el CC-100 trae versiculos biblicos); y sin ruido de formato
        if RELIGIOSO.search(s) or re.match(r"^[\d*\[(—\-•]", s) or "[" in s or "http" in s:
            continue
        letras = sum(c.isalpha() for c in s) / max(len(s), 1)
        t = TOK(s)
        if letras < 0.8 or not t:
            continue
        oov = sum(x not in vocab for x in t) / len(t)
        if 0.10 <= oov <= 0.45:
            filas.append((s, round(oov, 3)))
    df = pd.DataFrame(filas, columns=["qu", "oov"]).drop_duplicates("qu")
    # Fichas de Wikipedia con plantilla ("X sutiyuq runaqa (paqarisqa ...)", "Y nisqaqa ... huk llaqtam"): se limitan para que no dominen
    plantilla = df["qu"].str.contains(r"sutiyuq (?:runa|warmi)|nisqa(?:qa)? .*huk \w+(?:m|am)\b", regex=True)
    df = pd.concat([df[~plantilla], df[plantilla].sample(min(int(plantilla.sum()), 1500), random_state=semilla)])
    print(f"plantillas: {int(plantilla.sum())} -> se conservan {min(int(plantilla.sum()), 1500)}", flush=True)
    print(f"oraciones que cumplen el filtro: {len(df)} (de {len(oraciones)})", flush=True)
    df = df.sample(min(n, len(df)), random_state=semilla).reset_index(drop=True)
    df.to_csv(arch, sep="\t", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--bloque", type=int, default=2000)
    ap.add_argument("--modelo", default=str(RAIZ / "models" / "principal_s0" / "mejor"))
    ap.add_argument("--solo-candidatas", action="store_true")
    a = ap.parse_args()
    df = candidatas(a.n)
    print("candidatas:", len(df), flush=True)
    if a.solo_candidatas:
        return
    torch.cuda.set_per_process_memory_fraction(0.35)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    m = AutoModelForSeq2SeqLM.from_pretrained(a.modelo).to("cuda").eval()
    chrf = CHRF(word_order=2)
    for i in range(0, len(df), a.bloque):
        arch = SALIDA / f"bloque_{i // a.bloque:04d}.tsv"
        if arch.exists():  # retoma desde donde quedo
            continue
        qu = df["qu"].iloc[i:i + a.bloque].tolist()
        es = traducir(tok, m, qu, "qu", "es", beams=2, lote=48, no_repeat_ngram_size=3)
        vuelta = traducir(tok, m, es, "es", "qu", beams=2, lote=48, no_repeat_ngram_size=3)
        rt = [round(chrf.sentence_score(v, [q]).score, 1) for v, q in zip(vuelta, qu)]
        pd.DataFrame({"qu": qu, "es": es, "rt_chrf": rt}).to_csv(arch.with_suffix(".tmp"), sep="\t", index=False)
        arch.with_suffix(".tmp").replace(arch)
        print(f"bloque {i // a.bloque:04d} listo ({min(i + a.bloque, len(df))}/{len(df)})", flush=True)


if __name__ == "__main__":
    main()
