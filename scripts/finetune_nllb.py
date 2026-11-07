"""Fine-tuning de NLLB-200-distilled-600M para es<->quy (Ayacucho), bidireccional.

NO se ejecuta solo: lo lanza el usuario. Ejemplo:
  D:/Backend/redesneuronales/.venv/Scripts/python.exe -X utf8 scripts/finetune_nllb.py --epocas 3

Decisiones (ver AGENTS.md §6):
  - Una variedad (Ayacucho) porque NLLB solo tiene quy_Latn y los papers no
    mezclan variedades.
  - Un modelo para ambas direcciones: cada par genera dos ejemplos (es->qu y qu->es).
  - El contenido biblico se reduce (--frac-biblico) para no sobreajustar a ese texto.
  - Se elige el mejor punto por chrF++ en valid NO biblico (el dominio mas debil).
Solo usa PyTorch + transformers + sacrebleu (sin accelerate/Trainer).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "facebook/nllb-200-distilled-600M"
CODIGOS = {"es": "spa_Latn", "qu": "quy_Latn"}
DIRECCIONES = (("es", "qu"), ("qu", "es"))
MAX_LEN = 128
SEMILLA = 0


def parsear_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.05, help="fraccion de pasos de calentamiento")
    p.add_argument("--frac-biblico", type=float, default=0.3, help="fraccion de filas biblicas de train que se conserva")
    p.add_argument("--n-valid", type=int, default=500, help="frases de valid no biblico para medir chrF++ por epoca")
    p.add_argument("--max-pasos", type=int, default=0, help="0 = sin limite; util para una prueba de humo")
    p.add_argument("--salida", type=Path, default=RAIZ / "models" / "nllb600m_quy")
    return p.parse_args()


def cargar_train(frac_biblico: float) -> pd.DataFrame:
    df = pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho" / "train.tsv", sep="\t", keep_default_na=False)
    df["biblico"] = df["biblico"].astype(str).str.lower().eq("true")
    mantener = pd.concat([df[~df["biblico"]], df[df["biblico"]].sample(frac=frac_biblico, random_state=SEMILLA)])
    return mantener.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)


def armar_lotes(df: pd.DataFrame, batch: int) -> list[tuple[str, str, list[str], list[str]]]:
    """Lotes de una sola direccion (el tokenizador fija src/tgt por llamada)."""
    lotes = []
    for origen, destino in DIRECCIONES:
        src, tgt = df[origen].tolist(), df[destino].tolist()
        for i in range(0, len(df), batch):
            lotes.append((origen, destino, src[i:i + batch], tgt[i:i + batch]))
    random.shuffle(lotes)
    return lotes


def tokenizar(tok, lote, dispositivo):
    origen, destino, src, tgt = lote
    tok.src_lang, tok.tgt_lang = CODIGOS[origen], CODIGOS[destino]
    enc = tok(src, text_target=tgt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    enc["labels"][enc["labels"] == tok.pad_token_id] = -100
    return {k: v.to(dispositivo) for k, v in enc.items()}


@torch.inference_mode()
def evaluar(tok, modelo, valid: pd.DataFrame) -> dict[str, float]:
    modelo.eval()
    res: dict[str, float] = {}
    for origen, destino in DIRECCIONES:
        tok.src_lang = CODIGOS[origen]
        hip: list[str] = []
        textos = valid[origen].tolist()
        for i in range(0, len(textos), 32):
            enc = tok(textos[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(modelo.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = modelo.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODIGOS[destino]), num_beams=4, max_new_tokens=MAX_LEN)
            hip.extend(tok.batch_decode(gen, skip_special_tokens=True))
        res[f"chrF++_{origen}->{destino}"] = round(CHRF(word_order=2).corpus_score(hip, [valid[destino].tolist()]).score, 1)
    modelo.train()
    return res


def main() -> None:
    args = parsear_args()
    if not torch.cuda.is_available():
        raise SystemExit("Se necesita CUDA para entrenar; no se entrena en CPU.")
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    dispositivo = "cuda"

    train = cargar_train(args.frac_biblico)
    valid_all = pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho" / "valid.tsv", sep="\t", keep_default_na=False)
    valid_all["biblico"] = valid_all["biblico"].astype(str).str.lower().eq("true")
    valid = valid_all[~valid_all["biblico"]].sample(min(args.n_valid, int((~valid_all["biblico"]).sum())), random_state=SEMILLA)
    print(f"train: {len(train)} pares ({int(train['biblico'].sum())} biblicos) | valid no biblico: {len(valid)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(MODELO).to(dispositivo).train()  # pesos fp32, autocast bf16

    lotes_por_epoca = 2 * -(-len(train) // args.batch)
    total = args.max_pasos or lotes_por_epoca * args.epocas
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)

    args.salida.mkdir(parents=True, exist_ok=True)
    historial, mejor, paso = [], -1.0, 0
    t0 = time.time()
    for epoca in range(1, args.epocas + 1):
        perdidas: list[float] = []
        for lote in armar_lotes(train, args.batch):
            enc = tokenizar(tok, lote, dispositivo)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perdida = modelo(**enc).loss
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            perdidas.append(perdida.item())
            paso += 1
            if paso % 200 == 0:
                print(f"epoca {epoca} paso {paso}/{total} loss {sum(perdidas[-200:]) / 200:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if args.max_pasos and paso >= args.max_pasos:
                break
        metricas = evaluar(tok, modelo, valid)
        puntaje = sum(metricas.values()) / len(metricas)
        historial.append({"epoca": epoca, "loss": round(sum(perdidas) / max(len(perdidas), 1), 4), **metricas})
        print(historial[-1], flush=True)
        if puntaje > mejor:
            mejor = puntaje
            modelo.save_pretrained(args.salida / "mejor")
            tok.save_pretrained(args.salida / "mejor")
        if args.max_pasos and paso >= args.max_pasos:
            break

    (args.salida / "historial.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "historial": historial}, indent=2), encoding="utf-8")
    print(f"listo en {time.time() - t0:.0f}s; mejor chrF++ medio en valid no biblico: {mejor:.1f}", flush=True)


if __name__ == "__main__":
    main()
