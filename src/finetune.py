"""Fine-tuning de NLLB-200-distilled-600M para es<->quy (Ayacucho), bidireccional.

Ejemplo (un experimento completo = una configuracion de configs/):
  python -X utf8 src/finetune.py --nombre principal_s0 --semilla 0 --frac-datos 1.0

Decisiones (ver AGENTS.md §6):
  - Una variedad (Ayacucho): NLLB solo tiene quy_Latn y los papers no mezclan variedades.
  - Un modelo para ambas direcciones: cada par genera dos ejemplos (es->qu y qu->es).
  - Se entrena SOLO con texto no religioso (data/limpio/ayacucho_general, ver src/marcar_religioso.py).
  - Ablacion: --frac-datos usa una fraccion del entrenamiento (0.25, 0.5, 1.0).
  - Mejor punto por chrF++ medio en valid; ese punto se evalua UNA vez en test.
  - Precision: bf16 si la GPU lo soporta; si no (Colab T4) fp16 con GradScaler.
Solo usa PyTorch + transformers + sacrebleu (sin accelerate/Trainer).
Salidas: models/<nombre>/mejor (pesos), results/<nombre>/{historial,test}.json y test_predicciones.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "facebook/nllb-200-distilled-600M"
CODIGOS = {"es": "spa_Latn", "qu": "quy_Latn"}
DIRECCIONES = (("es", "qu"), ("qu", "es"))
MAX_LEN = 128
BATCH_EVAL = 32
BEAMS = 4
FRACCION_VRAM = 0.8


def parsear_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nombre", required=True, help="nombre del experimento (carpeta en models/ y results/)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--frac-datos", type=float, default=1.0, help="fraccion del entrenamiento a usar (ablacion de tamano)")
    p.add_argument("--train-archivo", default="train", help="nombre (sin .tsv) del entrenamiento en data/limpio/ayacucho_general; 'train' es el original")
    p.add_argument("--n-valid", type=int, default=500)
    p.add_argument("--n-test", type=int, default=0, help="0 = test completo")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--max-pasos", type=int, default=0, help="0 = sin limite; prueba de humo")
    p.add_argument("--guardar-pesos", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def leer(split: str) -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data" / "limpio" / "ayacucho_general" / f"{split}.tsv", sep="	", keep_default_na=False)


def cargar_train(frac: float, semilla: int, archivo: str = "train") -> pd.DataFrame:
    df = leer(archivo)
    return df.sample(frac=frac, random_state=semilla).reset_index(drop=True)


def armar_lotes(df: pd.DataFrame, batch: int) -> list[tuple[str, str, list[str], list[str]]]:
    lotes = []
    for origen, destino in DIRECCIONES:
        src, tgt = df[origen].tolist(), df[destino].tolist()
        lotes.extend((origen, destino, src[i:i + batch], tgt[i:i + batch]) for i in range(0, len(df), batch))
    random.shuffle(lotes)
    return lotes


def tokenizar(tok, lote, dispositivo):
    origen, destino, src, tgt = lote
    tok.src_lang, tok.tgt_lang = CODIGOS[origen], CODIGOS[destino]
    enc = tok(src, text_target=tgt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    enc["labels"][enc["labels"] == tok.pad_token_id] = -100
    return {k: v.to(dispositivo) for k, v in enc.items()}


def elegir_precision(pedida: str) -> torch.dtype:
    if pedida == "fp16":
        return torch.float16
    if pedida == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.inference_mode()
def traducir(tok, modelo, textos: list[str], origen: str, destino: str, dtipo: torch.dtype) -> list[str]:
    tok.src_lang = CODIGOS[origen]
    # Ordenar por longitud acelera el lote; se restaura el orden original al final.
    orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    salida: list[str] = [""] * len(textos)
    for i in range(0, len(orden), BATCH_EVAL):
        idx = orden[i:i + BATCH_EVAL]
        enc = tok([textos[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(modelo.device)
        with torch.autocast("cuda", dtype=dtipo):
            gen = modelo.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODIGOS[destino]), num_beams=BEAMS, max_new_tokens=MAX_LEN)
        for j, h in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            salida[j] = h
    return salida


def metricas(hip: list[str], ref: list[str]) -> dict[str, float]:
    return {"chrF++": round(CHRF(word_order=2).corpus_score(hip, [ref]).score, 1),
            "BLEU": round(BLEU().corpus_score(hip, [ref]).score, 1)}


def evaluar_valid(tok, modelo, valid: pd.DataFrame, dtipo) -> dict[str, float]:
    modelo.eval()
    res = {f"chrF++_{o}->{d}": metricas(traducir(tok, modelo, valid[o].tolist(), o, d, dtipo), valid[d].tolist())["chrF++"]
           for o, d in DIRECCIONES}
    modelo.train()
    return res


def evaluar_test(tok, modelo, test: pd.DataFrame, dtipo, salida: Path) -> dict:
    """Metricas por direccion y dominio + predicciones completas para el analisis de errores."""
    modelo.eval()
    filas, resumen = [], {}
    for o, d in DIRECCIONES:
        hip = traducir(tok, modelo, test[o].tolist(), o, d, dtipo)
        resumen[f"{o}->{d}"] = {**metricas(hip, test[d].tolist()), "n": len(test)}
        filas.append(pd.DataFrame({"direccion": f"{o}->{d}", "src": test[o], "ref": test[d], "hip": hip}))
    pd.concat(filas).to_csv(salida / "test_predicciones.tsv", sep="\t", index=False)
    return resumen


def main() -> None:
    args = parsear_args()
    if not torch.cuda.is_available():
        raise SystemExit("Se necesita CUDA para entrenar; no se entrena en CPU.")
    fijar_semillas(args.semilla)
    # Tope de VRAM: sin el, en Windows (WDDM) la GPU desborda a RAM compartida y se frena ~4x.
    torch.cuda.set_per_process_memory_fraction(FRACCION_VRAM)
    dtipo = elegir_precision(args.precision)
    usar_escala = dtipo == torch.float16
    escala = torch.amp.GradScaler("cuda", enabled=usar_escala)
    carpeta_res = RAIZ / "results" / args.nombre
    carpeta_mod = RAIZ / "models" / args.nombre
    carpeta_res.mkdir(parents=True, exist_ok=True)

    train = cargar_train(args.frac_datos, args.semilla, args.train_archivo)
    valid_all = leer("valid")
    valid = valid_all.sample(min(args.n_valid, len(valid_all)), random_state=args.semilla)
    test = leer("test")
    if args.n_test:
        test = test.sample(args.n_test, random_state=0).reset_index(drop=True)
    print(f"[{args.nombre}] precision={dtipo} train={len(train)} valid={len(valid)} test={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(MODELO).to("cuda").train()  # pesos fp32 + autocast
    lotes_por_epoca = 2 * -(-len(train) // args.batch)
    total = args.max_pasos or lotes_por_epoca * args.epocas
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)

    historial, mejor, paso = [], -1.0, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoca in range(1, args.epocas + 1):
        perdidas: list[float] = []
        for lote in armar_lotes(train, args.batch):
            enc = tokenizar(tok, lote, "cuda")
            with torch.autocast("cuda", dtype=dtipo):
                perdida = modelo(**enc).loss
            escala.scale(perdida).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(opt)
            escala.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            perdidas.append(perdida.item())
            paso += 1
            if paso % 500 == 0:
                torch.cuda.empty_cache()
            if paso % 200 == 0:
                print(f"epoca {epoca} paso {paso}/{total} loss {sum(perdidas[-200:]) / 200:.3f} ({time.time() - t0:.0f}s, pico {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
            if args.max_pasos and paso >= args.max_pasos:
                break
        m = evaluar_valid(tok, modelo, valid, dtipo)
        puntaje = sum(m.values()) / len(m)
        historial.append({"epoca": epoca, "loss": round(sum(perdidas) / max(len(perdidas), 1), 4), **m, "t_s": round(time.time() - t0)})
        print(historial[-1], flush=True)
        if puntaje > mejor:
            mejor = puntaje
            mejor_estado = {k: v.detach().clone().cpu() for k, v in modelo.state_dict().items()}
            historial[-1]["mejor"] = True
        if args.max_pasos and paso >= args.max_pasos:
            break

    t_train = time.time() - t0
    modelo.load_state_dict(mejor_estado)
    if args.guardar_pesos:
        modelo.save_pretrained(carpeta_mod / "mejor")
        tok.save_pretrained(carpeta_mod / "mejor")
    t1 = time.time()
    test_res = evaluar_test(tok, modelo, test, dtipo, carpeta_res)
    info = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "precision": str(dtipo),
        "gpu": torch.cuda.get_device_name(0),
        "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "horas_gpu_entrenamiento": round(t_train / 3600, 3),
        "horas_gpu_test": round((time.time() - t1) / 3600, 3),
        "mejor_chrF++_medio_valid": round(mejor, 1),
        "historial": historial,
    }
    (carpeta_res / "historial.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (carpeta_res / "test.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")
    print(json.dumps(test_res, indent=1), flush=True)
    print(f"listo: train {t_train / 60:.1f} min, mejor valid {mejor:.1f}", flush=True)


if __name__ == "__main__":
    main()
