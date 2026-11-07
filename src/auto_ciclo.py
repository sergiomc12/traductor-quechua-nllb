"""Ciclo automatico: diagnostico de puntos ciegos -> datos para cubrirlos -> reentrenamiento -> evaluacion -> decision.

Reglas de seguridad (no negociables):
  * NO toca models/principal_s0, backup/, el servidor ni los numeros del paper.
  * Todo vive en models/<prefijo>_itN, results/<prefijo>_itN, data/auto y results/auto.
  * Las DECISIONES se toman con validacion propia + FLORES dev-check (297 oraciones de dev). FLORES devtest y la prueba propia
    solo se REGISTRAN; nunca se usan para elegir.
  * Un modelo nuevo se acepta solo si mejora el puntaje de seleccion en >= MARGEN y la validacion propia no cae > CAIDA_MAX.
  * Reanudable: estado en results/auto/estado.json. Si el proceso muere, volver a ejecutar el mismo comando.

  python -X utf8 src/auto_ciclo.py --horas 9 --esperar build/bt_exp.fin
  python -X utf8 src/auto_ciclo.py --prueba          # humo rapido de punta a punta (minutos)
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import glob
import gzip
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from evaluar_flores import repite, traducir  # noqa: E402
from marcar_religioso import RE as RELIGIOSO  # noqa: E402

PY = sys.executable
MARGEN, CAIDA_MAX = 0.3, 0.7
MINUTOS_POR_VUELTA = 100  # estimacion para decidir si cabe otra vuelta
TOK = lambda s: re.findall(r"[^\W\d_]+", s.lower())
NORM = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
CENTRAL = re.compile(r"\b\w+chaw\b|\b\w+pita\b|\b\w+ntsik\b|\bnoqa\b|\bnuqa\b|\bshu\w+|\b\w+shqa\b", re.I)
CUSCO = re.compile(r"[a-záéíóúñ][’'ʼ`´][a-záéíóúñ]|\w*(?:qh|kh|ph|th|chh)\w*", re.I)

A = RAIZ / "results" / "auto"
D_AUTO = RAIZ / "data" / "auto"
D_BT = RAIZ / "data" / "bt"
LOG = A / "log.txt"
ESTADO = A / "estado.json"


def log(msg: str) -> None:
    A.mkdir(parents=True, exist_ok=True)
    linea = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")


def mantener_despierto() -> None:
    """Evita la suspension por inactividad mientras corre el ciclo (no cambia ajustes del sistema)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"aviso: no se pudo evitar la suspension ({e})")


def cargar_estado() -> dict:
    if ESTADO.exists():
        return json.loads(ESTADO.read_text(encoding="utf-8"))
    return {"inicio": time.time(), "mejor": None, "iteraciones": [], "bt_aceptado": [], "sin_mejora": 0}


def guardar_estado(e: dict) -> None:
    ESTADO.write_text(json.dumps(e, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- datos
def leer(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, sep="\t", keep_default_na=False)


def flores(sp: str) -> pd.DataFrame:
    f = RAIZ / "data" / "flores" / "flores200_dataset" / sp
    return pd.DataFrame({"es": (f / f"spa_Latn.{sp}").read_text(encoding="utf-8").splitlines(), "qu": (f / f"quy_Latn.{sp}").read_text(encoding="utf-8").splitlines()})


def conjuntos(n: int | None) -> dict[str, pd.DataFrame]:
    """Conjuntos de evaluacion. 'sel' = los que deciden; el resto solo se registran."""
    L = RAIZ / "data" / "limpio" / "ayacucho_general"
    dev = flores("dev")
    check_idx = dev.sample(297, random_state=7).index
    s = {
        "valid": leer(L / "valid.tsv").sample(800, random_state=123).reset_index(drop=True),
        "devcheck": dev.loc[check_idx].reset_index(drop=True),
        "devtest": flores("devtest"),
        "test": leer(L / "test.tsv"),
    }
    return {k: (v.head(n) if n else v) for k, v in s.items()}


def flores_train() -> pd.DataFrame:
    dev = flores("dev")
    return dev.drop(dev.sample(297, random_state=7).index).reset_index(drop=True)


def runasimi_pares() -> pd.DataFrame:
    d = leer(RAIZ / "data" / "candidatos" / "runasimi_ayacucho_es.tsv")
    d["qu"] = d["qu"].str.split(r"[;,]").str[0].str.strip().str.lower()
    d["es"] = d["es"].str.split(";").str[0].str.strip().str.lower()
    d = d[(d["es"].str.split().str.len().between(1, 3)) & (d["qu"].str.len() > 1) & ~d["es"].str.contains(r"[()\[\]]") & ~d["qu"].str.contains(r"[()\[\]]")]
    d = d[~(d["es"] + " " + d["qu"]).str.contains(RELIGIOSO)]
    return d[["es", "qu"]].drop_duplicates().reset_index(drop=True)


def vocabulario(qus: list[str]) -> set[str]:
    return {w for s in qus for w in TOK(s)}


def pool_monolingue() -> pd.DataFrame:
    """Oraciones quechuas (Wikipedia qu, CC-100, corpus comunitario) filtradas por variedad, religion y ruido."""
    arch = D_AUTO / "pool.tsv"
    if arch.exists():
        return leer(arch)
    D_AUTO.mkdir(parents=True, exist_ok=True)
    C = RAIZ / "data" / "candidatos"
    txt = "\n".join(gzip.open(f, "rt", encoding="utf-8", errors="ignore").read() for f in sorted(glob.glob(str(C / "wiki" / "p*.gz"))))
    for extra in ("cc100-quechua.txt", "que_community_2017-sentences.txt"):
        txt += "\n" + (C / extra).read_text(encoding="utf-8", errors="ignore")
    excl = set()
    for sp in ("dev", "devtest"):
        excl |= {NORM(x) for x in flores(sp)["qu"]}
    for sp in ("valid", "test"):
        excl |= set(map(NORM, leer(RAIZ / "data" / "limpio" / "ayacucho_general" / f"{sp}.tsv")["qu"]))
    filas = {}
    for s in {x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", txt)}:
        w = s.split()
        if not 6 <= len(w) <= 40 or CENTRAL.search(s) or CUSCO.search(s) or RELIGIOSO.search(s) or NORM(s) in excl:
            continue
        if re.match(r"^[\d*\[(—\-•]", s) or "[" in s or "http" in s or sum(c.isalpha() for c in s) / max(len(s), 1) < 0.8:
            continue
        filas[s] = 1
    df = pd.DataFrame({"qu": list(filas)})
    plantilla = df["qu"].str.contains(r"sutiyuq (?:runa|warmi)|nisqa(?:qa)? .*huk \w+(?:m|am)\b", regex=True)
    df = pd.concat([df[~plantilla], df[plantilla].sample(min(int(plantilla.sum()), 1500), random_state=0)]).reset_index(drop=True)
    df.to_csv(arch, sep="\t", index=False)
    log(f"pool monolingue: {len(df)} oraciones")
    return df


def seleccionar_pool(pool: pd.DataFrame, vocab: set[str], gap: Counter, n: int, usadas: set[str]) -> pd.DataFrame:
    """Elige las oraciones que contienen mas palabras del vacio (desconocidas y presentes en FLORES dev / validacion)."""
    filas = []
    for s in pool["qu"]:
        if s in usadas:
            continue
        t = TOK(s)
        if not t:
            continue
        oov = sum(w not in vocab for w in t) / len(t)
        distintas = len({w for w in t if w in gap and w not in vocab})
        if distintas >= 1 and oov <= 0.5:
            filas.append((s, distintas, oov))
    df = pd.DataFrame(filas, columns=["qu", "gap", "oov"]).sort_values(["gap", "oov"], ascending=[False, True])
    sel = df.head(n)
    if len(sel) < n:  # completa con oraciones de alto OOV moderado
        resto = pool[~pool["qu"].isin(set(sel["qu"]) | usadas)].sample(frac=1, random_state=0).head(n - len(sel))
        sel = pd.concat([sel[["qu"]], resto[["qu"]]])
    return sel[["qu"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- modelo
def cargar_modelo(ruta: str):
    tok = AutoTokenizer.from_pretrained(ruta)
    m = AutoModelForSeq2SeqLM.from_pretrained(ruta).to("cuda").eval()
    return tok, m


def liberar(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def evaluar_modelo(ruta: str, etiqueta: str, rapido: bool) -> dict:
    tok, m = cargar_modelo(ruta)
    n = 30 if rapido else None
    res, preds = {}, {}
    chrf, bleu = CHRF(word_order=2), BLEU()
    for nom, df in conjuntos(n).items():
        for o, d in (("es", "qu"), ("qu", "es")):
            hip = traducir(tok, m, df[o].tolist(), o, d, beams=4, lote=16, no_repeat_ngram_size=3)
            res[f"{nom}_{o}->{d}"] = {"chrF++": round(chrf.corpus_score(hip, [df[d].tolist()]).score, 1), "BLEU": round(bleu.corpus_score(hip, [df[d].tolist()]).score, 1),
                                      "repeticiones": int(sum(repite(h) for h in hip)), "n": len(df)}
            if nom in ("valid", "devcheck"):
                preds[f"{nom}_{o}->{d}"] = hip
    liberar(m, tok)
    sel = [res[k]["chrF++"] for k in ("valid_es->qu", "valid_qu->es", "devcheck_es->qu", "devcheck_qu->es")]
    res["_seleccion"] = round(sum(sel) / 4, 2)
    res["_valid"] = round((res["valid_es->qu"]["chrF++"] + res["valid_qu->es"]["chrF++"]) / 2, 2)
    A.mkdir(parents=True, exist_ok=True)
    (A / f"eval_{etiqueta}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def diagnostico(ruta: str, vocab: set[str], rapido: bool) -> tuple[Counter, str]:
    """Puntos ciegos del modelo actual: palabras desconocidas del vacio y como cae el chrF++ segun el % de palabras no vistas."""
    tok, m = cargar_modelo(ruta)
    n = 30 if rapido else None
    sets = conjuntos(n)
    dev_all = flores("dev").head(n) if n else flores("dev")
    gap = Counter(w for s in list(dev_all["qu"]) + list(sets["valid"]["qu"]) for w in TOK(s) if w not in vocab)
    sc = CHRF(word_order=2)
    filas = []
    for nom in ("valid", "devcheck"):
        df = sets[nom]
        hip = traducir(tok, m, df["qu"].tolist(), "qu", "es", beams=4, lote=16, no_repeat_ngram_size=3)
        for s, h, r in zip(df["qu"], hip, df["es"]):
            t = TOK(s)
            filas.append((nom, sum(w not in vocab for w in t) / max(len(t), 1), sc.sentence_score(h, [r]).score))
    liberar(m, tok)
    d = pd.DataFrame(filas, columns=["conjunto", "oov", "chrf"])
    d["tramo"] = pd.cut(d["oov"], [-0.01, 0.05, 0.15, 0.3, 1.0], labels=["0-5 %", "5-15 %", "15-30 %", ">30 %"])
    tabla = d.groupby(["conjunto", "tramo"], observed=True)["chrf"].agg(["mean", "count"]).round(1)
    md = f"palabras desconocidas del vacio: {len(gap)} formas distintas ({sum(gap.values())} apariciones)\n\n```\n{tabla.to_string()}\n```"
    return gap, md


# ----------------------------------------------------------------------------- retrotraduccion
def generar_bt(ruta: str, qus: list[str], carpeta: Path, bloque: int) -> pd.DataFrame:
    carpeta.mkdir(parents=True, exist_ok=True)
    tok, m = cargar_modelo(ruta)
    chrf = CHRF(word_order=2)
    for i in range(0, len(qus), bloque):
        arch = carpeta / f"bloque_{i // bloque:04d}.tsv"
        if arch.exists():
            continue
        qu = qus[i:i + bloque]
        es = traducir(tok, m, qu, "qu", "es", beams=2, lote=48, no_repeat_ngram_size=3)
        vuelta = traducir(tok, m, es, "es", "qu", beams=2, lote=48, no_repeat_ngram_size=3)
        pd.DataFrame({"qu": qu, "es": es, "rt_chrf": [round(chrf.sentence_score(v, [q]).score, 1) for v, q in zip(vuelta, qu)]}).to_csv(arch.with_suffix(".tmp"), sep="\t", index=False)
        arch.with_suffix(".tmp").replace(arch)
        log(f"  bt bloque {i // bloque:04d} ({min(i + bloque, len(qus))}/{len(qus)})")
    liberar(m, tok)
    return pd.concat([leer(f) for f in sorted(carpeta.glob("bloque_*.tsv"))])


def filtrar_bt(d: pd.DataFrame, umbral: int) -> pd.DataFrame:
    nes, nqu = d["es"].str.split().str.len(), d["qu"].str.split().str.len().clip(lower=1)
    ok = nes.between(3, 60) & (nes / nqu).between(0.4, 3.0) & ~d["es"].map(repite) & (d["rt_chrf"] >= umbral)
    return d[ok][["es", "qu"]].drop_duplicates()


# ----------------------------------------------------------------------------- vuelta
def configurar(i: int, est: dict) -> dict | None:
    """Politica adaptativa. Devuelve None para detener el ciclo."""
    hist = est["iteraciones"]
    ult = hist[-1] if hist else None
    if i == 1:
        return {"bt_n": 6000, "umbral": 35, "extra": True, "epocas": 3, "motivo": "FLORES dev + RunaSimi + retrotraduccion dirigida a los vacios"}
    if est["sin_mejora"] >= 2:
        return None
    if ult and ult["aceptado"]:
        if i <= 3:
            return {"bt_n": 6000, "umbral": 35 if i == 2 else 40, "extra": True, "epocas": 3, "motivo": "se acepto la vuelta previa: mas retrotraduccion dirigida (lote nuevo)"}
        return {"bt_n": 0, "umbral": 35, "extra": True, "epocas": 4, "motivo": "ultima vuelta: mismos datos aceptados con 4 epocas"}
    if ult and ult["usa_bt"]:
        return {"bt_n": 0, "umbral": 35, "extra": True, "epocas": 3, "motivo": "la retrotraduccion no ayudo: solo FLORES dev + RunaSimi (ablacion)"}
    return {"bt_n": 4000, "umbral": 50, "extra": False, "epocas": 3, "motivo": "ultimo intento: retrotraduccion muy estricta (umbral 50), sin extras"}


def una_vuelta(i: int, est: dict, rapido: bool, pref: str) -> None:
    cfg = configurar(i, est)
    if cfg is None:
        log("politica: sin mas intentos utiles; fin del ciclo")
        est["terminado"] = True
        return
    log(f"=== vuelta {i}: {cfg['motivo']} | config {cfg}")
    mejor = est["mejor"]["ruta"]
    # 1. diagnostico del mejor modelo actual
    L = RAIZ / "data" / "limpio" / "ayacucho_general"
    D_AUTO.mkdir(parents=True, exist_ok=True)
    extra_df = pd.concat([pd.concat([flores_train()] * 4), runasimi_pares()]) if cfg["extra"] else pd.DataFrame(columns=["es", "qu"])
    bt_prev = [leer(RAIZ / p) for p in est["bt_aceptado"]]
    vocab = vocabulario(list(leer(L / "train.tsv")["qu"]) + list(extra_df["qu"]) + [q for b in bt_prev for q in b["qu"]])
    gap, md = diagnostico(mejor, vocab, rapido)
    (A / f"diagnostico_it{i}.md").write_text(f"# Diagnostico antes de la vuelta {i}\n\nmodelo: {mejor}\n\n{md}\n", encoding="utf-8")
    log(f"  diagnostico: {len(gap)} palabras del vacio")
    # 2. datos: retrotraduccion dirigida
    bt_nuevo = pd.DataFrame(columns=["es", "qu"])
    if cfg["bt_n"]:
        pool = pool_monolingue()
        archivo_usadas = D_AUTO / f"usadas_{pref}.txt"  # tambien se evita repetir las que no pasaron el control
        usadas = {q for b in bt_prev for q in b["qu"]} | (set(archivo_usadas.read_text(encoding="utf-8").splitlines()) if archivo_usadas.exists() else set())
        n = 60 if rapido else cfg["bt_n"]
        sel = seleccionar_pool(pool, vocab, gap, n, usadas)
        with archivo_usadas.open("a", encoding="utf-8") as f:
            f.write("\n".join(sel["qu"]) + "\n")
        gen = generar_bt(mejor, sel["qu"].tolist(), D_BT / f"{pref}_gen_it{i}", 60 if rapido else 1500)
        bt_nuevo = filtrar_bt(gen, cfg["umbral"])
        log(f"  retrotraduccion: {len(gen)} generadas -> {len(bt_nuevo)} tras control (umbral {cfg['umbral']})")
    bt_total = pd.concat([*bt_prev, bt_nuevo]).drop_duplicates() if (bt_prev or len(bt_nuevo)) else pd.DataFrame(columns=["es", "qu"])
    nombre = f"{pref}_it{i}"
    extra_df.to_csv(D_AUTO / f"extra_{nombre}.tsv", sep="\t", index=False)
    bt_total.to_csv(D_BT / f"auto_{nombre}.tsv", sep="\t", index=False)
    # 3. reentrenamiento desde el modelo base (carpeta nueva; no se sobrescribe nada)
    cmd = [PY, "-X", "utf8", str(RAIZ / "src" / "finetune.py"), "--nombre", nombre, "--semilla", "0", "--epocas", str(cfg["epocas"])]
    if cfg["extra"]:
        cmd += ["--extra-archivo", f"extra_{nombre}"]
    if len(bt_total):
        cmd += ["--bt-archivo", f"auto_{nombre}"]
    if rapido:
        cmd += ["--max-pasos", "20", "--epocas", "1", "--n-valid", "30", "--n-test", "30", "--no-guardar-pesos"]
    ruta_modelo = RAIZ / "models" / nombre / "mejor"
    if not (ruta_modelo / "config.json").exists() or rapido:
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (A / f"entrenamiento_{nombre}.log").write_text(r.stdout[-6000:] + "\n" + r.stderr[-3000:], encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"fallo el entrenamiento ({r.returncode}); ver results/auto/entrenamiento_{nombre}.log")
        log(f"  entrenado en {(time.time() - t0) / 60:.1f} min")
    # 4. evaluacion y decision
    if rapido:
        log("  [prueba] se omite la evaluacion del modelo entrenado (no se guardaron pesos); se evalua el modelo actual como humo")
        ev = evaluar_modelo(mejor, f"{nombre}_humo", True)
        est["iteraciones"].append({"vuelta": i, "config": cfg, "usa_bt": len(bt_total) > 0, "aceptado": False, "seleccion": ev["_seleccion"], "valid": ev["_valid"]})
        return
    ev = evaluar_modelo(str(ruta_modelo), nombre, False)
    b = est["mejor"]
    mejora = ev["_seleccion"] - b["seleccion"]
    caida = b["valid"] - ev["_valid"]
    aceptado = mejora >= MARGEN and caida <= CAIDA_MAX
    log(f"  seleccion {ev['_seleccion']} vs {b['seleccion']} (delta {mejora:+.2f}) | valid {ev['_valid']} vs {b['valid']} (caida {caida:+.2f}) -> {'ACEPTADO' if aceptado else 'RECHAZADO'}")
    est["iteraciones"].append({"vuelta": i, "nombre": nombre, "config": cfg, "usa_bt": len(bt_total) > 0, "aceptado": aceptado, "seleccion": ev["_seleccion"], "valid": ev["_valid"],
                               "devtest": {k: ev[k]["chrF++"] for k in ("devtest_es->qu", "devtest_qu->es")}, "test": {k: ev[k]["chrF++"] for k in ("test_es->qu", "test_qu->es")}})
    if aceptado:
        anterior = b["ruta"]
        est["mejor"] = {"ruta": str(ruta_modelo), "seleccion": ev["_seleccion"], "valid": ev["_valid"], "nombre": nombre}
        est["bt_aceptado"] = [f"data/bt/auto_{nombre}.tsv"] if len(bt_total) else est["bt_aceptado"]
        est["sin_mejora"] = 0
        if "principal_s0" not in anterior and f"{pref}_it" in anterior:  # libera disco: solo se conserva el mejor
            shutil.rmtree(Path(anterior).parent, ignore_errors=True)
    else:
        est["sin_mejora"] += 1
        shutil.rmtree(ruta_modelo.parent, ignore_errors=True)  # el rechazado no se conserva (sus metricas si)


def descubrir() -> None:
    """Busca datasets nuevos de quechua en el catalogo de Hugging Face. SOLO los registra: no se usan sin revision humana."""
    try:
        import urllib.parse
        import urllib.request
        conocidos = {"somosnlp-hackathon-2022/spanish-to-quechua", "pollitoconpapass/cuzco-quechua-translation-spanish", "gmezan/quechua-spanish-parallel", "Thermostatic/rosettia-chanka-data",
                     "somosnlp-hackathon-2026/rosettia-chanka-data", "makitty/quechua-chanka-unified", "makitty/quechua-chanka-unified-v2", "Llamacha/monolingual-quechua-iic",
                     "ffatty/plaintext-wikipedia-full-quechua", "VoiceProfiler/QuEsT"}
        nuevos = {}
        for q in ("quechua", "chanka", "ayacucho quechua", "quy spanish"):
            with urllib.request.urlopen(f"https://huggingface.co/api/datasets?search={urllib.parse.quote(q)}&limit=50", timeout=30) as r:
                for d in json.load(r):
                    if d["id"] not in conocidos:
                        nuevos[d["id"]] = (d.get("downloads"), [t[8:] for t in d.get("tags", []) if t.startswith("license:")], [t[9:] for t in d.get("tags", []) if t.startswith("language:")][:5])
        txt = "# Datasets de quechua en Hugging Face aun no revisados\n\nNo se usan automaticamente (licencia/procedencia sin verificar). Revisar a mano.\n\n| id | descargas | licencia | lenguas |\n|---|---|---|---|\n"
        txt += "\n".join(f"| {k} | {v[0]} | {', '.join(v[1]) or '?'} | {', '.join(v[2]) or '?'} |" for k, v in sorted(nuevos.items(), key=lambda x: -(x[1][0] or 0)))
        (A / "datasets_nuevos.md").write_text(txt + "\n", encoding="utf-8")
        log(f"descubrir: {len(nuevos)} datasets no revisados registrados en results/auto/datasets_nuevos.md")
    except Exception as e:  # noqa: BLE001
        log(f"descubrir fallo (no es critico): {e}")


def resumen(est: dict) -> None:
    filas = ["| Vuelta | Config | Seleccion | Valid propia | FLORES devtest es→qu / qu→es | Prueba propia es→qu / qu→es | Decision |", "|---|---|---|---|---|---|---|"]
    b = est.get("base")
    if b:
        filas.append(f"| base (principal_s0) | modelo actual | {b['seleccion']} | {b['valid']} | {b['devtest']['devtest_es->qu']} / {b['devtest']['devtest_qu->es']} | {b['test']['test_es->qu']} / {b['test']['test_qu->es']} | referencia |")
    for it in est["iteraciones"]:
        c = it["config"]
        dv = it.get("devtest", {})
        ts = it.get("test", {})
        filas.append(f"| {it['vuelta']} | {c['motivo']} | {it['seleccion']} | {it['valid']} | {dv.get('devtest_es->qu', '-')} / {dv.get('devtest_qu->es', '-')} | {ts.get('test_es->qu', '-')} / {ts.get('test_qu->es', '-')} | {'ACEPTADO' if it['aceptado'] else 'rechazado'} |")
    txt = f"# Ciclo automatico: resumen\n\nActualizado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nMejor modelo actual: `{est['mejor']['ruta']}` (seleccion {est['mejor']['seleccion']})\n\n"
    txt += "Seleccion = media de chrF++ en {valid propia, FLORES dev-check} x {es→qu, qu→es}. FLORES devtest y la prueba propia solo se registran.\n\n" + "\n".join(filas) + "\n\n"
    txt += "Para volver atras: el modelo original esta intacto en `models/principal_s0` y respaldado en `backup/v1_principal_s0` (etiqueta git v1.0-baseline-5nov).\n"
    (A / "RESUMEN.md").write_text(txt, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=9.0)
    ap.add_argument("--max-vueltas", type=int, default=4)
    ap.add_argument("--esperar", default="", help="archivo cuya aparicion indica que la GPU quedo libre")
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    global A, LOG, ESTADO
    pref = "autoprueba" if a.prueba else "auto"
    if a.prueba:  # aislado: estado y log propios; los datos usan nombres con prefijo, asi no chocan con la corrida real
        A = A.parent / "auto_prueba"
        LOG, ESTADO = A / "log.txt", A / "estado.json"
        shutil.rmtree(A, ignore_errors=True)
    mantener_despierto()
    A.mkdir(parents=True, exist_ok=True)
    if a.esperar:
        log(f"esperando {a.esperar} (GPU ocupada por la cadena anterior)")
        espera0 = time.time()
        while not (RAIZ / a.esperar).exists():
            if time.time() - espera0 > 4 * 3600:  # tope: si la cadena anterior fallo, no esperar para siempre
                log("tope de espera (4 h) alcanzado: se continua igualmente")
                break
            time.sleep(30)
    est = cargar_estado()
    inicio = est["inicio"]
    limite = inicio + a.horas * 3600
    descubrir()
    if est["mejor"] is None:
        log("evaluando el modelo base (principal_s0) con el mismo protocolo")
        ev = evaluar_modelo(str(RAIZ / "models" / "principal_s0" / "mejor"), "base_principal_s0", a.prueba)
        est["mejor"] = {"ruta": str(RAIZ / "models" / "principal_s0" / "mejor"), "seleccion": ev["_seleccion"], "valid": ev["_valid"], "nombre": "principal_s0"}
        est["base"] = {"seleccion": ev["_seleccion"], "valid": ev["_valid"], "devtest": {k: ev[k]["chrF++"] for k in ("devtest_es->qu", "devtest_qu->es")}, "test": {k: ev[k]["chrF++"] for k in ("test_es->qu", "test_qu->es")}}
        guardar_estado(est)
        resumen(est)
    for i in range(len(est["iteraciones"]) + 1, a.max_vueltas + 1):
        if est.get("terminado"):
            break
        restante = (limite - time.time()) / 60
        if not a.prueba and restante < MINUTOS_POR_VUELTA:
            log(f"quedan {restante:.0f} min (< {MINUTOS_POR_VUELTA}): no se inicia otra vuelta")
            break
        try:
            una_vuelta(i, est, a.prueba, pref)
        except Exception:  # noqa: BLE001
            log(f"ERROR en la vuelta {i}:\n{traceback.format_exc()}")
            est["iteraciones"].append({"vuelta": i, "config": {"motivo": "fallo (ver log)"}, "usa_bt": False, "aceptado": False, "seleccion": 0, "valid": 0})
            est["sin_mejora"] += 1
            torch.cuda.empty_cache()
        guardar_estado(est)
        resumen(est)
    log("CICLO TERMINADO")
    guardar_estado(est)
    resumen(est)
    (A / "FIN").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


if __name__ == "__main__":
    main()
