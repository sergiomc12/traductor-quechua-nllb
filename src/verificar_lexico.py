"""Lote de verificacion lexica: frases en espanol (Tatoeba, CC-BY 2.0) -> quechua con el modelo -> revision palabra por palabra
contra diccionarios (RunaSimi ayacuchano, glosario MINEDU chanka, corpus de entrenamiento y Wikcionario en linea).

  python -X utf8 src/verificar_lexico.py --n 500 --modelo models/auto_it1/mejor --nombre lote1
Salida: results/lexico/<nombre>.tsv y <nombre>_resumen.md
Un par NO es 'verificado' por tener buena pinta: solo si cada palabra de contenido del espanol tiene respaldo en una fuente.
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
LEX = RAIZ / "data" / "lexico"
SAL = RAIZ / "results" / "lexico"

PARADAS = set("""el la los las un una unos unas de del al a en y o u que se su sus mi mis tu tus me te lo le les nos es son era fue ha han he
hay estoy esta estan esta estas este esto ese eso esos muy mas menos no si ya pero por para con sin sobre como cuando donde cual quien yo tu el ella
ellos ellas nosotros usted ustedes ser estar tener haber hacer ir voy va van vamos puede pueden podemos tengo tiene tienen""".split())


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def palabras_es(s: str) -> list[str]:
    return re.findall(r"[a-zñ]+", sin_tildes(s.lower()).replace("ñ", "ñ"))


def tema(w: str) -> str:
    return w[:5]


def contenido_es(s: str) -> list[str]:
    return [w for w in palabras_es(s) if w not in PARADAS and len(w) > 2]


def tokens_qu(s: str) -> list[str]:
    return re.findall(r"[a-zñ]+", s.lower())


def candidatos_raiz(t: str) -> set[str]:
    c = set()
    for i in range(3, len(t) + 1):
        c.add(t[:i]); c.add(t[:i] + "y")
    return c


# ---------- seleccion de frases ----------
def seleccionar(n: int) -> list[str]:
    train = pd.read_csv(RAIZ / "data/limpio/ayacucho_general/train.tsv", sep="\t", keep_default_na=False)
    vocab = {w for s in train["es"] for w in palabras_es(s)}
    ya = {s.strip().lower() for s in train["es"]}
    filas = []
    with bz2.open(LEX / "spa.tsv.bz2", "rt", encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            s = p[2].strip()
            w = s.split()
            if not 4 <= len(w) <= 10 or re.search(r"[\d\"“”@#/]|\.\.\.", s) or s.lower() in ya:
                continue
            if any(x[0].isupper() for x in w[1:] if x[0].isalpha()):
                continue  # nombres propios
            filas.append(s)
    # greedy: prioriza frases con palabras que el entrenamiento nunca vio (los vacios), sin repetir palabras nuevas entre frases
    vistas, elegidas = set(), []
    puntaje = lambda s: len({w for w in contenido_es(s) if w not in vocab} - vistas)
    cand = sorted(set(filas))
    import random
    random.Random(0).shuffle(cand)
    for _ in range(n):
        mejor = max(cand[:60000], key=puntaje)
        elegidas.append(mejor)
        vistas |= {w for w in contenido_es(mejor) if w not in vocab}
        cand.remove(mejor)
    return elegidas


# ---------- lexicos locales ----------
def lexico_runasimi() -> dict[str, set[str]]:
    r = pd.read_csv(RAIZ / "data/candidatos/runasimi_ayacucho_es.tsv", sep="\t", keep_default_na=False)
    d = defaultdict(set)
    for q, e in zip(r["qu"], r["es"]):
        for w in contenido_es(e):
            d[q.lower().strip()].add(tema(w))
    return d


def lexico_glosario() -> dict[str, set[str]]:
    d = defaultdict(set)
    for f in ("manual_quechua_chanka_glossary_entries.parquet", "manual_quechua_chanka_glossary_simple_terms.parquet"):
        g = pd.read_parquet(RAIZ / "data/candidatos" / f)
        for st, tt in zip(g["source_term"], g["target_text"]):
            es = {tema(w) for w in contenido_es(str(st))}
            for q in tokens_qu(str(tt)):
                d[q].update(es)
    return d


def lexico_corpus() -> dict[str, set[str]]:
    """Asociacion (tema es, token qu) en el corpus de entrenamiento: Dice >= 0.25 y >= 3 co-ocurrencias."""
    t = pd.read_csv(RAIZ / "data/limpio/ayacucho_general/train.tsv", sep="\t", keep_default_na=False)
    ce, cq, cc = Counter(), Counter(), Counter()
    for e, q in zip(t["es"], t["qu"]):
        se = {tema(w) for w in contenido_es(e)}
        sq = {x for x in tokens_qu(q) if len(x) > 2}
        ce.update(se); cq.update(sq)
        if len(se) * len(sq) > 400:
            continue
        for a in se:
            for b in sq:
                cc[(a, b)] += 1
    d = defaultdict(set)
    for (a, b), c in cc.items():
        if c >= 3 and 2 * c / (ce[a] + cq[b]) >= 0.25:
            d[b].add(a)
    return d


# ---------- Wikcionario en linea ----------
def wikcionario(palabra: str) -> list[str]:
    url = "https://es.wiktionary.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "titles": palabra, "prop": "revisions", "rvprop": "content", "rvslots": "main", "format": "json", "formatversion": 2})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraductorQuechua-MIA203/1.0 (investigacion academica)"})
        data = json.load(urllib.request.urlopen(req, timeout=20))
        txt = data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return []
    out = []
    for m in re.finditer(r"\{\{t\+?\|(?:qu|quz|quy|que|qub|qvc|qxu)\|([^}]*)\}\}", txt):
        out += [x.split("=")[-1] for x in m.group(1).split("|") if x and not x.startswith("i")]
    return [o.lower().strip() for o in out if o.strip()]


def glosbe(idioma: str, palabra: str) -> list[str]:
    """Traducciones es->quechua que lista Glosbe (meta description). idioma: quy (ayacuchano), quz (cuzqueno), qu (generico)."""
    import html
    url = f"https://es.glosbe.com/es/{idioma}/{urllib.parse.quote(palabra)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TraductorQuechua-MIA203 investigacion academica)"})
        h = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        return []
    m = re.search(r'name="description" content="([^"]*)"', h)
    if not m:
        return []
    d = html.unescape(m.group(1))
    mm = re.search(r"Glosbe:\s*([^.]*?)(?:\.|$)", d)
    return [x.strip().lower() for x in mm.group(1).split(",") if x.strip()] if mm else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--modelo", default="models/auto_it1/mejor")
    ap.add_argument("--nombre", default="lote1")
    a = ap.parse_args()
    SAL.mkdir(parents=True, exist_ok=True)
    frases = seleccionar(a.n)
    print("frases seleccionadas:", len(frases), flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from evaluar_flores import traducir
    tok = AutoTokenizer.from_pretrained(RAIZ / a.modelo)
    m = AutoModelForSeq2SeqLM.from_pretrained(RAIZ / a.modelo).to("cuda").eval()
    trad = traducir(tok, m, frases, "es", "qu", beams=4, lote=16, no_repeat_ngram_size=3)
    del m; torch.cuda.empty_cache()

    L = {"runasimi": lexico_runasimi(), "glosario": lexico_glosario(), "corpus": lexico_corpus()}
    print({k: len(v) for k, v in L.items()}, flush=True)

    def respaldo(tema_es: str, qu_toks: list[str]) -> str | None:
        for fuente, dic in L.items():
            for t in qu_toks:
                if any(tema_es in dic.get(c, ()) for c in candidatos_raiz(t)):
                    return fuente
        return None

    filas, faltan = [], Counter()
    for es, qu in zip(frases, trad):
        qt = [t for t in tokens_qu(qu) if len(t) > 1]
        ces = list(dict.fromkeys(contenido_es(es)))
        sin = [w for w in ces if respaldo(tema(w), qt) is None]
        filas.append({"es": es, "qu_modelo": qu, "contenido_es": " ".join(ces), "sin_respaldo_local": " ".join(sin)})
        faltan.update(sin)
    # segunda pasada: varias fuentes en linea (Wikcionario, Glosbe ayacuchano, Glosbe cuzqueno, Glosbe generico)
    cache_f = LEX / "online_cache.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    fuentes_online = {"wikcionario": wikcionario, "glosbe_quy": lambda w: glosbe("quy", w),
                      "glosbe_quz": lambda w: glosbe("quz", w), "glosbe_qu": lambda w: glosbe("qu", w)}
    palabras = list(faltan)
    print("consultando fuentes en linea:", len(palabras), "palabras x", len(fuentes_online), "fuentes", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def consultar(w: str) -> None:
        reg = cache.setdefault(w, {})
        for nom, fn in fuentes_online.items():
            if nom not in reg:
                reg[nom] = fn(w); time.sleep(0.2)

    with ThreadPoolExecutor(max_workers=3) as ex:  # 3 hilos: cortes con las fuentes, sin saturarlas
        for i, _ in enumerate(ex.map(consultar, palabras)):
            if i % 50 == 0:
                cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print("  ", i, "/", len(palabras), flush=True)
    cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    norm = lambda x: x.replace("'", "").replace("’", "").replace("ʼ", "").lower()

    def apoyos_online(w: str, qt: list[str]) -> list[str]:
        ok = []
        for nom, trs in cache.get(w, {}).items():
            if any(any(c == norm(y) or c.startswith(norm(y)) for c in candidatos_raiz(t)) for t in qt for y in trs if len(norm(y)) >= 3):
                ok.append(nom)
        return ok

    for f in filas:
        qt = [t for t in tokens_qu(f["qu_modelo"]) if len(t) > 1]
        ces = f["contenido_es"].split()
        det, sin, una = {}, [], []
        for w in ces:
            loc = [k for k, dic in L.items() if any(tema(w) in dic.get(c, ()) for t in qt for c in candidatos_raiz(t))]
            fuentes = loc + (apoyos_online(w, qt) if w in cache else [])
            det[w] = fuentes
            if not fuentes:
                sin.append(w)
            elif len(fuentes) == 1:
                una.append(w)
        f["respaldo"] = json.dumps(det, ensure_ascii=False)
        f["sin_respaldo"] = " ".join(sin)
        f["una_sola_fuente"] = " ".join(una)
        n_c = len(ces)
        f["cobertura"] = round(1 - len(sin) / n_c, 2) if n_c else 1.0
        f["estado"] = ("verificado" if not sin and not una else "una_fuente" if not sin else
                       "parcial" if f["cobertura"] >= 0.5 else "dudoso")
    df = pd.DataFrame(filas)
    df.to_csv(SAL / f"{a.nombre}.tsv", sep="\t", index=False)
    c = df["estado"].value_counts().to_dict()
    sin_global = Counter(w for s in df["sin_respaldo"] for w in s.split())
    (SAL / f"{a.nombre}_resumen.md").write_text(
        f"# Lote {a.nombre}\n\nFrases: {len(df)}\n\nEstados: {c}\n\nPalabras del español más frecuentes SIN respaldo en ninguna fuente ({len(sin_global)} distintas):\n\n"
        + ", ".join(f"{w} ({n})" for w, n in sin_global.most_common(80)) + "\n", encoding="utf-8")
    print(c)


if __name__ == "__main__":
    main()
