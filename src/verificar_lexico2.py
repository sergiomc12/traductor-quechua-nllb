"""Verificador v2 sobre un lote ya traducido (results/lexico/<nombre>.tsv): lematiza el espanol, separa nombres propios y calcos
(palabras del espanol copiadas al quechua) y agrega busqueda INVERSA (palabra quechua del modelo -> significados en espanol)
en Glosbe quy/quz/qu y Wikcionario, ademas de la directa. Un par es 'verificado' solo si cada palabra de contenido
(no nombre, no calco) tiene >= 2 fuentes independientes.

  python -X utf8 src/verificar_lexico2.py --nombre lote1
Salida: results/lexico/<nombre>_v2.tsv y <nombre>_v2_resumen.md
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import simplemma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verificar_lexico import (LEX, PARADAS, SAL, RAIZ, contenido_es, glosbe, lexico_corpus, lexico_glosario,  # noqa: E402
                              lexico_runasimi, sin_tildes, tema, tokens_qu, wikcionario)

SUFIJOS = sorted("""qa ta pi wan man manta pa nchik nchis yki ykichik niy nki ni n m mi si chu taq kuna cha ku ra mu pu yku chka rqa sqa
spa spaqa nqa saq sun rqan rqanku nku wan ymi y q sqan lla ña pas pis qa taqmi hina pura raq""".split(), key=len, reverse=True)


def lema_es(w: str) -> str:
    return sin_tildes(simplemma.lemmatize(w, lang="es").lower())


def raices_qu(t: str) -> list[str]:
    """Candidatas de raiz: la palabra, tras pelar 1 y 2 sufijos, y con -y (infinitivo)."""
    sal, nivel = [t], [t]
    for _ in range(2):
        nuevo = []
        for x in nivel:
            for s in SUFIJOS:
                if x.endswith(s) and len(x) - len(s) >= 3:
                    nuevo.append(x[: -len(s)])
        nivel = list(dict.fromkeys(nuevo))
        sal += nivel
    sal += [r + "y" for r in sal if not r.endswith("y")]
    return list(dict.fromkeys(sal))


def glosbe_inv(idioma: str, palabra: str) -> list[str]:
    """Significados en espanol de una palabra quechua (meta description de Glosbe quechua->espanol)."""
    url = f"https://es.glosbe.com/{idioma}/es/{urllib.parse.quote(palabra)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TraductorQuechua-MIA203 investigacion academica)"})
        h = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        return []
    m = re.search(r'name="description" content="([^"]*)"', h)
    if not m:
        return []
    mm = re.search(r"al español\?:\s*([^.]*?)(?:\.|$)", html.unescape(m.group(1)))
    return [x.strip().lower() for x in mm.group(1).split(",") if x.strip()] if mm else []


def wik_inv(palabra: str) -> list[str]:
    """es.wiktionary no tiene entradas quechuas utiles en general; se consulta la API por si existe la pagina."""
    url = "https://es.wiktionary.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "titles": palabra, "prop": "extracts", "explaintext": 1, "exintro": 1, "format": "json", "formatversion": 2})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraductorQuechua-MIA203/1.0 (investigacion academica)"})
        data = json.load(urllib.request.urlopen(req, timeout=20))
        txt = data["query"]["pages"][0].get("extract", "")
    except Exception:
        return []
    return [lema_es(w) for w in re.findall(r"[a-záéíóúñ]+", txt.lower())][:60] if "quechua" in txt.lower() else []


FUENTES_DIR = {"wikcionario": wikcionario, "glosbe_quy": lambda w: glosbe("quy", w),
               "glosbe_quz": lambda w: glosbe("quz", w), "glosbe_qu": lambda w: glosbe("qu", w)}
FUENTES_INV = {"inv_glosbe_quy": lambda w: glosbe_inv("quy", w), "inv_glosbe_quz": lambda w: glosbe_inv("quz", w),
               "inv_glosbe_qu": lambda w: glosbe_inv("qu", w), "inv_wik": wik_inv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="lote1")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_csv(SAL / f"{a.nombre}.tsv", sep="\t", keep_default_na=False)
    L = {"runasimi": lexico_runasimi(), "glosario": lexico_glosario(), "corpus": lexico_corpus()}
    vocab_train = {w for s in pd.read_csv(RAIZ / "data/limpio/ayacucho_general/train.tsv", sep="\t", keep_default_na=False)["es"]
                   for w in re.findall(r"[a-zñ]+", sin_tildes(s.lower()))}

    # ---- analisis por frase: palabras, lemas, nombres, calcos ----
    info = []
    dir_pend, inv_pend = set(), set()
    for es, qu in zip(df["es"], df["qu_modelo"]):
        qt = [t for t in tokens_qu(qu) if len(t) > 1]
        primera = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]+", es)[:1]
        pals = []
        for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]+", es):
            wl = sin_tildes(w.lower())
            if wl in PARADAS or len(wl) <= 2:
                continue
            lem = lema_es(w)
            if lem in PARADAS:
                continue
            pals.append((wl, lem))
        pals = list(dict.fromkeys(pals))
        reg = {"nombres": [], "calcos": [], "reales": []}
        for wl, lem in pals:
            es_nombre = bool(primera) and sin_tildes(primera[0].lower()) == wl and primera[0][0].isupper() \
                and wl not in vocab_train and any(t.startswith(wl) for t in qt)
            calco = any(difflib.SequenceMatcher(None, sin_tildes(t), x).ratio() >= 0.8 or sin_tildes(t).startswith(x[:6])
                        for t in qt for x in (wl, lem) if len(x) >= 4)
            if es_nombre:
                reg["nombres"].append(wl)
            elif calco:
                reg["calcos"].append(wl)
            else:
                reg["reales"].append((wl, lem))
                dir_pend.add(lem)
        for t in qt:
            for r in raices_qu(t)[:4]:
                inv_pend.add(r)
        info.append((qt, reg))

    cache_f = LEX / "online_cache_v2.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    old = json.loads((LEX / "online_cache.json").read_text(encoding="utf-8")) if (LEX / "online_cache.json").exists() else {}
    for w, v in old.items():
        cache.setdefault("dir:" + w, v)
    tareas = [("dir:" + w, w, FUENTES_DIR) for w in sorted(dir_pend)] + [("inv:" + r, r, FUENTES_INV) for r in sorted(inv_pend)]
    print("consultas pendientes:", len(tareas), flush=True)

    def consultar(t):
        clave, palabra, fuentes = t
        reg = cache.setdefault(clave, {})
        for nom, fn in fuentes.items():
            if nom not in reg:
                reg[nom] = fn(palabra); time.sleep(0.15)

    with ThreadPoolExecutor(max_workers=a.hilos) as ex:
        for i, _ in enumerate(ex.map(consultar, tareas)):
            if i % 100 == 0:
                cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print("  ", i, "/", len(tareas), flush=True)
    cache_f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    norm = lambda x: x.replace("'", "").replace("’", "").replace("ʼ", "").lower()

    def apoyos(wl: str, lem: str, qt: list[str]) -> list[str]:
        ok = []
        raices = {c for t in qt for c in raices_qu(t)}
        for fuente, dic in L.items():
            if any(tema(lem) in dic.get(c, ()) or tema(wl) in dic.get(c, ()) for c in raices):
                ok.append(fuente)
        for nom, trs in cache.get("dir:" + lem, {}).items():  # directa: espanol -> quechua listado por la fuente
            if any(len(norm(y)) >= 3 and any(c == norm(y) or c.startswith(norm(y)) for c in raices) for y in trs):
                ok.append(nom)
        for nom in FUENTES_INV:  # inversa: quechua del modelo -> significados en espanol
            hit = False
            for r in raices:
                sig = cache.get("inv:" + r, {}).get(nom, [])
                if any(lema_es(x) == lem or lema_es(x) == sin_tildes(wl) or lem in sin_tildes(x).split() for x in sig):
                    hit = True; break
            if hit:
                ok.append(nom)
        return ok

    filas = []
    for (es, qu), (qt, reg) in zip(zip(df["es"], df["qu_modelo"]), info):
        det = {wl: apoyos(wl, lem, qt) for wl, lem in reg["reales"]}
        sin = [w for w, f in det.items() if not f]
        una = [w for w, f in det.items() if len(f) == 1]
        n = len(det)
        cob = round(1 - len(sin) / n, 2) if n else 1.0
        if not n:
            est = "solo_calcos_o_nombres"
        elif not sin and not una and not reg["calcos"]:
            est = "verificado"
        elif not sin and not una:
            est = "verificado_con_calcos"
        elif not sin:
            est = "una_fuente"
        else:
            est = "parcial" if cob >= 0.5 else "dudoso"
        filas.append({"es": es, "qu_modelo": qu, "estado": est, "cobertura": cob, "nombres": " ".join(reg["nombres"]),
                      "calcos": " ".join(reg["calcos"]), "sin_respaldo": " ".join(sin), "una_sola_fuente": " ".join(una),
                      "respaldo": json.dumps(det, ensure_ascii=False)})
    out = pd.DataFrame(filas)
    out.to_csv(SAL / f"{a.nombre}_v2.tsv", sep="\t", index=False)
    c = out["estado"].value_counts().to_dict()
    sin_g = Counter(w for s in out["sin_respaldo"] for w in s.split())
    calc_g = Counter(w for s in out["calcos"] for w in s.split())
    src = Counter(f for r in out["respaldo"] for v in json.loads(r).values() for f in v)
    (SAL / f"{a.nombre}_v2_resumen.md").write_text(
        f"# Lote {a.nombre} (verificador v2)\n\nFrases: {len(out)}\n\nEstados: {c}\n\nPalabras respaldadas por fuente: {dict(src)}\n\n"
        f"Calcos del español más frecuentes ({len(calc_g)} distintos): " + ", ".join(f"{w} ({n})" for w, n in calc_g.most_common(60))
        + f"\n\nSin respaldo ({len(sin_g)} distintas): " + ", ".join(f"{w} ({n})" for w, n in sin_g.most_common(60)) + "\n", encoding="utf-8")
    print(c, dict(src))


if __name__ == "__main__":
    main()
