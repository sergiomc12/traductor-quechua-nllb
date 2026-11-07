"""Sensibilidad del modelo al apóstrofo en palabras con eyectiva (léxico normalizado): ' recto, ’ tipográfico y sin apóstrofo."""
import json, re, sys, unicodedata
from pathlib import Path
import pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import finetune as ft
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
RAIZ = ft.RAIZ
APO = "'’‘´`ʼ"
INTERNO = re.compile(rf"(?<=\w)[{APO}](?=\w)")
norm = lambda s: "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
lex = pd.read_csv(RAIZ / "data/lexico_normalizado.csv", encoding="utf-8-sig", keep_default_na=False)
items = []
for _, r in lex.iterrows():
    for forma in {r["quechua"], r["variante_ayacucho"], r["variante_alterna"]}:
        if forma and INTERNO.search(forma):
            items.append((forma, [g.strip() for g in re.split(r"[/,;]", r["espanol"]) if g.strip()]))
items = sorted(set((f, tuple(g)) for f, g in items))
variantes = {"recto '": lambda w: INTERNO.sub("'", w), "tipográfico ’": lambda w: INTERNO.sub("’", w), "sin apóstrofo": lambda w: INTERNO.sub("", w)}
ruta = RAIZ / "models" / "exp_norm15k_s0" / "mejor"
tok = AutoTokenizer.from_pretrained(ruta)
modelo = AutoModelForSeq2SeqLM.from_pretrained(ruta).to("cuda").eval()
dt = ft.elegir_precision("auto")
res = {}
for nombre, f in variantes.items():
    ent = [f(w) for w, _ in items]
    sal = ft.traducir(tok, modelo, ent, "qu", "es", dt)
    ok = [any(norm(g) in norm(s) for g in gl) for s, (_, gl) in zip(sal, items)]
    res[nombre] = {"aciertos": sum(ok), "n": len(items), "detalle": [(e, s, o) for e, s, o in zip(ent, sal, ok)]}
    print(nombre, sum(ok), "de", len(items))
# español -> quechua: ¿escribe el apóstrofo y en qué forma?
es_ent = sorted({g[0] for _, g in items})
es_sal = ft.traducir(tok, modelo, es_ent, "es", "qu", dt)
res["es->qu"] = list(zip(es_ent, es_sal))
print("es->qu:", res["es->qu"])
Path(RAIZ / "results" / "prueba_apostrofo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
