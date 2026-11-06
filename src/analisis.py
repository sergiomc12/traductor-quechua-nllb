"""Agrega resultados (R4) y analiza errores (R5). Escribe results/tablas.md y results/fig_*.png.

  python -X utf8 src/analisis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
GRUPOS = {"principal (100 % de los datos)": ["principal_s0", "principal_s1", "principal_s2"],
          "ablacion: 50 % de los datos": ["ablacion_datos50_s0"],
          "ablacion: 25 % de los datos": ["ablacion_datos25_s0"]}
DIRS = ["es->qu", "qu->es"]
SUFIJOS = ["kuna", "pa", "manta", "wan", "man", "pi", "paq", "mi", "taq", "chik", "ku", "sqa"]


def cargar(nombre: str) -> dict | None:
    f = RES / nombre / "test.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def pm(x: np.ndarray) -> str:
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}" if len(x) > 1 else f"{x.mean():.1f}"


def tabla_principal() -> str:
    filas = ["| Sistema | Direccion | chrF++ | BLEU |", "|---|---|---|---|"]
    base = cargar("baseline_zeroshot")
    if base:
        for d in DIRS:
            filas.append(f"| zero-shot NLLB-600M (linea base) | {d} | {base[d]['chrF++']} | {base[d]['BLEU']} |")
    for grupo, nombres in GRUPOS.items():
        rs = [r for r in map(cargar, nombres) if r]
        for d in DIRS if rs else []:
            c = np.array([r[d]["chrF++"] for r in rs])
            b = np.array([r[d]["BLEU"] for r in rs])
            filas.append(f"| {grupo} (n={len(rs)}) | {d} | {pm(c)} | {pm(b)} |")
    return chr(10).join(filas)


def costo() -> str:
    filas = ["| Experimento | GPU | Precision | Pico VRAM (GB) | Horas GPU (entren.) | Mejor chrF++ valid |", "|---|---|---|---|---|---|"]
    total = 0.0
    for nombres in GRUPOS.values():
        for n in nombres:
            f = RES / n / "historial.json"
            if f.exists():
                h = json.loads(f.read_text(encoding="utf-8"))
                total += h["horas_gpu_entrenamiento"] + h["horas_gpu_test"]
                filas.append(f"| {n} | {h['gpu']} | {h['precision']} | {h['pico_vram_gb']} | {h['horas_gpu_entrenamiento']} | {h['mejor_chrF++_medio_valid']} |")
    filas.append(f"\nTotal horas GPU (entrenamiento + test): **{total:.2f} h**")
    return "\n".join(filas)


def chrf_seg(hip: pd.Series, ref: pd.Series) -> float:
    return CHRF(word_order=2).corpus_score(hip.tolist(), [ref.tolist()]).score if len(hip) else float("nan")


def errores(nombre: str = "principal_s0") -> str:
    df = pd.read_csv(RES / nombre / "test_predicciones.tsv", sep="\t", keep_default_na=False)
    df["n_pal_ref"] = df["ref"].str.split().str.len()
    df["bucket"] = pd.cut(df["n_pal_ref"], [0, 5, 10, 20, 40, 10_000], labels=["1-5", "6-10", "11-20", "21-40", ">40"])
    out = []
    out.append("### chrF++ por longitud de la referencia (palabras)\n\n| Direccion | " + " | ".join(["1-5", "6-10", "11-20", "21-40", ">40"]) + " |\n|---|---|---|---|---|---|")
    for d in DIRS:
        sub = df[df["direccion"] == d]
        out.append(f"| {d} | " + " | ".join(f"{chrf_seg(g['hip'], g['ref']):.1f} (n={len(g)})" for _, g in sub.groupby("bucket", observed=False)) + " |")
    # Longitud de la hipotesis frente a la referencia: subtraduccion / sobretraduccion
    df["ratio_len"] = df["hip"].str.len() / df["ref"].str.len().clip(lower=1)
    out.append("\n### Ratio de longitud hipotesis/referencia (mediana) y % de salidas vacias o identicas a la fuente\n\n| Direccion | Mediana ratio | % vacias | % copia de la fuente |\n|---|---|---|---|")
    for d in DIRS:
        sub = df[df["direccion"] == d]
        out.append(f"| {d} | {sub['ratio_len'].median():.2f} | {100 * (sub['hip'].str.strip() == '').mean():.1f} | {100 * (sub['hip'] == sub['src']).mean():.1f} |")
    # Sufijos en el quechua de referencia (es->qu): fraccion de palabras con el sufijo que la hipotesis reproduce
    sub = df[df["direccion"] == "es->qu"]
    out.append("\n### es->qu: recall de sufijos (palabras de la referencia que terminan en el sufijo y aparecen exactas en la hipotesis)\n\n| Sufijo | Apariciones en ref. | Recall |\n|---|---|---|")
    filas = []
    for suf in SUFIJOS:
        tot = hit = 0
        for ref, hip in zip(sub["ref"], sub["hip"]):
            hp = set(re.findall(r"\w+", hip.lower()))
            for w in re.findall(r"\w+", ref.lower()):
                if w.endswith(suf) and len(w) > len(suf) + 1:
                    tot += 1
                    hit += w in hp
        if tot:
            filas.append((hit / tot, f"| -{suf} | {tot} | {100 * hit / tot:.1f}% |"))
    out.extend(r for _, r in sorted(filas, reverse=True))
    # Peores ejemplos (chrF por frase) para el analisis cualitativo
    from sacrebleu.metrics import CHRF as C
    sentencia = C(word_order=2)
    df["chrf"] = [sentencia.sentence_score(h, [r]).score for h, r in zip(df["hip"], df["ref"])]
    out.append("\n### Ejemplos de error (chrF++ por frase mas bajo con ref. de 6-30 palabras)\n")
    pool = df[df["n_pal_ref"].between(6, 30)].nsmallest(6, "chrf")
    for _, r in pool.iterrows():
        out.append(f"- **{r['direccion']}** (chrF++ {r['chrf']:.0f})\n  - src: {r['src']}\n  - ref: {r['ref']}\n  - hip: {r['hip']}")
    return "\n".join(out)


def figuras() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for (nombre, nombres), color in zip(GRUPOS.items(), ("C0", "C1", "C2")):
        for i, n in enumerate(nombres):
            f = RES / n / "historial.json"
            if not f.exists():
                continue
            h = json.loads(f.read_text(encoding="utf-8"))["historial"]
            ep = [x["epoca"] for x in h]
            et = nombre.split(" (")[0] if i == 0 else None
            ax[0].plot(ep, [x["loss"] for x in h], color=color, marker="o", label=et)
            ax[1].plot(ep, [(x["chrF++_es->qu"] + x["chrF++_qu->es"]) / 2 for x in h], color=color, marker="o", label=et)
    ax[0].set(xlabel="Epoca", ylabel="Perdida de entrenamiento (media)", title="Perdida")
    ax[1].set(xlabel="Epoca", ylabel="chrF++ medio (validacion)", title="Validacion")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=7); a_.set_xticks([1, 2, 3])
    fig.tight_layout()
    fig.savefig(RES / "fig_curvas.png", dpi=200)


if __name__ == "__main__":
    texto = "## Resultados en test (media ± desv. est. entre semillas)\n\n" + tabla_principal() + "\n\n## Costo computacional\n\n" + costo() + "\n\n## Analisis de errores (principal_s0)\n\n" + errores()
    (RES / "tablas.md").write_text(texto, encoding="utf-8")
    figuras()
    print(texto)


def metricas_errores(nombre: str = "principal_s0") -> dict:
    """Cifras del analisis de errores en JSON (las lee paper/resultados.py)."""
    df = pd.read_csv(RES / nombre / "test_predicciones.tsv", sep="\t", keep_default_na=False)
    mes = r"(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"

    def ruido(s: str) -> bool:
        s = s.strip()
        if not s or s.startswith("[") or re.search(mes, s) or re.match(r"(?i)^p[aá]gina|^pag\b", s):
            return True
        return len(s) < 4 or sum(c.isalpha() for c in s) / max(len(s), 1) < 0.5

    def repite(h: str) -> bool:
        w = h.lower().split()
        tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
        return bool(re.search(r"\b(\S+)(?: \1){3,}", h.lower())) or any(tri.count(t) >= 3 for t in set(tri))

    df["ruido"] = [ruido(a) or ruido(b) for a, b in zip(df["src"], df["ref"])]
    df["rep"] = [repite(h) for h in df["hip"]]
    df["n_pal"] = df["ref"].str.split().str.len()
    df["bucket"] = pd.cut(df["n_pal"], [0, 5, 10, 20, 40, 10_000], labels=["1-5", "6-10", "11-20", "21-40", ">40"])
    f = lambda x: round(chrf_seg(x["hip"], x["ref"]), 1)
    out: dict = {"n_pares": int(len(df) // 2), "n_ruido": int(df["ruido"].sum() // 2), "por_direccion": {}}
    for d in DIRS:
        s = df[df["direccion"] == d]
        suf = {}
        if d == "es->qu":
            for sf in SUFIJOS:
                tot = hit = 0
                for ref, hip in zip(s["ref"], s["hip"]):
                    hp = set(re.findall(r"\w+", hip.lower()))
                    for w in re.findall(r"\w+", ref.lower()):
                        if w.endswith(sf) and len(w) > len(sf) + 1:
                            tot += 1
                            hit += w in hp
                suf[sf] = {"n": tot, "recall": round(100 * hit / max(tot, 1), 1)}
        out["por_direccion"][d] = {
            "chrf_completo": f(s), "chrf_sin_ruido": f(s[~s["ruido"]]), "chrf_solo_ruido": f(s[s["ruido"]]),
            "n_sin_ruido": int((~s["ruido"]).sum()), "repeticiones": int(s["rep"].sum()),
            "por_longitud": {str(k): {"chrf": f(g), "n": int(len(g))} for k, g in s.groupby("bucket", observed=False)},
            "sufijos": suf,
        }
    return out


if __name__ == "__main__":
    (RES / "errores.json").write_text(json.dumps(metricas_errores(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("results/errores.json escrito")
