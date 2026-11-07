"""Genera las figuras del paper que no salen de un entrenamiento: arquitectura y concordancia por persona/tiempo.

Ambas miden el ancho de una columna IEEE (3.4 pulgadas) para que el texto se lea igual en el PDF y en el Word.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = Path(__file__).resolve().parent.parent / "results"


def _caja(ax, x, y, w, h, texto, color="#dbe7f5", tam=7.5, borde="#35506b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec=borde, lw=1))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam)


def _flecha(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, color="#35506b"))


def arquitectura(destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.9), dpi=250)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.6)
    ax.axis("off")
    _caja(ax, 1.5, 10.1, 7.0, 1.2, "Oración fuente (es o qu)", "#f2efe6", tam=7)
    _caja(ax, 1.5, 8.2, 7.0, 1.4, "Etiqueta de idioma + SentencePiece\n(vocabulario de 256 206 tokens)", "#f2efe6", tam=6.5)
    _caja(ax, 1.5, 5.2, 7.0, 2.4, "", "#dbe7f5")
    ax.text(5.0, 7.15, "Codificador", ha="center", fontsize=7.5, weight="bold")
    ax.text(5.0, 6.0, "12 capas Transformer, d = 1024\n16 cabezas, FFN 4096", ha="center", va="center", fontsize=6.5)
    _caja(ax, 1.5, 2.2, 7.0, 2.4, "", "#e3f0dc", borde="#3d6b35")
    ax.text(5.0, 4.15, "Decodificador", ha="center", fontsize=7.5, weight="bold")
    ax.text(5.0, 3.0, "12 capas con atención cruzada\n(todos los pesos se ajustan)", ha="center", va="center", fontsize=6.5)
    _caja(ax, 1.5, 0.1, 7.0, 1.4, "Traducción: haz de 4,\nsin repetir 3-gramas", "#f2efe6", tam=6.5)
    for y1, y2 in [(10.1, 9.6), (8.2, 7.6), (5.2, 4.6), (2.2, 1.5)]:
        _flecha(ax, 5.0, y1, 5.0, y2)
    ax.text(8.9, 3.55, "etiqueta\nde destino\nforzada", ha="left", va="center", fontsize=5.8, color="#9a6a1f")
    _flecha(ax, 8.85, 3.55, 8.55, 3.55)
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)


def concordancia(destino: Path) -> None:
    claves = [("1s", "1.ª sg."), ("2s", "2.ª sg."), ("3s", "3.ª sg."), ("1pi", "1.ª pl. incl."), ("1pe", "1.ª pl. excl."), ("2p", "2.ª pl."), ("3p", "3.ª pl."),
              ("presente", "Presente"), ("pasado", "Pasado"), ("futuro", "Futuro")]
    datos = {n: json.loads((RES / "concordancia" / f"{n}.json").read_text(encoding="utf-8"))["acierto"] for n in ("principal_s0", "exp_norm15k_s0")}
    fig, ejes = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=250, sharex=True)
    for ax, d, titulo in zip(ejes, ("es->qu", "qu->es"), ("es→qu", "qu→es")):
        x = range(len(claves))
        ax.bar([i - 0.2 for i in x], [datos["principal_s0"][f"{d}:{k}"] for k, _ in claves], 0.4, label="Principal", color="#b8b8b8")
        ax.bar([i + 0.2 for i in x], [datos["exp_norm15k_s0"][f"{d}:{k}"] for k, _ in claves], 0.4, label="Normalizado 15k", color="#2f6aa3")
        ax.set_title(titulo, fontsize=7.5, pad=2)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Acierto (%)", fontsize=7)
        ax.axvline(6.5, color="#888", lw=0.6, ls=":")
    ejes[0].legend(fontsize=6, loc="upper right", ncol=2)
    ejes[1].set_xticks(list(range(len(claves))))
    ejes[1].set_xticklabels([e for _, e in claves], rotation=50, ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    arquitectura(RES / "fig_arquitectura.png")
    concordancia(RES / "fig_concordancia.png")
    print("figuras generadas")
