"""Secciones que dependen de los resultados (IV-VI), contribuciones y disponibilidad.

Mientras faltan results/*/test.json, devuelve bloques pendientes explicitos (borrador).
"""
from __future__ import annotations

from pathlib import Path

import contenido as C

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"

# Tareas por integrante: las declara el equipo. No se inventan.
TAREAS: dict[str, str] = {}


def hay_resultados() -> bool:
    return (RES / "principal_s0" / "test.json").exists()


def bloques() -> list[tuple]:
    if not hay_resultados():
        return [
            ("sec", "Resultados"),
            ("pend", "los experimentos siguen en ejecución; esta sección se completa con las tablas y figuras finales"),
            ("sec", "Discusión"),
            ("pend", "se redacta con los resultados"),
            ("sec", "Conclusiones"),
            ("pend", "se redacta con los resultados"),
        ]
    raise NotImplementedError("completar con los resultados reales")


def contribuciones() -> list[tuple]:
    filas = [(n, TAREAS.get(n, "(por completar por el equipo)")) for n, _ in C.AUTORES]
    return [("contrib", filas), ("disp", C.TEXTO_DISPONIBILIDAD)]
