"""Resumen y secciones IV-VI. Todas las cifras se leen de results/ (test.json, historial.json, errores.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import contenido as C

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"

# Tareas por integrante: las declara el equipo. No se inventan.
TAREAS: dict[str, str] = {}

PRINCIPAL = ["principal_s0", "principal_s1", "principal_s2"]
DIRS = ["es->qu", "qu->es"]


def _j(*partes: str) -> dict:
    return json.loads(RES.joinpath(*partes).read_text(encoding="utf-8"))


def hay_resultados() -> bool:
    return all((RES / n / "test.json").exists() for n in PRINCIPAL) and (RES / "errores.json").exists()


def _stats() -> dict:
    base = _j("baseline_zeroshot", "test.json")
    prin = [_j(n, "test.json") for n in PRINCIPAL]
    abl = {50: _j("ablacion_datos50_s0", "test.json"), 25: _j("ablacion_datos25_s0", "test.json")}
    ms = lambda d, k: (float(np.mean([p[d][k] for p in prin])), float(np.std([p[d][k] for p in prin], ddof=1)))
    horas = {n: _j(n, "historial.json") for n in PRINCIPAL + ["ablacion_datos50_s0", "ablacion_datos25_s0"]}
    return {"base": base, "prin": prin, "abl": abl, "ms": ms, "horas": horas, "err": _j("errores.json")}


def _n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def _pm(m: float, s: float) -> str:
    return f"{m:.1f} ± {s:.1f}"


def resumen() -> str:
    s = _stats()
    e, q = s["ms"]("es->qu", "chrF++"), s["ms"]("qu->es", "chrF++")
    be, bq = s["ms"]("es->qu", "BLEU"), s["ms"]("qu->es", "BLEU")
    h = sum(v["horas_gpu_entrenamiento"] for v in s["horas"].values())
    return (
        "El quechua ayacuchano tiene pocos textos paralelos con el español, y los conjuntos públicos mezclan duplicados, fuga entre particiones y mucho contenido religioso. "
        "Ajustamos NLLB-200 de 600 millones de parámetros como un solo modelo bidireccional español–quechua. Limpiamos y repartimos sin fuga 122 802 pares públicos, "
        "excluimos el 55 % marcado como religioso y entrenamos con 49 414 pares. "
        f"En una prueba de 2 662 pares sin texto religioso, tres semillas dan un chrF++ de {_pm(*e)} (es→qu) y {_pm(*q)} (qu→es), frente a {s['base']['es->qu']['chrF++']} y {s['base']['qu->es']['chrF++']} "
        f"del mismo modelo sin ajustar, y un BLEU de {be[0]:.1f} y {bq[0]:.1f}. "
        f"Entrenar con el 25 %, el 50 % y el 100 % de los datos sube el chrF++ es→qu de {s['abl'][25]['es->qu']['chrF++']} a {s['abl'][50]['es->qu']['chrF++']} y a {e[0]:.1f}, sin señal de saturación. "
        f"El análisis de errores muestra que el {100 * s['err']['n_ruido'] / s['err']['n_pares']:.0f} % de la prueba son encabezados y fechas, que hay pares desalineados y que "
        f"el {100 * s['err']['por_direccion']['es->qu']['repeticiones'] / s['err']['n_pares']:.1f} % de las salidas es→qu cae en repeticiones. "
        f"Los cinco entrenamientos suman {h:.1f} horas en una GPU de 48 GB. El modelo está desplegado como servicio web y aplicación Android. "
        "La prueba proviene de la misma fuente que el entrenamiento y no se validó con hablantes, así que los números no miden el uso real."
    )


def bloques() -> list[tuple]:
    if not hay_resultados():
        return [("sec", "Resultados"), ("pend", "faltan results/*/test.json o results/errores.json")]
    s = _stats()
    ms, base, abl, err = s["ms"], s["base"], s["abl"], s["err"]
    e_c, q_c = ms("es->qu", "chrF++"), ms("qu->es", "chrF++")
    e_b, q_b = ms("es->qu", "BLEU"), ms("qu->es", "BLEU")
    ed, qd = err["por_direccion"]["es->qu"], err["por_direccion"]["qu->es"]
    horas = {k: v["horas_gpu_entrenamiento"] for k, v in s["horas"].items()}
    h_tot = sum(horas.values())
    h_test = sum(v["horas_gpu_test"] for v in s["horas"].values())
    pico = max(v["pico_vram_gb"] for v in s["horas"].values())
    pct_ruido = 100 * err["n_ruido"] / err["n_pares"]

    filas = [
        ["Sin ajustar (línea base)", f"{base['es->qu']['chrF++']}", f"{base['es->qu']['BLEU']}", f"{base['qu->es']['chrF++']}", f"{base['qu->es']['BLEU']}"],
        ["Ajustado, 25 % de los datos", f"{abl[25]['es->qu']['chrF++']}", f"{abl[25]['es->qu']['BLEU']}", f"{abl[25]['qu->es']['chrF++']}", f"{abl[25]['qu->es']['BLEU']}"],
        ["Ajustado, 50 % de los datos", f"{abl[50]['es->qu']['chrF++']}", f"{abl[50]['es->qu']['BLEU']}", f"{abl[50]['qu->es']['chrF++']}", f"{abl[50]['qu->es']['BLEU']}"],
        ["Ajustado, 100 % (3 semillas)", _pm(*e_c), _pm(*e_b), _pm(*q_c), _pm(*q_b)],
    ]
    suf_filas = [[f"-{k}", str(v["n"]), f"{v['recall']} %"] for k, v in ed["sufijos"].items() if k in ("pa", "pi", "kuna", "paq", "wan", "manta")]
    longitudes = ["1-5", "6-10", "11-20", "21-40"]
    lon_filas = [[f"{b} palabras", f"{ed['por_longitud'][b]['chrf']}", f"{qd['por_longitud'][b]['chrf']}", f"{ed['por_longitud'][b]['n']}"] for b in longitudes]

    return [
        ("resumen", resumen()),
        ("sec", "Resultados"),
        ("sub", "Comparación con la línea base"),
        ("p", f"La Tabla II compara el modelo ajustado con NLLB-200 sin ajustar sobre los mismos {_n(err['n_pares'])} pares de prueba. "
              f"El ajuste mejora el chrF++ en {e_c[0] - base['es->qu']['chrF++']:.1f} puntos en es→qu y en {q_c[0] - base['qu->es']['chrF++']:.1f} en qu→es, "
              f"y el BLEU de {base['es->qu']['BLEU']} a {e_b[0]:.1f} y de {base['qu->es']['BLEU']} a {q_b[0]:.1f}. "
              f"La desviación entre las tres semillas es de {max(e_c[1], q_c[1]):.1f} puntos de chrF++ o menos, mucho menor que la mejora. "
              "Las dos direcciones no son comparables entre sí: el idioma de destino cambia la métrica."),
        ("table", "tab:resultados", "Resultados en la prueba (chrF++ y BLEU; media ± d. e. de 3 semillas)",
         ["Sistema", "chrF++ es→qu", "BLEU es→qu", "chrF++ qu→es", "BLEU qu→es"], filas),
        ("sub", "Ablación del tamaño del entrenamiento"),
        ("p", f"Con el 25 %, el 50 % y el 100 % de los 49 414 pares, el chrF++ es→qu pasa de {abl[25]['es->qu']['chrF++']} a {abl[50]['es->qu']['chrF++']} y a {e_c[0]:.1f}, "
              f"y el qu→es de {abl[25]['qu->es']['chrF++']} a {abl[50]['qu->es']['chrF++']} y a {q_c[0]:.1f} (Tabla II). "
              "Cada duplicación de los datos aporta entre 0.7 y 2.1 puntos (la de 0.7 está cerca de la variación entre semillas), y la curva no se aplana en este rango, de modo que más datos de Ayacucho probablemente ayudarían. "
              "Cada punto de la ablación tiene una sola semilla; las diferencias de 1 punto o más superan la desviación observada (0.4) pero las menores no deben interpretarse. "
              "La Figura 1 muestra la pérdida de entrenamiento y el chrF++ de validación por época."),
        ("fig", RES / "fig_curvas.png", "Pérdida de entrenamiento y chrF++ medio de validación por época.", "fig:curvas"),
        ("sub", "Interpretabilidad y análisis de errores"),
        ("p", "Analizamos las predicciones de la semilla 0 sobre toda la prueba. La Tabla III separa el chrF++ por longitud de la referencia. "
              "En es→qu el chrF++ baja poco, de 40.2 en frases de 1 a 5 palabras a 37.3 en las de 11 a 20; en qu→es baja de 38.1 a 32.7 en esos mismos tramos."),
        ("table", "tab:longitud", "chrF++ por longitud de la referencia (semilla 0)", ["Longitud", "es→qu", "qu→es", "n por dirección"], lon_filas),
        ("p", "Para ver dónde falla la morfología medimos, en es→qu, cuántas palabras de la referencia con un sufijo dado aparecen idénticas en la salida (Tabla IV). "
              "Es una medida estricta: exige la palabra completa, y la traducción de referencia puede ser libre, así que un valor bajo no prueba por sí solo un error. "
              "Aun así, los sufijos que marcan relaciones (-manta, -wan, -paq) se reproducen mucho menos que -pa, lo que sugiere que la flexión es la parte más difícil."),
        ("table", "tab:sufijos", "Recall de sufijos en es→qu (palabra completa idéntica a la referencia)", ["Sufijo", "Apariciones", "Recall"], suf_filas),
        ("p", f"Revisamos además los casos de menor puntaje. Tres hallazgos: (i) {err['n_ruido']} de {_n(err['n_pares'])} pares de prueba ({pct_ruido:.1f} %) son encabezados, fechas o números de página "
              f"(por ejemplo «PÁGINA 13»); sin ellos el chrF++ es {ed['chrf_sin_ruido']} (es→qu) y {qd['chrf_sin_ruido']} (qu→es), es decir, 0.3 puntos menos, "
              f"así que no explican el promedio. (ii) Entre los peores casos restantes hay pares desalineados, con el quechua de una oración junto a la traducción de otra; no medimos su proporción. "
              f"(iii) Hay fallas reales del modelo: {ed['repeticiones']} salidas es→qu ({100 * ed['repeticiones'] / err['n_pares']:.2f} %) y {qd['repeticiones']} qu→es repiten la misma palabra o tres palabras "
              "varias veces («sinchi sinchi sinchi…»)."),
        ("sub", "Costo computacional"),
        ("p", f"Los cinco entrenamientos (tres semillas y dos ablaciones) suman {h_tot:.2f} horas de GPU, a las que se añaden {h_test:.2f} h de evaluación, en una NVIDIA RTX PRO 5000 Blackwell de 48 GB con bf16. "
              f"Una corrida con todos los datos dura entre {min(horas[n] for n in PRINCIPAL):.2f} y {max(horas[n] for n in PRINCIPAL):.2f} h, con un pico de {pico} GB de memoria de vídeo y lotes de 32. "
              "En el servidor, que solo tiene CPU, traducir una oración tarda unos 6 s (medido con tres peticiones). "
              "No ejecutamos el entrenamiento en una T4: su memoria de 16 GB queda por debajo de nuestro pico, por lo que habría que reducir el lote y usar fp16, algo que el script admite pero que no probamos. "
              "No estimamos la huella de carbono."),

        ("sec", "Discusión"),
        ("p", "El ajuste fino funciona porque NLLB-200 ya conoce la etiqueta quy_Latn [1]: con unas decenas de miles de pares adapta la ortografía y el vocabulario cotidiano, "
              "y el modelo sin ajustar ya alcanza 26–28 de chrF++. La ablación indica que el límite es el volumen de datos y no la capacidad del modelo, porque la curva sigue subiendo al pasar del 50 % al 100 %."),
        ("p", "Hay límites que acotan lo que decimos. Primero, la prueba sale de la misma fuente que el entrenamiento: el reparto sin fuga evita oraciones repetidas, pero no cambia el estilo, "
              "así que el resultado puede ser optimista para texto de otro origen. Segundo, el filtro de contenido religioso es una lista de palabras: es conservador, por lo que también descarta oraciones generales, "
              "y puede dejar pasar texto religioso sin esas palabras. Tercero, la prueba contiene pares desalineados que penalizan a todos los sistemas por igual y que no auditamos con un hablante. "
              "Cuarto, no comparamos con otros sistemas de forma directa: el chrF de 34.6 del ganador de 2021 [3], [4] viene de otra prueba. "
              "Quinto, ninguna persona hablante de quechua evaluó las traducciones, y en una lengua aglutinante una cifra de chrF++ no distingue un error de flexión de uno de sentido."),
        ("p", "Hay también un límite de variedad. Una prueba pequeña, con 24 fichas de cartillas escolares en quechua central, dio un chrF++ inferior al de la prueba de Ayacucho en ambas direcciones, "
              "lo que indica que el rendimiento cae fuera de la variedad de entrenamiento; no la incluimos entre los resultados porque la fuente tiene derechos reservados y no podemos redistribuirla, ni es una muestra grande."),
        ("p", "Sobre impacto y uso responsable: un traductor con este nivel de calidad puede ayudar a leer o redactar mensajes sencillos, pero puede inventar contenido (las repeticiones y los pares desalineados lo muestran) "
              "y no debe usarse para textos médicos, legales o administrativos sin revisión humana. El conjunto de datos procede en parte de publicaciones con derechos de autor y no declara licencia, "
              "y NLLB-200 se distribuye con licencia CC-BY-NC, por lo que el modelo es solo para uso no comercial. El servicio y la aplicación son prototipos y no se probaron con la comunidad hablante."),

        ("sec", "Conclusiones"),
        ("p", f"Un solo NLLB-200 ajustado con 49 414 pares no religiosos de quechua ayacuchano traduce en ambos sentidos con {e_c[0]:.1f} (es→qu) y {q_c[0]:.1f} (qu→es) de chrF++, "
              f"unos {e_c[0] - base['es->qu']['chrF++']:.0f} y {q_c[0] - base['qu->es']['chrF++']:.0f} puntos más que sin ajustar, con resultados estables entre semillas y mejoras que crecen con el tamaño del entrenamiento. "
              "La limpieza y el reparto sin fuga fueron necesarios para poder medir algo."),
        ("p", "Como trabajo futuro: (1) evaluar en pruebas externas, como FLORES-200 con quy_Latn y los conjuntos de AmericasNLP, que provienen de otras fuentes; "
              "(2) revisar con un hablante una muestra de la prueba para medir los pares desalineados; (3) reducir las repeticiones con restricciones de decodificación y medir el efecto; "
              "(4) entrenar con adaptadores (LoRA) para que quepa en una T4; y (5) añadir más texto de Ayacucho de otras fuentes, ya que la ablación sugiere que el modelo sigue mejorando con más datos."),
    ]


def contribuciones() -> list[tuple]:
    filas = [(n, TAREAS.get(n, "(por completar por el equipo)")) for n, _ in C.AUTORES]
    return [("contrib", filas), ("disp", C.TEXTO_DISPONIBILIDAD)]
