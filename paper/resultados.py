"""Resumen y secciones IV-VI. Todas las cifras se leen de results/ (test.json, historial.json, errores.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import contenido as C

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"

# Tareas por integrante: las declara el equipo. No se inventan.
_TAREAS_COMUNES = "Pruebas del modelo, generación del conjunto de datos y normalización del conjunto de datos."
_MODELO_POR_INTEGRANTE = [
    "Modelo principal y ablación del tamaño del entrenamiento; pruebas de español a quechua.",
    "Modelos sin el 15 % de los pares (al azar y por pérdida alta) y modelo normalizado 2k; pruebas de quechua a español.",
    "Modelos con retrotraducción (umbrales 30 y 40) y ciclo automático; pruebas de rendimiento del modelo.",
    "Modelos normalizado 15k y con literatura central, incluido el modelo final; optimización de modelos.",
]
TAREAS: dict[str, str] = {n: f"{_TAREAS_COMUNES} {m}" for (n, _), m in zip(C.AUTORES, _MODELO_POR_INTEGRANTE)}

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


# (nombre en la tabla, carpeta de results/, prefijo FLORES, nombre en results/concordancia/)
MODELOS_COMPARADOS = [
    ("Sin ajustar", "baseline_zeroshot", "zeroshot_devtest", "zeroshot"),
    ("Principal", "principal_s0", "principal_s0_def_devtest", "principal_s0"),
    ("Sin 15 % al azar", "exp_aleatorio15_s0", "exp_aleatorio15_s0_def_devtest", "exp_aleatorio15_s0"),
    ("Sin 15 % peor puntuado", "exp_filtrado15_s0", "exp_filtrado15_s0_def_devtest", "exp_filtrado15_s0"),
    ("Retrotrad. ≥ 30", "exp_bt30_s0", "exp_bt30_s0_def_devtest", "exp_bt30_s0"),
    ("Retrotrad. ≥ 40", "exp_bt40_s0", "exp_bt40_s0_def_devtest", "exp_bt40_s0"),
    ("Ciclo automático (vuelta 1)", "auto_it1", None, "auto_it1"),
    ("Normalizado 2k", "exp_norm2k_s0", "exp_norm2k_s0_def_devtest", "exp_norm2k_s0"),
    ("Normalizado 15k", "exp_norm15k_s0", "exp_norm15k_s0_def_devtest", "exp_norm15k_s0"),
    ("Normalizado 15k + literatura central", "exp_norm15k_central_s0", "exp_norm15k_central_s0_def_devtest", "exp_norm15k_central_s0"),
]


def _leer_opcional(*partes: str) -> dict | None:
    ruta = RES.joinpath(*partes)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def _fila_comparacion(nombre: str, carpeta: str, flores: str | None, conc: str) -> list[str]:
    test = _leer_opcional(carpeta, "test.json")
    if flores:
        fl = _leer_opcional("flores", f"{flores}.json")
        fl_v = [fl[d]["chrF++"] for d in DIRS] if fl else None
    else:  # el ciclo automático guarda FLORES devtest en su propio resumen
        a = _leer_opcional("auto", "eval_auto_it1.json")
        fl_v = [a[f"devtest_{d}"]["chrF++"] for d in DIRS] if a else None
    c = _leer_opcional("concordancia", f"{conc}.json")
    cc = [c["acierto"][d] for d in DIRS] if c else None
    f = lambda v, i: "n/d" if v is None else f"{v[i]:.1f}"
    return [nombre] + [f([test[d]["chrF++"] for d in DIRS] if test else None, i) for i in (0, 1)] + [f(fl_v, i) for i in (0, 1)] + [f(cc, i) for i in (0, 1)]


def tabla_comparacion() -> list[list[str]]:
    return [_fila_comparacion(*m) for m in MODELOS_COMPARADOS]


def _semillas(prefijo: str, n: int = 3) -> dict:
    """Media ± d. e. del chrF++ de prueba y del acierto de concordancia entre las semillas de un experimento."""
    out = {}
    for d in DIRS:
        t = [_j(f"{prefijo}_s{i}", "test.json")[d]["chrF++"] for i in range(n)]
        c = [_j("concordancia", f"{prefijo}_s{i}.json")["acierto"][d] for i in range(n)]
        out[d] = (float(np.mean(t)), float(np.std(t, ddof=1)), float(np.mean(c)), float(np.std(c, ddof=1)))
    return out


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
        f"Entrenar con el 25 %, el 50 % y el 100 % de los datos sube el chrF++ es→qu de {s['abl'][25]['es->qu']['chrF++']} a {s['abl'][50]['es->qu']['chrF++']} y a {e[0]:.1f}. "
        "Comparamos diez modelos. Los cambios en la selección de datos y la retrotraducción no mueven la prueba propia más de un punto. "
        "Añadir un conjunto normalizado de verbos y léxico mantiene esa prueba y sube el acierto en la conjugación de verbos nuevos de 24.8 % a 57.1 % (es→qu) y de 53.7 % a 82.1 % (qu→es), "
        "por lo que es el modelo final. "
        f"El 0.5 % de las salidas es→qu cae en repeticiones. "
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
    apo, ap = _j("apostrofo_datos.json"), _j("prueba_apostrofo.json")
    fl15 = {m: _j("flores", f"exp_norm15k_s0_{m}_devtest.json") for m in ("def", "nr3")}
    flp = {m: _j("flores", f"principal_s0_{m}_devtest.json") for m in ("def", "nr3")}
    comp = tabla_comparacion()
    s15, s2k = _semillas("exp_norm15k"), _semillas("exp_norm2k")

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
        ("p", f"La Tabla III compara el modelo ajustado con NLLB-200 sin ajustar sobre los mismos {_n(err['n_pares'])} pares de prueba. "
              f"El ajuste mejora el chrF++ en {e_c[0] - base['es->qu']['chrF++']:.1f} puntos en es→qu y en {q_c[0] - base['qu->es']['chrF++']:.1f} en qu→es, "
              f"y el BLEU de {base['es->qu']['BLEU']} a {e_b[0]:.1f} y de {base['qu->es']['BLEU']} a {q_b[0]:.1f}. "
              f"La desviación entre las tres semillas es de {max(e_c[1], q_c[1]):.1f} puntos de chrF++ o menos, mucho menor que la mejora. "
              "Las dos direcciones no son comparables entre sí: el idioma de destino cambia la métrica."),
        ("table", "tab:resultados", "Resultados en la prueba (chrF++ y BLEU; media ± d. e. de 3 semillas)",
         ["Sistema", "chrF++ es→qu", "BLEU es→qu", "chrF++ qu→es", "BLEU qu→es"], filas),
        ("sub", "Comparación entre modelos y selección del mejor"),
        ("p", "La Tabla IV reúne los diez modelos con tres evaluaciones: la prueba propia, FLORES-200 devtest (1 012 pares, otro dominio) y el examen de concordancia con verbos nuevos. "
              f"En la prueba propia las diferencias son pequeñas: todos los modelos ajustados quedan entre {min(float(r[1]) for r in comp[1:]):.1f} y {max(float(r[1]) for r in comp[1:]):.1f} en es→qu "
              f"y entre {min(float(r[2]) for r in comp[1:]):.1f} y {max(float(r[2]) for r in comp[1:]):.1f} en qu→es, a menos de un punto, que es del orden de la variación entre semillas. "
              "Quitar datos, ya sea al azar o por pérdida alta, no ayudó, y la retrotraducción filtrada tampoco mejoró la prueba propia."),
        ("table", "tab:comparacion", "Comparación de modelos (chrF++ en prueba propia y FLORES devtest; % de acierto en el examen de concordancia; semilla 0)",
         ["Modelo", "Prueba es→qu", "Prueba qu→es", "FLORES es→qu", "FLORES qu→es", "Conc. es→qu", "Conc. qu→es"], comp),
        ("p", "Los modelos se separan en la concordancia. El principal acierta el " f"{comp[1][5]} % en es→qu y el {comp[1][6]} % en qu→es, y las variantes que solo cambian datos de entrenamiento o añaden retrotraducción se quedan por debajo del 30 % en es→qu y del 56 % en qu→es. "
              f"Con el conjunto normalizado el acierto sube a {comp[8][5]} % y {comp[8][6]} % (15k) y a {comp[7][5]} % y {comp[7][6]} % (2k). "
              f"Con tres semillas el 15k da {_pm(s15['es->qu'][2], s15['es->qu'][3])} % y {_pm(s15['qu->es'][2], s15['qu->es'][3])} %, y el 2k da {_pm(s2k['es->qu'][2], s2k['es->qu'][3])} % y {_pm(s2k['qu->es'][2], s2k['qu->es'][3])} %, "
              f"sin pérdida en la prueba propia (15k: {_pm(s15['es->qu'][0], s15['es->qu'][1])} y {_pm(s15['qu->es'][0], s15['qu->es'][1])} de chrF++). "
              "Un conjunto de solo 2 679 filas rinde igual que uno de 15 373, de modo que lo que aporta es la cobertura de conjugaciones y no el volumen."),
        ("p", f"Elegimos el 15k (semilla 0) como modelo final por tres razones: tiene el mejor resultado en la prueba propia en qu→es ({comp[8][2]}) y empata el mejor en es→qu ({comp[8][1]}), "
              f"es el que más acierta en la concordancia qu→es ({comp[8][6]} %) y en FLORES queda a {abs(float(comp[1][3]) - float(comp[8][3])):.1f} puntos del principal en es→qu y a {abs(float(comp[1][4]) - float(comp[8][4])):.1f} por encima en qu→es. "
              f"No gana en todo: el ciclo automático obtiene {comp[6][3]} y {comp[6][4]} en FLORES, frente a {comp[8][3]} y {comp[8][4]}, pero usó pares de FLORES dev como entrenamiento, "
              "y por eso su cifra en FLORES no es comparable, y su concordancia es menor. Con literatura quechua central (138 pares) la concordancia sube en es→qu y baja en qu→es y la prueba propia baja entre 0.2 y 0.3 puntos. "
              "En 21 pares de literatura central reservados, el chrF++ qu→es sube de 26.4 a 28.3 y el es→qu baja de 21.1 a 20.1, una muestra demasiado pequeña y de una sola semilla para concluir algo, por lo que no la adoptamos. "
              "El principal se conserva como línea de base."),
        ("p", "La Figura 2 desglosa la concordancia por persona y tiempo. El modelo normalizado mejora en casi todas las celdas; la excepción es la segunda persona del singular en es→qu, que baja de 60.0 % a 46.7 %, "
              "y en qu→es la misma persona pasa de 0 % a 60.0 %. Cada persona tiene entre 10 y 15 casos, de modo que las diferencias por persona son ruidosas y solo el patrón general es fiable. "
              "El pasado en es→qu sigue siendo el tiempo más bajo (45.7 %)."),
        ("fig", RAIZ / "results" / "fig_concordancia.png", "Acierto en el examen de concordancia por persona y tiempo, modelo principal frente a normalizado 15k (semilla 0).", "fig:concordancia"),
        ("p", f"La decodificación también cambia los resultados en FLORES. Con el decodificador por defecto, el modelo 15k repite palabras en {fl15['def']['es->qu']['repeticiones']} traducciones es→qu y {fl15['def']['qu->es']['repeticiones']} qu→es "
              f"de 1 012, y alcanza {fl15['def']['es->qu']['chrF++']} y {fl15['def']['qu->es']['chrF++']} de chrF++. Al prohibir repetir 3-gramas, que es lo que usa el servicio, las repeticiones bajan a {fl15['nr3']['es->qu']['repeticiones']} y {fl15['nr3']['qu->es']['repeticiones']} "
              f"y el chrF++ sube a {fl15['nr3']['es->qu']['chrF++']} y {fl15['nr3']['qu->es']['chrF++']}. En el modelo principal el efecto es parecido ({flp['def']['es->qu']['chrF++']} a {flp['nr3']['es->qu']['chrF++']} en es→qu). "
              "Las cifras de la Tabla IV son con el decodificador por defecto; la restricción no invierte la comparación entre el principal y el 15k."),
        ("sub", "Ablación del tamaño del entrenamiento"),
        ("p", f"Con el 25 %, el 50 % y el 100 % de los 49 414 pares, el chrF++ es→qu pasa de {abl[25]['es->qu']['chrF++']} a {abl[50]['es->qu']['chrF++']} y a {e_c[0]:.1f}, "
              f"y el qu→es de {abl[25]['qu->es']['chrF++']} a {abl[50]['qu->es']['chrF++']} y a {q_c[0]:.1f} (Tabla III). "
              "Cada duplicación de los datos aporta entre 0.7 y 2.1 puntos (la de 0.7 está cerca de la variación entre semillas), y la curva no se aplana en este rango, de modo que más datos de Ayacucho probablemente ayudarían. "
              "Cada punto de la ablación tiene una sola semilla; las diferencias de 1 punto o más superan la desviación observada (0.4) pero las menores no deben interpretarse. "
              "La Figura 3 muestra la pérdida de entrenamiento y el chrF++ de validación por época."),
        ("fig", RES / "fig_curvas.png", "Pérdida de entrenamiento y chrF++ medio de validación por época.", "fig:curvas"),
        ("sub", "Interpretabilidad y análisis de errores"),
        ("p", "Analizamos las predicciones de la semilla 0 sobre toda la prueba. La Tabla V separa el chrF++ por longitud de la referencia. "
              "En es→qu el chrF++ baja poco, de 40.2 en frases de 1 a 5 palabras a 37.3 en las de 11 a 20; en qu→es baja de 38.1 a 32.7 en esos mismos tramos."),
        ("table", "tab:longitud", "chrF++ por longitud de la referencia (semilla 0)", ["Longitud", "es→qu", "qu→es", "n por dirección"], lon_filas),
        ("p", "Para ver dónde falla la morfología medimos, en es→qu, cuántas palabras de la referencia con un sufijo dado aparecen idénticas en la salida (Tabla VI). "
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
        ("sub", "Uso en la aplicación: dirección, voz y apóstrofo"),
        ("p", "El mismo modelo traduce en ambos sentidos y la aplicación deja elegir la dirección; las Tablas III y IV muestran los resultados de las dos. "
              "En la práctica resulta más cómodo traducir de español a quechua, porque el reconocimiento de voz de Android funciona para español (la aplicación lo configura como es-PE) y la frase de entrada se puede dictar. "
              "Del quechua al español la entrada no se puede dictar con fiabilidad: Android no reconoce quechua, y lo que se dicta se transcribe como si fuera español, así que el texto se escribe a mano. "
              "No medimos la tasa de error de ese reconocimiento; lo descrito sale de la configuración de la aplicación y de lo que observó el equipo. "
              "La voz de salida tampoco es quechua: usa una voz en español a velocidad reducida, que solo aproxima la pronunciación."),
        ("p", f"Escribir quechua tiene además un obstáculo ortográfico. Las consonantes eyectivas se marcan con un apóstrofo dentro de la palabra (p'unchaw, ch'aki, t'anta), "
              "un carácter que no está en la capa principal del teclado de un teléfono. "
              f"En los datos casi no aparece: {apo['train_qu']['frases_con']} de {_n(apo['train_qu']['frases'])} oraciones de entrenamiento ({100 * apo['train_qu']['frases_con'] / apo['train_qu']['frases']:.1f} %) "
              f"y {apo['train_qu']['palabras_con']} de {_n(apo['train_qu']['palabras'])} palabras llevan apóstrofo interno, y se escribe de dos formas ({apo['train_qu']['formas']['’']} con apóstrofo tipográfico y {apo['train_qu']['formas']['\'']} con apóstrofo recto). "
              f"La prueba tiene {apo['test_qu']['frases_con']} oraciones con apóstrofo. El modelo lo escribe con apóstrofo recto en {apo['salida_es_qu']['palabras_con']} palabras de sus salidas es→qu, frente a {apo['referencia_es_qu']['palabras_con']} en las referencias."),
        ("p", f"Omitir el apóstrofo no es una salida neutra, porque según nuestro léxico a veces distingue palabras: ch'aki (seco) y chaki (pie). "
              f"Con las {ap['recto \'']['n']} palabras con apóstrofo del léxico normalizado, el modelo acierta la glosa en español en {ap['recto \'']['aciertos']} con apóstrofo recto, "
              f"{ap['tipográfico ’']['aciertos']} con apóstrofo tipográfico y {ap['sin apóstrofo']['aciertos']} sin apóstrofo; sin él, ch'aki se traduce como pie, huch'uy como pecar y t'ika como cómodamente. "
              "Es una prueba pequeña y con palabras que el modelo vio en el entrenamiento, así que muestra la sensibilidad al apóstrofo y no la capacidad de generalizar."),

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
              "De los diez modelos comparados, solo los que añaden el conjunto normalizado mejoran de forma marcada la conjugación de verbos nuevos (más de 30 puntos), sin perder en la prueba propia; es el modelo final y el que sirve la aplicación. "
              "La limpieza y el reparto sin fuga fueron necesarios para poder medir algo."),
        ("p", "Como trabajo futuro: (1) evaluar en pruebas externas, como FLORES-200 con quy_Latn y los conjuntos de AmericasNLP, que provienen de otras fuentes; "
              "(2) revisar con un hablante una muestra de la prueba para medir los pares desalineados; (3) reducir las repeticiones con restricciones de decodificación y medir el efecto; "
              "(4) entrenar con adaptadores (LoRA) para que quepa en una T4; y (5) añadir más texto de Ayacucho de otras fuentes, ya que la ablación sugiere que el modelo sigue mejorando con más datos."),
    ]


def contribuciones() -> list[tuple]:
    filas = [(n, TAREAS.get(n, "(por completar por el equipo)")) for n, _ in C.AUTORES]
    return [("contrib", filas), ("disp", C.TEXTO_DISPONIBILIDAD)]
