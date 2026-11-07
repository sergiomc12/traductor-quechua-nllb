"""Regenera data/dataset_quechua_espanol_20_millones_palabras.csv con gramatica correcta.

El original (1 990 368 filas) tenia solo 68 824 pares unicos, todos generados por plantillas con
dos defectos: (1) el verbo quechua llevaba siempre la terminacion de "yo" (-ni/-rqani/-saq)
sin importar el sujeto, y (2) el espanol tenia formas inexistentes ("comero", "hacerre", "el/la").

Este script NO parchea filas: reconstruye cada una desde un lexico pequeno y reglas explicitas,
asi que cada frase concuerda en persona y tiempo. Ademas emite la variante ortografica
Ayacucho de las palabras con variante verificada (columna `dialecto`).

Uso:  python scripts/normalizar_dataset_quechua.py [--salida RUTA.csv]
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SALIDA = ROOT / "data" / "dataset_quechua_espanol_normalizado.csv"
LEXICO = ROOT / "data" / "lexico_normalizado.csv"

PERSONAS = ["1s", "2s", "3s", "1pi", "1pe", "2p", "3p"]
TIEMPOS = ["presente", "pasado", "futuro"]
DIALECTOS = ["base", "ayacucho", "alterna"]

PRON_QU = {"1s": "ñuqa", "2s": "qam", "3s": "pay", "1pi": "ñuqanchik", "1pe": "ñuqayku", "2p": "qamkuna", "3p": "paykuna"}
PRON_ES = {"1s": "yo", "2s": "tú", "3s": "él/ella", "1pi": "nosotros (inclusivo)",
           "1pe": "nosotros (exclusivo)", "2p": "ustedes", "3p": "ellos/ellas"}
IDX_ES = {"1s": 0, "2s": 1, "3s": 2, "1pi": 3, "1pe": 3, "2p": 4, "3p": 4}

# Terminaciones verbales (paradigma Cusco/Chanka, el del dataset original). Las personas 2s y 2p
# del futuro coinciden con las del presente; por eso el futuro de 2s/2p se omite (ver generar()).
SUFIJOS = {
    "presente": {"1s": "ni", "2s": "nki", "3s": "n", "1pi": "nchik", "1pe": "niku", "2p": "nkichik", "3p": "nku"},
    "pasado": {"1s": "rqani", "2s": "rqanki", "3s": "rqan", "1pi": "rqanchik", "1pe": "rqaniku",
               "2p": "rqankichik", "3p": "rqanku"},
    "futuro": {"1s": "saq", "2s": "nki", "3s": "nqa", "1pi": "sunchik", "1pe": "saqku",
               "2p": "nkichik", "3p": "nqaku"},
}

# raiz quechua -> (infinitivo espanol, transitivo)
VERBOS = {
    "puri": ("caminar", False), "miku": ("comer", True), "upya": ("beber", True),
    "puñu": ("dormir", False), "ruwa": ("hacer", True), "rima": ("hablar", False),
    "yacha": ("saber", False), "qawa": ("mirar", True), "taki": ("cantar", False),
    "tusu": ("bailar", False),
}

# Formas irregulares: [1s, 2s, 3s, 1p, 3p/ustedes]
IRREGULARES = {
    "hacer": {"presente": ["hago", "haces", "hace", "hacemos", "hacen"],
              "pasado": ["hice", "hiciste", "hizo", "hicimos", "hicieron"],
              "futuro": ["haré", "harás", "hará", "haremos", "harán"]},
    "saber": {"presente": ["sé", "sabes", "sabe", "sabemos", "saben"],
              "pasado": ["supe", "supiste", "supo", "supimos", "supieron"],
              "futuro": ["sabré", "sabrás", "sabrá", "sabremos", "sabrán"]},
    "dormir": {"presente": ["duermo", "duermes", "duerme", "dormimos", "duermen"],
               "pasado": ["dormí", "dormiste", "durmió", "dormimos", "durmieron"],
               "futuro": ["dormiré", "dormirás", "dormirá", "dormiremos", "dormirán"]},
}
_REGULAR = {
    "ar": {"presente": ["o", "as", "a", "amos", "an"], "pasado": ["é", "aste", "ó", "amos", "aron"]},
    "er": {"presente": ["o", "es", "e", "emos", "en"], "pasado": ["í", "iste", "ió", "imos", "ieron"]},
    "ir": {"presente": ["o", "es", "e", "imos", "en"], "pasado": ["í", "iste", "ió", "imos", "ieron"]},
}
_FUTURO = ["é", "ás", "á", "emos", "án"]

# Sustantivos: raiz -> (espanol, genero, articulo especial)
SUSTANTIVOS = {
    "allpa": ("tierra", "f"), "allqu": ("perro", "m"), "alpaka": ("alpaca", "f"), "apu": ("espíritu", "m"),
    "chaki": ("pie", "m"), "inti": ("sol", "m"), "killa": ("luna", "f"), "kino'a": ("quinua", "f"),
    "kuntur": ("cóndor", "m"), "llama": ("llama", "f"), "llaqta": ("pueblo", "m"), "maki": ("mano", "f"),
    "mama": ("madre", "f"), "mayu": ("río", "m"), "mikhuna": ("comida", "f"), "misi": ("gato", "m"),
    "nina": ("fuego", "m"), "orqo": ("cerro", "m"), "p'unchaw": ("día", "m"), "pacha": ("mundo", "m"),
    "pana": ("hermana", "f"), "papa": ("papa", "f"), "punku": ("puerta", "f"), "qhari": ("hombre", "m"),
    "runa": ("persona", "f"), "sach'a": ("árbol", "m"), "sara": ("maíz", "m"), "simi": ("boca", "f"),
    "sonqo": ("corazón", "m"), "t'anta": ("pan", "m"), "t'ika": ("flor", "f"), "taruka": ("venado", "m"),
    "tayta": ("padre", "m"), "uma": ("cabeza", "f"), "warmi": ("mujer", "f"), "wasi": ("casa", "f"),
    "wawa": ("hijo", "m"), "wawqi": ("hermano", "m"), "wayra": ("viento", "m"),
    "yachaywasi": ("escuela", "f"), "yaku": ("agua", "f"), "ñawi": ("ojo", "m"),
}
ARTICULO_ESPECIAL = {"yaku": "el"}  # "el agua": a tonica
# 'kunan' (hoy) y 'paqarin' (manana) son adverbios de tiempo, no admiten adjetivo ni caso.

ADJETIVOS = {
    "hatun": ("grande", "grande"), "huch'uy": ("pequeño", "pequeña"), "mana allin": ("malo", "mala"),
    "sumaq": ("hermoso", "hermosa"), "musuq": ("nuevo", "nueva"), "mauk'a": ("viejo", "vieja"),
    "puka": ("rojo", "roja"), "anqas": ("azul", "azul"), "q'omer": ("verde", "verde"),
    "yuraq": ("blanco", "blanca"), "yana": ("negro", "negra"), "sinchi": ("fuerte", "fuerte"),
    "chuya": ("limpio", "limpia"), "ch'aki": ("seco", "seca"), "hoq'o": ("mojado", "mojada"),
}
_LIQUIDOS_Y_ASTROS = {"yaku", "nina", "inti", "killa", "wayra", "pacha", "p'unchaw", "apu"}
ADJ_BLOQUEO = {"ch'aki": _LIQUIDOS_Y_ASTROS, "hoq'o": _LIQUIDOS_Y_ASTROS | {"mayu"}}

CASOS = {"pi": "en", "manta": "desde", "man": "hacia", "wan": "con", "paq": "para", "rayku": "por", "kama": "hasta"}

# --- Revision semantica (reglas simples del autor; la decision final es de un hablante) -----------
CLASE = {}
for _clase, _lista in {
    "persona": ["runa", "mama", "tayta", "pana", "wawqi", "wawa", "qhari", "warmi"],
    "cuerpo": ["uma", "maki", "chaki", "ñawi", "simi", "sonqo"],
    "abstracto": ["pacha", "p'unchaw", "apu"],
}.items():
    CLASE.update({s: _clase for s in _lista})
COLORES = {"puka", "anqas", "q'omer", "yuraq", "yana"}
DECISIONES = ROOT / "data" / "revision_adjetivo_sustantivo.csv"  # columna `decision`: "si" / "no"


def sugerencia_par(adj: str, sust: str):
    """(nivel, motivo) para un adjetivo+sustantivo; nivel in {'ok','dudoso','sin_sentido'}."""
    clase = CLASE.get(sust)
    if adj in COLORES:
        if clase == "abstracto":
            return "sin_sentido", "color aplicado a un concepto abstracto o al tiempo"
        if sust == "wayra":
            return "sin_sentido", "el viento no tiene color"
        if clase == "persona":
            return "dudoso", "color aplicado a una persona o parentesco"
        if sust in ("inti", "killa") and adj in ("q'omer", "anqas"):
            return "dudoso", "color poco habitual para el sol o la luna"
        if sust == "sonqo":
            return "dudoso", "color poco habitual para el corazon"
    if adj == "musuq" and clase == "cuerpo":
        return "dudoso", "parte del cuerpo descrita como 'nueva'"
    return "ok", ""


def cargar_decisiones() -> dict:
    """{(adj, sust): 'si'|'no'} desde el CSV de revision manual, si existe."""
    if not DECISIONES.exists():
        return {}
    with DECISIONES.open(encoding="utf-8-sig", newline="") as fh:
        return {(r["adjetivo"], r["sustantivo"]): r["decision"].strip().lower()
                for r in csv.DictReader(fh) if r.get("decision", "").strip()}

# Prefijos de oracion. Glosas aclaradas por el usuario; "Cheqapta" va como "En verdad" en frases.
ADVERBIOS = {None: None, "Cheqapta": "En verdad", "Nisqanman hina": "Según lo que dijo,",
             "Kunan": "Hoy", "Qayna": "Ayer", "Paqarin": "Mañana"}
TIEMPO_ADV = {"presente": "Kunan", "pasado": "Qayna", "futuro": "Paqarin"}  # confirmados en el diccionario
ADV_GENERALES = [None, "Cheqapta", "Nisqanman hina"]

# Sinonimos en espanol: se guardan en `espanol_alternativas`, no como filas duplicadas.
SINONIMOS_ES = {
    "En verdad": ["Verdaderamente", "De verdad"], "Según lo que dijo,": ["Según lo dicho,"],
    "mojado": ["húmedo"], "mojada": ["húmeda"], "quinua": ["quinoa"], "Hoy": ["Ahora"],
}

# Entradas sueltas aclaradas por el usuario: (quechua, espanol, alternativas, estado, nota)
LEXICO_ESPECIAL = [
    ("Pasaq", "Pasaré", "", "por_validar",
     "Glosa del usuario. El diccionario solo trae la raiz pasay (ir; pasar); el futuro estandar de 'yo' seria pasasaq."),
    ("Maypichus", "¿Dónde está?", "¿Dónde estás?", "confirmado_por_usuario",
     "maypi = ¿donde? (diccionario). El usuario fija 'donde esta' y lo valido ante frases nominales; no se usa ante oraciones con verbo."),
    ("Allin puni", "Muy bueno", "Definitivamente bueno", "confirmado_diccionario",
     "allin = bueno/bien y -puni = seguramente, en cada caso (diccionario); 'Muy bueno' es la glosa del usuario."),
    ("Nisqa", "Dijo", "", "confirmado_por_usuario", "nisqa = dijo (usuario). No figura en el diccionario Ayacucho."),
    ("Nisqanman hina", "Según lo que dijo", "Según lo dicho", "confirmado_por_usuario",
     "nisqa = dijo (usuario); -nman hina = segun."),
    ("Cheqapta", "Verdad", "En verdad | De verdad | Verdaderamente", "confirmado_parcial",
     "cheqa = verdadero/cierto (diccionario). En oraciones se usa 'En verdad'."),
    ("Kunan", "Hoy", "Ahora", "confirmado_diccionario", "adv.t. del diccionario Ayacucho."),
    ("Qayna", "Ayer", "", "confirmado_diccionario", "adv.t. del diccionario Ayacucho."),
    ("Paqarin", "Mañana", "", "confirmado_diccionario", "adv.t. del diccionario Ayacucho."),
]
# Estado de palabras con aclaraciones. El resto de palabras queda como "glosa_original".
ESTADO = {
    "yacha": ("confirmado_por_usuario", "yacha = saber (confirmado por el usuario). El diccionario Ayacucho trae yachay = vivir (habitar): posible segunda lectura."),
    "kino'a": ("confirmado_por_usuario", "kinoa = cereal andino; variante kinoa añadida. No figura en el diccionario."),
    "hoq'o": ("confirmado_por_usuario", "mojado/humedo. El diccionario Ayacucho trae 'qoqo' (mojado, humedo): posible variante regional, por validar."),
    "ch'aki": ("confirmado_diccionario_y_usuario",
               "El diccionario trae chaki = pie Y seco. El usuario indica que en textos 'ch'aki' tambien se usa para 'pie': se generan ambas lecturas."),
    "chaki": ("confirmado_diccionario_y_usuario", "pie; tambien se escribe ch'aki (variante 'alterna')."),
}

OBJETOS_FISICOS = ["wasi", "runa", "allqu", "misi", "mayu", "orqo", "inti", "killa", "yaku", "nina", "allpa",
                   "wayra", "t'ika", "sach'a", "mikhuna", "yachaywasi", "uma", "maki", "chaki", "ñawi"]
# Compatibilidad semantica: juicio del autor del script, para que un hablante lo revise.
OBJETOS_DE = {
    "miku": ["mikhuna", "papa", "sara", "kino'a", "t'anta"],
    "upya": ["yaku"],
    "qawa": OBJETOS_FISICOS,
    "ruwa": ["wasi", "yachaywasi", "mikhuna", "t'anta", "punku", "nina"],
}

# Variantes Ayacucho confirmadas en data/candidatos/runasimi_ayacucho_es.tsv
VARIANTE_AYACUCHO = {"mikhuna": "mikuna", "huch'uy": "uchuy", "q'omer": "qomer",
                      "p'unchaw": "punchaw", "t'anta": "tanta", "mauk'a": "mawka"}
# Ortografia alterna indicada por el usuario: kinoa, y 'chaki' (pie) escrito con apostrofe.
VARIANTE_ALTERNA = {"kino'a": "kinoa", "chaki": "ch'aki"}
VARIANTES = {"ayacucho": VARIANTE_AYACUCHO, "alterna": VARIANTE_ALTERNA}


def ort(texto: str, dialecto: str) -> str:
    """Ortografia de cada palabra segun el dialecto (las palabras sin variante no cambian)."""
    if dialecto == "base":
        return texto
    mapa = VARIANTES[dialecto]
    return " ".join(mapa.get(w, w) for w in texto.split())


def verbo_quechua(raiz: str, tiempo: str, persona: str) -> str:
    return raiz + SUFIJOS[tiempo][persona]


def verbo_espanol(infinitivo: str, tiempo: str, persona: str) -> str:
    i = IDX_ES[persona]
    if infinitivo in IRREGULARES:
        return IRREGULARES[infinitivo][tiempo][i]
    if tiempo == "futuro":
        return infinitivo + _FUTURO[i]
    base = infinitivo[:-2]
    return base + _REGULAR[infinitivo[-2:]][tiempo][i]


def _articulo(sust: str) -> str:
    return ARTICULO_ESPECIAL.get(sust) or ("la" if SUSTANTIVOS[sust][1] == "f" else "el")


def _adjetivo_es(adj: str, sust: str) -> str:
    masc, fem = ADJETIVOS[adj]
    return fem if SUSTANTIVOS[sust][1] == "f" else masc


PERSONA_DE = {pron: p for p, pron in PRON_QU.items()}


def oracion(verbo: str, objeto, sujeto: str, tiempo: str, adverbio, dialecto: str = "base"):
    """`sujeto` es el pronombre quechua (ñuqa, qam, ...)."""
    persona = PERSONA_DE[sujeto]
    q = [adverbio] if adverbio else []
    q += [sujeto]
    es = [ADVERBIOS[adverbio]] if adverbio else []
    es += [PRON_ES[persona], verbo_espanol(VERBOS[verbo][0], tiempo, persona)]
    if objeto:
        q.append(ort(objeto, dialecto) + "ta")
        es += [_articulo(objeto), SUSTANTIVOS[objeto][0]]
    q.append(verbo_quechua(verbo, tiempo, persona))
    texto_es = " ".join(es)
    return " ".join(q), texto_es[0].upper() + texto_es[1:]


def frase_nominal(adj: str, sust: str, dialecto: str, ancha: bool = False):
    q = (["Ancha"] if ancha else []) + [ort(adj, dialecto), ort(sust, dialecto)]
    es = [SUSTANTIVOS[sust][0]] + (["muy"] if ancha else []) + [_adjetivo_es(adj, sust)]
    return " ".join(q), " ".join(es)


def caso(adj, sust: str, sufijo: str, dialecto: str, ancha: bool = False):
    q = (["Ancha"] if (adj and ancha) else []) + ([ort(adj, dialecto)] if adj else [])
    q.append(f"{ort(sust, dialecto)}-{sufijo}")
    es = [CASOS[sufijo], _articulo(sust), SUSTANTIVOS[sust][0]]
    if adj:
        es += (["muy"] if ancha else []) + [_adjetivo_es(adj, sust)]
    return " ".join(q), " ".join(es)


def _permitido(adj: str, sust: str) -> bool:
    decision = _DECISIONES.get((adj, sust))
    if decision in ("si", "no"):
        return decision == "si"
    return sust not in ADJ_BLOQUEO.get(adj, set())


def _motivo(adj, sust: str) -> str:
    """Motivo de revision para el par (vacio si es razonable o si el usuario lo aprobo)."""
    if not adj or _DECISIONES.get((adj, sust)) == "si":
        return ""
    return sugerencia_par(adj, sust)[1]


def _motivo_oracion(verbo: str, objeto) -> str:
    if verbo == "yacha" and objeto is None:
        return "saber sin complemento: ¿saber qué?"
    return ""


_DECISIONES = cargar_decisiones()


def _minuscula(texto: str) -> str:
    return texto[0].lower() + texto[1:]


def _con_allin_puni(base):
    """'Allin puni' (Muy bueno) validado por el usuario como prefijo de frases, casos y oraciones."""
    pronombres = set(PRON_QU.values())
    for q, e, categoria, tiempo, persona, revisar in base:
        if categoria == "Léxico" or q.startswith("Ancha "):
            continue
        if categoria.startswith("Oración") and q.split()[0] not in pronombres:
            continue  # ya lleva otro adverbio
        yield f"Allin puni {q}", f"Muy bueno {_minuscula(e)}", categoria, tiempo, persona, revisar


def _con_maypichus(dialecto: str):
    """'Maypichus' (¿Dónde está?) solo tiene sentido ante una frase nominal: '¿Dónde está la tierra roja?'."""
    for adj in ADJETIVOS:
        for sust in SUSTANTIVOS:
            if _permitido(adj, sust):
                q, e = frase_nominal(adj, sust, dialecto)
                yield (f"Maypichus {q}", f"¿Dónde está {_articulo(sust)} {e}?", "Frase nominal", "", "",
                       _motivo(adj, sust))


def _filas_por_dialecto(dialecto: str):
    base = list(_filas_base(dialecto))
    yield from base
    yield from _con_allin_puni(base)
    yield from _con_maypichus(dialecto)


def _filas_base(dialecto: str):
    for adj in ADJETIVOS:
        for sust in SUSTANTIVOS:
            if not _permitido(adj, sust):
                continue
            for ancha in (False, True):
                q, e = frase_nominal(adj, sust, dialecto, ancha)
                yield q, e, "Frase nominal", "", "", _motivo(adj, sust)
    for sust in SUSTANTIVOS:
        for suf in CASOS:
            q, e = caso(None, sust, suf, dialecto)
            yield q, e, "Caso gramatical", "", "", ""
            for adj in ADJETIVOS:
                if not _permitido(adj, sust):
                    continue
                for ancha in (False, True):
                    q, e = caso(adj, sust, suf, dialecto, ancha)
                    yield q, e, "Caso gramatical", "", "", _motivo(adj, sust)
    for verbo, (_, transitivo) in VERBOS.items():
        objetos = OBJETOS_DE[verbo] if transitivo else [None]
        for tiempo in TIEMPOS:
            for persona in PERSONAS:
                # El futuro de 2s/2p se escribe igual que el presente: solo se desambigua con 'Paqarin'.
                ambiguo = tiempo == "futuro" and persona in ("2s", "2p")
                adverbios = [TIEMPO_ADV[tiempo]] if ambiguo else ADV_GENERALES + [TIEMPO_ADV[tiempo]]
                for objeto in objetos:
                    for adv in adverbios:
                        q, e = oracion(verbo, objeto, PRON_QU[persona], tiempo, adv, dialecto)
                        yield q, e, f"Oración {tiempo}", tiempo, persona, _motivo_oracion(verbo, objeto)
    yield from _filas_lexico(dialecto)


def _filas_lexico(dialecto: str):
    for r, (es, _) in SUSTANTIVOS.items():
        yield ort(r, dialecto), es, "Léxico", "", "", ""
    for r, (m, f) in ADJETIVOS.items():
        yield ort(r, dialecto), m if m == f else f"{m}/{f}", "Léxico", "", "", ""
    for r, (inf, _) in VERBOS.items():
        yield r + "y", inf, "Léxico", "", "", ""
    if dialecto == "base":
        for q, e, *_ in LEXICO_ESPECIAL:
            yield q, e, "Léxico", "", "", ""


def alternativas(espanol: str) -> str:
    presentes = {o: s for o, s in SINONIMOS_ES.items()
                 if re.search(rf"(?<!\w){re.escape(o)}(?!\w)", espanol)}
    if not presentes:
        return ""
    salida = []
    for k in range(max(len(s) for s in presentes.values())):
        texto = espanol
        for original, sinonimos in presentes.items():  # todos a la vez: "mojado/mojada" -> "húmedo/húmeda"
            texto = texto.replace(original, sinonimos[min(k, len(sinonimos) - 1)])
        salida.append(texto)
    return " | ".join(salida)


def _fusionar_lexico(filas):
    """Una palabra con varias lecturas (ej. ch'aki = seco / pie) queda en una sola fila."""
    lexico, resto = {}, []
    for f in filas:
        if f["categoria"] != "Léxico":
            resto.append(f)
            continue
        previo = lexico.get(f["quechua"])
        if previo is None:
            lexico[f["quechua"]] = f
        elif f["espanol"] not in previo["espanol"].split("; "):
            previo["espanol"] += "; " + f["espanol"]
    return resto + list(lexico.values())


def generar():
    vistos, filas = set(), []
    for dialecto in DIALECTOS:
        for q, e, categoria, tiempo, persona, revisar in _filas_por_dialecto(dialecto):
            if (q, e) in vistos:
                continue
            vistos.add((q, e))
            filas.append({"quechua": q, "espanol": e, "categoria": categoria,
                          "dialecto": dialecto, "tiempo": tiempo, "persona": persona, "revisar": revisar})
    especiales = {q: alt for q, _, alt, *_ in LEXICO_ESPECIAL}
    filas = _fusionar_lexico(filas)
    for i, f in enumerate(filas, 1):
        f["id"] = i
        f["espanol_alternativas"] = especiales.get(f["quechua"]) if f["categoria"] == "Léxico" and f["quechua"] in especiales \
            else alternativas(f["espanol"])
    return filas


def escribir_lexico(ruta: Path) -> None:
    def fila(tipo, r, es, extra):
        estado, nota = ESTADO.get(r, ("glosa_original", ""))
        if r in VARIANTE_AYACUCHO:
            estado, nota = "confirmado_diccionario", nota or "variante Ayacucho verificada en el diccionario"
        return (tipo, r, VARIANTE_AYACUCHO.get(r, ""), VARIANTE_ALTERNA.get(r, ""), es, extra, estado, nota)

    filas = []
    for r, (es, g) in SUSTANTIVOS.items():
        filas.append(fila("sustantivo", r, es, g))
    for r, (m, f) in ADJETIVOS.items():
        filas.append(fila("adjetivo", r, m if m == f else f"{m}/{f}", ""))
    for r, (inf, tr) in VERBOS.items():
        filas.append(fila("verbo", r, inf, ";".join(OBJETOS_DE[r]) if tr else "(intransitivo)"))
    for q, es, alt, estado, nota in LEXICO_ESPECIAL:
        filas.append(("expresion", q, "", "", es, alt, estado, nota))
    with ruta.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "quechua", "variante_ayacucho", "variante_alterna", "espanol",
                    "genero_objetos_o_alternativas", "estado", "nota"])
        w.writerows(filas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", type=Path, default=SALIDA)
    a = p.parse_args()
    filas = generar()
    campos = ["id", "quechua", "espanol", "espanol_alternativas", "categoria", "dialecto", "tiempo", "persona", "revisar"]
    with a.salida.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    escribir_lexico(LEXICO)
    print(f"{len(filas)} filas -> {a.salida}\nlexico -> {LEXICO}")


if __name__ == "__main__":
    main()
