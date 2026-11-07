# AGENTS.md — traductorquechua

> Instrucciones para agentes de IA (Claude Code y similares) que operen este proyecto.
> Idioma de trabajo con el usuario: **español**.

## 1. Objetivo del proyecto

Traductor Quechua <-> Español. Este documento fija, de momento, la **política de fuentes de datos léxicos** (diccionarios) tras analizar si se podía consumir Glosbe. Código actual en `scripts/`: `limpiar_datos.py` (limpieza y reparto sin fuga → `data/limpio/`), `baseline_nllb.py` (zero-shot de NLLB-600M) y `finetune_nllb.py` (fine-tuning, **escrito y con sintaxis verificada, nunca ejecutado**). Decisiones y estado del modelo en §6.

## 2. Análisis de Glosbe (5-nov-2026) — conclusión: NO usar como fuente automatizada

Se probó contra `https://glosbe.com` con `curl`:

| Hallazgo | Evidencia |
|---|---|
| La API pública antigua `/gapi/translate?from=qu&dest=es&phrase=...` **ya no existe** | Responde una página HTML "Error in Glosbe", no JSON |
| No hay API oficial documentada | Solo hay páginas HTML: `/qu/es/<palabra>`, `/es/qu/<palabra>` |
| Protección anti-bot **Cloudflare** ("Just a moment...") | `403` con cabeceras de navegador en `/qu/es/allin` y `/terms` |
| `robots.txt` prohíbe lo que serviría para consumirlo | `Disallow: /ajax/`, `/ajax2/`, `*/fragment/*`, `/*?` (cualquier URL con query string) |
| Su contenido tiene licencia/términos propios | No se pudo leer `/terms` (bloqueado por el desafío) |

**Decisión:** no scrapear Glosbe ni intentar evadir el desafío de Cloudflare (Playwright/undetected, rotación de UA/IP, etc.). Va contra su `robots.txt` y sus protecciones, es frágil (HTML sin contrato) y no hay garantía de licencia. Si el usuario insiste en Glosbe, la única vía legítima es **pedirles autorización/acceso a datos** y leer sus términos en un navegador.

## 3. Alternativa verificada: Wiktionary (API MediaWiki, abierta)

Probado y funcionando sin autenticación:

- Extracto de una entrada (contiene la sección `== Quechua ==`):
  `GET https://en.wiktionary.org/w/api.php?action=query&titles=<palabra>&prop=extracts&explaintext=1&format=json&redirects=1`
- Definiciones estructuradas: `GET https://en.wiktionary.org/api/rest_v1/page/definition/<palabra>` (HTTP 200).
- Listado de lemas: categoría `Category:Quechua lemmas` (~2 255 páginas, 15 subcategorías). Las categorías `Southern/Cusco Quechua lemmas` no existen con ese nombre; enumerar las subcategorías reales con `list=categorymembers&cmtitle=Category:Quechua_lemmas&cmtype=subcat`.
- `qu.wiktionary.org` existe pero es pequeño (~362 artículos): útil como complemento, no como fuente principal.

Ejemplo real: `allin` → Quechua, adjetivo: *good, fine, okay, nice, right; sensible, useful; healthy*; adverbio: *well*; forma alternativa `alli`.

**Limitaciones a tener presentes:**
- Las glosas están en **inglés**, no en español: para Quechua→Español habrá que traducir la glosa (EN→ES) o cruzar con otra fuente.
- Una página mezcla idiomas (ej. `allin` trae también Finnish): filtrar solo la sección `Quechua`.
- Quechua tiene muchas variedades (Ayacucho `quy`, Cusco `quz`, etc.) y ortografías distintas: **registrar siempre la variedad** de cada entrada.
- Licencia de Wiktionary: CC BY-SA 4.0 → conservar atribución y compartir igual al redistribuir.

## 4. Otras fuentes a evaluar (no probadas aún)

Preferir **volcados descargables** antes que consultas web repetidas: dumps de Wikimedia, corpus paralelos en OPUS/Tatoeba, datasets de Hugging Face para quechua (verificar licencia antes de usar), y diccionarios institucionales (p. ej. MINEDU/Academia Mayor de la Lengua Quechua) si se obtiene permiso.

### 4.1 Churana (`github.com/Llamacha/Churana`) — revisado 5-nov-2026

**Veredicto:** es un **mapa de recursos, no una fuente de datos.** Solo contiene un `README.md` (~11 KB) con enlaces a papers y proyectos sobre lenguas peruanas (Aymara, Quechua Central/Sureño, Shipibo). Sin código, sin datasets, sin licencia declarada, 5 estrellas, última actualización dic-2025. Mantenido por gente de Siminchikkunarayku/Hinantin.

Recursos de su sección *Quechua Sureño* relevantes para español↔quechua:

| Recurso | Aporte | Disponibilidad |
|---|---|---|
| Ortega et al., 2020 — NMT quechua sureño con segmentación morfológica | Estado del arte; idea clave: segmentar sufijos | Paper en Springer; sin código/datos visibles |
| Huarcaya, 2020 — NMT quechua chanca con transformers (BLEU 39.5) | Baseline concreto | Tesis, repositorio UPEU |
| Rios et al., 2008 — treebank paralelo quechua-español | Corpus paralelo pequeño | Repositorio Utrecht |
| Rios, 2015 — toolkit, traducción híbrida español→quechua cusqueño | Reglas + lexicón morfológico | Repositorio ZORA (UZH) |
| Rios 2011/2014 — corrector ortográfico y normalización | Normalizar ortografía entre variedades | ZORA (UZH) |
| Ortega et al., 2018 — alineación de morfemas (quechua/finés) | Técnica para lenguas aglutinantes | ACL Anthology |

Los recursos de voz (Siminchik, ASR) no aplican a traducción de texto.

**Pendiente para conseguir datos reales (no hecho aún):**
1. Revisar ZORA (UZH): recursos y toolkit de Rios.
2. Revisar las organizaciones GitHub `hinantin` y Siminchikkunarayku.
3. Buscar corpus abiertos: OPUS/Tatoeba y datasets de Hugging Face para quechua (verificar licencia). El modelo NLLB de Meta soporta quechua ayacuchano (`quy_Latn`) y serviría como baseline.

### 4.2 mt_quechua_spanish (`github.com/johneortega/mt_quechua_spanish`) — revisado 5-nov-2026

**Veredicto:** **sí aporta datos reales** (a diferencia de Churana). Código y corpus del paper de Ortega, Castro Mamani y Cho (NYU) sobre NMT quechua→español. 5 estrellas, último push feb-2020, **sin licencia declarada**. Casi todo es texto plano + scripts Python.

| Carpeta | Contenido | Utilidad |
|---|---|---|
| `corpora/opus_jw_300/` | Corpus paralelo qu↔es: `train` 17 500 pares, más `validate` y `test` | Alta para fine-tuning; poco volumen |
| `corpora/magazine/kallpa.*` | 99 líneas de cuentos andinos, traducción profesional | Alta calidad; sirve como **test de evaluación** |
| `annotations/` | Evaluaciones humanas de 5 sistemas + scripts de creación | Comparar modelos |
| `bpe_guided/qu_es/` | Segmentación BPE guiada por sufijos (`suffixes.qu/.es`) | Idea aplicable: segmentar sufijos |

**NO es un diccionario:** `quechua_glossary.txt` / `spanish_glossary.txt` (~6 150 líneas) son salida de un script de segmentación (`parse_spanish_file.py`), no un léxico bilingüe palabra a palabra. Para diccionario seguir con Wiktionary (§3).

**Advertencias:**
1. **Licencia:** el corpus viene de publicaciones de una sola organización (© visible dentro de los datos); el repositorio de origen las retiró de su sitio por temas de licencia. Aceptable para investigación/pruebas internas; **verificar antes de uso en un producto público**.
2. **Dominio sesgado:** religioso y revistas (salud, familia). Traducirá peor texto cotidiano o administrativo.
3. **Variedad:** ortografía (`juntʼasqa`, `kausakusunman`) apunta a quechua sureño tipo Collao/boliviano; confirmar si coincide con la variedad objetivo (Ayacucho `quy` o Cusco `quz`).
4. **Volumen:** 17 500 pares no alcanzan para entrenar desde cero; sí para ajustar un modelo preentrenado (p. ej. NLLB `quy_Latn`).
5. **Ruido:** las primeras líneas de `train` son avisos de copyright o títulos sueltos con número de página; limpiar antes de usar.

**Uso recomendado:** dataset de fine-tuning y evaluación, combinado con otras fuentes; `magazine` como test de calidad.

### 4.3 IWSLT2025_Quechua_data (`github.com/Llamacha/IWSLT2025_Quechua_data`) — revisado 5-nov-2026

**Veredicto:** **aporta poco para traducción de texto.** Es el repo de la tarea de **traducción de voz** (IWSLT 2025, Quechua→Español, pista low-resource). 6 estrellas, último push abr-2025. Lo valioso es el evaluador y un set pequeño de validación.

| Carpeta | Contenido | Utilidad para texto es↔qu |
|---|---|---|
| `additional_mt_text/` | `opus_jw_300` y `magazine` | **Ninguna nueva**: mismo material (mismos tamaños en bytes) que §4.2; no bajarlo dos veces |
| `que_spa_synthetic_translation/train/txt/train.tsv` | 1 413 filas `que`/`spa` (+ audio, ~8 h, corpus Huqariq) | Media-baja (ver problemas) |
| `que_spa_unconstrained/{train,valid,test}/` | Segmentos de habla con `.que` y `.spa` (573 train, 125 valid, 125 test) | Media-baja; la parte de texto más confiable del repo |
| `scripts/` | `main.py`, `bleu_scorer.py`, `chrF_scorer.py` (BLEU y chrF++) | **Útil**: evaluador listo para medir el traductor |

**Problemas con los datos de texto:**
1. **Transcripciones de habla:** minúsculas, sin puntuación, mezcla español/quechua (ej. "es nuevo código procesal penal dos bueno seis vacantes..."). Estilo distinto al texto escrito.
2. **Traducciones sintéticas ruidosas:** hechas con Google Translate y posteditadas por un hablante. Hay filas con traducción vacía o truncada y otras con salidas erróneas (ej. "ayúdennos"). Filtrar antes de usar.
3. **Variedad:** quechua sureño con varios hablantes (Andahuaylas, entre otros); no está claro que coincida con Ayacucho/Cusco.
4. **Volumen:** unas pocas miles de líneas en total.

**Licencia — la más restrictiva hasta ahora: CC BY-NC-ND 3.0** (sin uso comercial, sin obras derivadas). Entrenar o ajustar un modelo con estos datos podría contar como obra derivada, lo que es un obstáculo serio para un traductor en producción. Apto solo para investigación o pruebas internas. Audio propiedad de Siminchikkunarayku y Llamacha.

**Uso recomendado:** reutilizar los scripts de BLEU/chrF++ para evaluar; usar `que_spa_unconstrained` como validación pequeña del dominio hablado.

### 4.4 Resumen comparativo de las fuentes de §4.1–4.3

| Repo | ¿Datos de texto? | Licencia | Veredicto |
|---|---|---|---|
| Churana | No | Ninguna | Solo un índice de referencias |
| mt_quechua_spanish | Sí (~17 500 pares) | Sin declarar; parte con © | Mejor fuente de texto |
| IWSLT2025 | Poco, de habla | CC BY-NC-ND 3.0 | Solo evaluación y validación |

### 4.5 Papers de traducción español↔quechua — buscados y descargados 5-nov-2026

**Aviso sobre "Q1":** la búsqueda **no encontró revistas Q1 accesibles** sobre el tema. Casi todo lo publicado es de **congresos/talleres** (ACL Anthology, AmericasNLP, LACCEI), que no tienen cuartil Scimago de revista. El único artículo de revista identificado es **Ortega, Castro Mamani y Cho (2020), *Machine Translation* (Springer), DOI `10.1007/s10590-020-09255-9`**: es de pago (la descarga devolvió HTML, no PDF; se borró) y su cuartil **no se verificó** (consultar Scimago antes de citarlo como Q1). Revisados con revisión por pares: los de ACL Anthology; **sin revisión formal**: los de arXiv.

PDFs en `papers/` (todos de acceso abierto, verificados como PDF):

| Archivo | Tema | Tipo |
|---|---|---|
| `2021_REPUcs_AmericasNLP_es-quz.pdf` | Sistema es→quechua sureño, 1.º en Track 2 de AmericasNLP 2021; construye corpus nuevos (poemas, canciones, léxicos, manuales) | Taller ACL |
| `2023_LCT-EHU_AmericasNLP_es-qu.pdf` | Modelos preentrenados + datos de dominio literario/legal; el preentrenado español-finés rindió mejor; datos sintéticos malos bajan la calidad | Taller ACL |
| `2025_UCSP_AmericasNLP_synthetic_mT5.pdf` | Frases sintéticas generadas con LLM + ajuste de mT5, ambas direcciones | Taller ACL |
| `2026_Duran_Silberztein_dictionary_qu-es_LoResMT.pdf` | Diccionario semiautomático qu↔es de 1 444 verbos (enlace/licencia de descarga no confirmados; leer el paper) | Taller ACL |
| `2024_LACCEI_Sulla_NMT_BERT_qu-es.pdf` | Transformer + LMfusion + segmentación PRPE, qu→es | Congreso LACCEI |
| `2023_NMT_indigenous_languages_Americas_intro_arXiv2306.06804.pdf` | Introducción a NMT para lenguas indígenas de América | arXiv |
| `2024_NLP_progress_indigenous_LatAm_NAACL_arXiv2404.05365.pdf` | Panorama NLP en lenguas indígenas latinoamericanas | arXiv |
| `2022_multilingual_transfer_indigenous_NMT_arXiv2205.06993.pdf` | Transferencia multilingüe para NMT indígena | arXiv |

Datos útiles extraídos de los resultados de búsqueda (no leídos a fondo): AmericasNLP se centra en quechua **Ayacucho (`quy`)**, así que los resultados pueden no trasladarse a Cusco; el entrenamiento oficial de quechua salió de publicaciones de una sola organización.

### 4.6 Datasets descargados (en `data/`) — con auditoría de calidad

| Carpeta | Fuente | Variedad | Filas (train/val/test) | Licencia |
|---|---|---|---|---|
| `data/hf_somosnlp_spanish_to_quechua/` | HF `somosnlp-hackathon-2022/spanish-to-quechua` | **Ayacucho** (`quy`), columnas `es`,`qu` | 102 747 / 12 844 / 12 843 | **No declarada** |
| `data/hf_pollitoconpapass_cuzco/` | HF `pollitoconpapass/cuzco-quechua-translation-spanish` | **Cusco** (`quz`), columnas `spa`,`quz` | 106 432 / 15 112 / 13 305 (+ `eval` 1 189) | **No declarada**; el README es mínimo y no cita fuentes |

Descartado: `VoiceProfiler/QuEsT` (solo audio, ~40 min, GPLv3) — no sirve para traducción de texto.

**Hallazgos de la auditoría (medidos, no supuestos):**
1. **Fuga entre particiones:** frases `es` de validación/test que ya están en `train`: Ayacucho 7.3 % / 6.9 %; Cusco 17.0 % / 3.2 %; **el split `eval` de Cusco está 100 % contenido en `train`** → inservible para evaluar. Deduplicar y reparticionar antes de medir BLEU/chrF++.
2. **Sesgo bíblico/religioso:** ~25 % de las filas de `train` contienen "Dios/Jehová/Biblia" en ambos datasets (consistente con esas publicaciones). El README de Ayacucho lo admite.
3. **Ruido en Cusco:** 2 637 filas con `spa == quz` (sin traducir), 345 filas vacías o de <3 caracteres, 1 465 filas de más de 500 caracteres (párrafos sin segmentar, ej. texto de tesis). En Ayacucho: 53 filas vacías/cortas, 3 idénticas, 8 largas.
4. **Duplicados en `train`:** 3 454 (Ayacucho) y 4 629 (Cusco).
5. **Procedencia:** Ayacucho agrupa blogs, Tatoeba, AmericasNLP 2021 y publicaciones de una sola organización (limpieza manual parcial); Cusco no documenta origen. Revisar derechos de cada fuente antes de uso comercial.
6. **Codificación:** al imprimir en consola de Windows los acentos salen corruptos; es un artefacto de la consola, no se verificó la codificación real del archivo. Leer siempre con `pandas.read_parquet` y `PYTHONIOENCODING=utf-8`.

**Pendiente antes de entrenar:** limpiar (quitar duplicados, vacías, `es==qu` y párrafos largos), reparticionar sin fuga, y decidir variedad objetivo (Ayacucho o Cusco; **no mezclarlas**). Los datasets pesan ~42 MB: añadir `data/` y `papers/` al `.gitignore` si se versiona.

## 5. Reglas para el agente

1. Responder siempre en español.
2. Respetar `robots.txt` y términos de cualquier sitio; no evadir protecciones anti-bot.
3. Al consumir una API pública: `User-Agent` identificable con contacto, límite de ritmo (≤1 req/s), caché local de respuestas, sin reintentos agresivos.
4. Guardar en cada registro léxico: `palabra`, `variedad`, `categoría gramatical`, `glosas`, `fuente`, `licencia`, `fecha_consulta`.
5. Validar la entrada (no confiar en datos externos) y manejar errores de red/HTTP explícitamente, sin silenciarlos.
6. Reportar con evidencia (código HTTP, muestra de respuesta) en vez de asumir que algo funciona.
7. Tras añadir código, actualizar la sección 1 y este documento; mantener este archivo como guía + índice, no como volcado de detalle.
8. No lanzar entrenamientos largos sin que el usuario lo pida; dejar el script listo y que él lo ejecute. Sacrebleu ya está instalado en el venv `D:/Backend/redesneuronales/.venv`.

## 6. Modelo: decisiones y estado (5-nov-2026)

**Variedad objetivo: Ayacucho (`quy`)** — decidida por razones técnicas, **sin confirmar el público final** (si los usuarios son de Cusco, revisar). Evidencia leída en los papers (REPUcs 2021, LCT-EHU 2023):
- LCT-EHU: *"Quechua should be regarded as a linguistic family rather than a single language"*, con inteligibilidad limitada entre variedades lejanas → no mezclar variedades.
- AmericasNLP trabaja sobre `quy`. REPUcs no usó los datos de Cusco ("por falta de tiempo"). En LCT-EHU, añadir datos de Cusco ayudó menos que añadir datos de Ayacucho, y el mejor modelo de test los excluyó (salvedad: el test es `quy`, así que el resultado favorece a Ayacucho por construcción).
- NLLB solo tiene `quy_Latn`.

**Modelo:** NLLB-200-distilled-600M ajustado, un solo modelo bidireccional es↔qu, bíblico reducido a 30 % de train (`--frac-biblico 0.3`). Selección del mejor punto por chrF++ en valid **no bíblico**. Hiperparámetros iniciales (lr 3e-5, batch 32, 3 épocas) son un punto de partida sin validar. GPU disponible: RTX PRO 5000 Blackwell, 48 GB.

**Referencia de papers (no comparable directamente, otro test):** LCT-EHU obtuvo chrF 37-38.6 en es→quy con opus-mt-es-fi ajustado (el preentrenado es→finés rindió mejor que es→inglés por ser aglutinante; el "corpus copiado" bajó el chrF). Nuestra línea base zero-shot da 27-33. El paper `2022_multilingual_transfer...` no trata quechua en concreto (solo se leyó por búsqueda de términos, no a fondo).

| Paso | Estado |
|---|---|
| Datos limpios + línea base | Terminado |
| Elegir variedad | Decidida (Ayacucho), pendiente confirmar público |
| Script de fine-tuning | Escrito, no ejecutado |
| Entrenar y evaluar | Pendiente |
| Evaluación externa (`kallpa`, valid de IWSLT, scorer de IWSLT) | Pendiente |
| Licencias para producción (licencia de los datos sin declarar; NLLB CC-BY-NC sin verificar) | Pendiente |
| ZORA (UZH), `hinantin`, Siminchikkunarayku | Pendiente |
| Git + `.gitignore` (`data/`, `papers/`, `models/`, `.env`) | Pendiente |

## 7. Este es ahora el proyecto del examen final MIA-203 (5-nov-2026)

Decisión del usuario: el traductor es↔qu reemplaza al proyecto de pentesting. La bitácora y el lab de pentesting se archivaron en `D:\Backend\redesneuronales\examenfinal\` (`bitacora.zip`, `examenfinal-kali-image.tar.gz`; restaurar la imagen con `gunzip -c ... | docker load`). Se borraron las imágenes Docker `examenfinal-kali` y `dvwa`.

Requisitos del PDF (`redesneuronales/examenfinal/ExamenFinal_MIA_203 (1).pdf`) y brechas actuales:
- R4: entrenar + al menos una ablación (p. ej. `--frac-biblico` 0.3 vs 1.0) y, si hay tiempo, varias semillas. **Pendiente.**
- R5: análisis de errores sistemático (por dominio, longitud, sufijos) o mapas de atención. **Pendiente.**
- R6: reportar hardware y horas de GPU (el script solo imprime el tiempo total) + limitaciones/sesgos (sesgo bíblico ya documentado). **Parcial.**
- Cómputo: el entrenamiento debe poder reproducirse en Colab T4 (16 GB); `finetune_nllb.py` usa **bf16, que la T4 no soporta de forma nativa** → pasar a fp16 con escalado o LoRA, o entregar pesos + versión reducida. **Pendiente.**
- Licencias: declarar en el paper que los datasets de HF no declaran licencia (origen: publicaciones de una sola organización) y verificar la de NLLB (CC-BY-NC). **Pendiente.**
- Repo: `README.md`, `requirements.txt`, `src/`, `configs/`, `results/`, notebook de demo en Colab, semillas fijas, enlace a pesos, commits de todos, etiqueta `v1.0-final`. Hoy no hay git ni nada de eso (los scripts están en `scripts/`).
- Entregables: paper IEEE ≤12 págs con sección *Contribuciones*, diapositivas (25 min), declaración de uso de asistentes de IA. Entrega: 6-nov (sección C) o 7-nov (A y B).

## 8. Decisiones de 5-nov-2026 (tarde): sin texto religioso

- Auditoria: el filtro de 5 palabras (Dios/Jehova/...) marcaba 34 %; el filtro ampliado (`src/marcar_religioso.py`, vocabulario de las revistas + citas + formas quechuas con sufijos) marca 55 % (train), 49 % (valid), 56 % (test). Hay ademas pares mal alineados.
- **Decision del usuario:** quitar ese texto de TODO el proyecto. Se entrena, valida y prueba solo con `data/limpio/ayacucho_general/` (49 414 / 3 477 / 2 662 pares). No se puede aislar la fuente exacta: el dataset de HF no marca la fuente por fila; el filtro es conservador y puede dejar algo sin detectar.
- Ablacion de R4 = tamano del entrenamiento (25 / 50 / 100 %), 3 semillas en el principal. Las corridas con 30 % y 100 % religioso se cancelaron.
- Linea base zero-shot (mismo test general): es->qu chrF++ 26.4 / BLEU 3.2; qu->es chrF++ 27.6 / BLEU 9.7 (`results/baseline_zeroshot/`).
- Entrenamiento en Windows: sin `torch.cuda.set_per_process_memory_fraction(0.8)` la GPU desborda a RAM compartida y se frena ~4x. NO usar `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb` (empeora).
- Despliegue: servidor AlmaLinux 13.140.220.221; `traductor.racinghorse.net` / `traductor.service` / `/opt/traductor` son de OTRA app (tutor de ingles): no tocar. Este proyecto usa `quechua.racinghorse.net`, `quechua-api`, puerto 8090, `/opt/quechua`. DNS en Contabo (registro A ya creado). Certbot con `sergiomc12@gmail.com`.
- Paper: `paper/construir.py` genera Paper.pdf (IEEEtran) y Paper.docx desde `paper/contenido.py` + `paper/resultados.py`. Faltan: resultados, tareas por integrante (Contribuciones), enlace de pesos.

## 9. Resultados finales y estado (5-nov-2026, noche)

- 3 semillas, principal (49 414 pares no religiosos): es->qu chrF++ 38.4 ± 0.4 / BLEU 9.8 ± 0.2; qu->es chrF++ 33.3 ± 0.1 / BLEU 14.7. Linea base zero-shot: 26.4 / 27.6. Ablacion de tamano (1 semilla): 25 % -> 34.9 / 31.5; 50 % -> 36.3 / 32.2. Total 1.58 h de GPU de entrenamiento. Tablas: `results/tablas.md`, `results/errores.json`.
- Ruido del test: 53 de 2 662 pares (2.0 %) son encabezados/fechas (sin ellos 38.4 / 33.0); hay pares desalineados sin cuantificar; 13 salidas es->qu (0.49 %) con repeticiones degeneradas.
- `src/finetune_ckpt.py`: variante con puntos de control cada 2 000 pasos (probada: interrumpida y retomada en el paso 50). NO se uso para los resultados del paper (el barajado por epoca cambia respecto a `finetune.py`, no reproduce los mismos numeros).
- `principal_s2` fallo una vez al arrancar (22:24) sin traceback; se relanzo. El PC se reinicio a las 19:15 y mato una corrida: no suspender el equipo durante entrenamientos.
- Los 7 PDF del MINEDU habian entrado a git por error (raiz del proyecto). Se quitaron del indice y se ignoran (`docs/*.pdf`, `docs/*.txt`); siguen en el historial local (no hay remoto). Antes de publicar el repo, reescribir historial o crear uno limpio.
- Pendiente: tareas por integrante (Contribuciones), enlace de pesos, diapositivas, evaluacion externa (FLORES-200 quy_Latn).
