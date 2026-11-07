# Traducción neuronal bidireccional español–quechua mediante ajuste fino de NLLB-200

Proyecto final de MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI FIIS). Ajustamos **NLLB-200-distilled-600M** como un solo modelo
bidireccional español–quechua (`quy_Latn`), lo comparamos con una línea base y con otros nueve modelos, y lo desplegamos como servicio web y
aplicación Android. El artículo en formato IEEE está en [`paper/`](paper/).

## Qué problema resuelve y por qué así

El quechua ayacuchano tiene pocos textos paralelos con el español, y los conjuntos públicos mezclan duplicados, fuga entre particiones
y mucho texto de publicaciones de una sola organización. Un traductor entrenado así aprende sobre todo ese estilo, y un promedio global sobre su
prueba no dice cuánto traduce fuera de él. Por eso:

1. Limpiamos y repartimos los datos **sin fuga** entre entrenamiento, validación y prueba.
2. **Excluimos las publicaciones de esa organización** (55 % de las filas) de las tres particiones.
3. Ajustamos un modelo multilingüe preentrenado, porque NLLB-200 ya incluye el quechua ayacuchano y unas cien mil oraciones no bastan para entrenar desde cero.
4. Medimos con tres semillas, una línea base, ablaciones y un examen que mide si el modelo aprende la **conjugación** verbal, no solo si memoriza.

## Resultados principales

Prueba propia: 2 662 pares ajenos a esas publicaciones. chrF++ (media ± desviación estándar de 3 semillas cuando se indica).

| Sistema | es→qu | qu→es |
|---|---|---|
| NLLB-200 sin ajustar (línea base) | 26.4 | 27.6 |
| Principal, 100 % de los datos (3 semillas) | 38.4 ± 0.4 | 33.3 ± 0.1 |
| Modelo final: principal + conjunto normalizado 15k (3 semillas) | 38.6 ± 0.1 | 33.6 ± 0.1 |

**Comparación de los diez modelos** (semilla 0). Prueba propia y FLORES-200 devtest en chrF++; concordancia en % de acierto con verbos que el
modelo no vio al entrenar:

| Modelo | Prueba es→qu | Prueba qu→es | FLORES es→qu | FLORES qu→es | Concord. es→qu | Concord. qu→es |
|---|---|---|---|---|---|---|
| Sin ajustar | 26.4 | 27.6 | 21.5 | 28.4 | 14.3 | 27.4 |
| Principal | 38.7 | 33.3 | 25.3 | 29.5 | 24.8 | 53.7 |
| Sin 15 % al azar | 38.2 | 33.1 | 25.1 | 29.6 | 21.9 | 53.7 |
| Sin 15 % peor puntuado | 38.4 | 33.2 | 25.3 | 29.5 | 21.9 | 50.5 |
| Retrotraducción ≥ 30 | 38.4 | 33.4 | 26.7 | 30.0 | 29.5 | 45.3 |
| Retrotraducción ≥ 40 | 38.6 | 33.4 | 26.3 | 30.0 | 28.6 | 49.5 |
| Ciclo automático (vuelta 1) | 38.5 | 33.4 | 30.1 | 32.2 | 28.6 | 55.8 |
| Normalizado 2k | 38.6 | 33.4 | 25.3 | 29.7 | 59.0 | 81.1 |
| **Normalizado 15k (modelo final)** | 38.7 | 33.7 | 24.9 | 29.9 | 57.1 | 82.1 |
| Normalizado 15k + literatura central | 38.5 | 33.4 | 25.1 | 29.9 | 61.9 | 76.8 |

Cómo leerlo, sin exagerar:

- En la prueba propia los diez modelos quedan a menos de un punto entre sí: no hay diferencia más allá del ruido entre semillas.
- Lo que separa a los modelos es la concordancia. Con el conjunto normalizado el acierto sube de 24.8 % a 57.1 % (es→qu) y de 53.7 % a 82.1 % (qu→es);
  con tres semillas el 15k da 57.7 ± 1.1 % y 79.3 ± 2.4 %. Un conjunto de solo 2 679 filas (2k) rinde igual que uno de 15 373.
- **El modelo final no gana en todo.** El ciclo automático obtiene más en FLORES (30.1 y 32.2), pero entrenó con pares de FLORES dev y su cifra no es comparable;
  además su concordancia es menor. Elegimos el 15k porque mejora la conjugación sin perder en la prueba propia.
- La referencia de la concordancia sale de **nuestras reglas** (paradigma de Cusco y Chanka); ninguna persona hablante la validó. Mide si el modelo aprendió ese paradigma, no si coincide con el ayacuchano estándar.
- Ninguna persona hablante evaluó las traducciones, y la prueba sale de la misma fuente que el entrenamiento: los números pueden ser optimistas para texto de otro origen.

Más resultados en el artículo: ablación del tamaño del entrenamiento (25 %, 50 %, 100 %), análisis de errores por longitud y sufijos, concordancia por persona y
tiempo, y el efecto de la decodificación en FLORES.

### Dificultades documentadas

- **Voz:** la aplicación usa el reconocimiento de voz de Android, que funciona para español pero no para quechua; por eso es más cómodo traducir de español a quechua. La voz de salida es una voz en español a velocidad reducida. No medimos la tasa de error de ese reconocimiento.
- **Apóstrofo:** las consonantes eyectivas se escriben con apóstrofo dentro de la palabra (p'unchaw, ch'aki), difícil de teclear en un teléfono. En los datos casi no aparece (57 de 49 414 oraciones de entrenamiento). Omitirlo cambia el sentido a veces: con 11 palabras del léxico, el modelo acierta 10 con apóstrofo y 8 sin él (`python -X utf8 scripts/prueba_apostrofo.py`). Es una prueba pequeña, con palabras que el modelo ya vio.

## Estructura

```
paper/          artículo IEEE: contenido.py y resultados.py (fuente única), construir.py -> Paper.tex, Paper.pdf, Paper.docx
src/            limpieza, entrenamiento (finetune.py), evaluación, examen de concordancia, ciclo automático, servicio (servicio/api.py)
scripts/        normalización del dataset, exportación de pares extra, pruebas del apóstrofo
configs/        hiperparámetros: experimentos.json (principal, ablaciones, semillas)
notebooks/      exploracion_resultados.ipynb (tablas, concordancia y errores desde results/), demo_colab.ipynb (inferencia con los pesos) y entrenar_colab_t4.ipynb (entrenamiento en T4)
data/           NO versionado; ver data/README.md (fuentes, licencias y cómo reconstruirlo)
results/        resultados de cada experimento: test.json, historial.json, predicciones, concordancia, FLORES, figuras y logs
tests/          pruebas unitarias del normalizador y de la construcción de literatura central
app/android/    cliente Android (Java) del servicio
deploy/         nginx, systemd y scripts de despliegue (AlmaLinux)
release/        APK firmados
```

## Reproducir

Probado con Python 3.14.2, Windows 11, CUDA 12.8 y una GPU NVIDIA RTX PRO 5000 Blackwell de 48 GB.

```bash
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # ajusta a tu CUDA
pip install -r requirements.txt
```

**1. Datos** (detalle y licencias en [`data/README.md`](data/README.md)):

```bash
python -X utf8 src/limpiar_datos.py
python -X utf8 src/marcar_religioso.py
python -X utf8 src/preparar_particiones.py
python -X utf8 scripts/normalizar_dataset_quechua.py
python -X utf8 scripts/exportar_extra_normalizado.py
```

**2. Línea base, modelo principal, ablaciones y semillas** (unas 1.6 h de GPU en total para estas 5 corridas; el proyecto completo, con todas las variantes, suma unas 13 h de entrenamiento):

```bash
python -X utf8 src/baseline_nllb.py
python -X utf8 src/correr_experimentos.py                # lee configs/experimentos.json
```

**3. Modelo final y variantes.** Todas usan `--epocas 3 --batch 32 --lr 3e-5`; solo cambian estos argumentos de `src/finetune.py`:

| Experimento | Argumentos adicionales |
|---|---|
| `exp_norm15k_s0`, `s1`, `s2` | `--semilla N --extra-archivo extra_norm15k` |
| `exp_norm2k_s0`, `s1`, `s2` | `--semilla N --extra-archivo extra_norm2k` |
| `exp_norm15k_central_s0` | `--extra-archivo extra_norm15k_central` |
| `exp_filtrado15_s0` | `--train-archivo train_filtrado_15` |
| `exp_aleatorio15_s0` | `--train-archivo train_aleatorio_15` |
| `exp_bt30_s0`, `exp_bt40_s0` | `--bt-archivo bt_rt30` o `bt_rt40` |
| `auto_it1` | `--bt-archivo auto_auto_it1 --extra-archivo extra_auto_it1` (lo genera `src/auto_ciclo.py`) |

```bash
python -X utf8 src/finetune.py --nombre exp_norm15k_s0 --semilla 0 --epocas 3 --batch 32 --lr 3e-5 --extra-archivo extra_norm15k
```

Nota: `results/exp_norm15k_s0/` no tiene `historial.json`; su `test.json` se obtuvo con `src/evaluar_test_guardado.py` sobre los pesos ya entrenados (el registro del entrenamiento está en `results/logs/exp_norm15k_s0.log`).

**4. Evaluaciones y tablas del artículo:**

```bash
python -X utf8 src/evaluar_flores.py --modelo models/exp_norm15k_s0/mejor --nombre exp_norm15k_s0_def --split devtest                  # FLORES, decodificador por defecto
python -X utf8 src/evaluar_flores.py --modelo models/exp_norm15k_s0/mejor --nombre exp_norm15k_s0_nr3 --split devtest --no-repeat 3   # sin repetir 3-gramas
python -X utf8 src/examen_concordancia.py --modelo exp_norm15k_s0=models/exp_norm15k_s0/mejor   # examen de concordancia
python -X utf8 src/analisis.py                                                                  # results/tablas.md y figuras
python -X utf8 scripts/medir_apostrofo.py && python -X utf8 scripts/prueba_apostrofo.py         # apóstrofo
python -X utf8 paper/figuras.py                                                                  # arquitectura y concordancia por persona
python -X utf8 paper/construir.py                                                                # Paper.tex, Paper.pdf y Paper.docx (requiere pdflatex)
```

Cada cifra del artículo se lee de `results/` al compilarlo (`paper/resultados.py`); no hay números escritos a mano en el texto salvo los de las tablas I y II (limpieza y particiones), que están en `paper/contenido.py` copiados del reporte `data/limpio/reporte_limpieza.json`.

### En una GPU de 16 GB (Colab T4)

`src/finetune.py` usa fp16 con escalado de gradiente si la GPU no admite bf16, y acepta `--acumulacion N` (divide cada lote en N micro-lotes), `--checkpointing` y `--limite-vram-gb`.
Con `--precision fp16 --acumulacion 4 --limite-vram-gb 15` el entrenamiento llega a un pico de 13.2 GB. `notebooks/entrenar_colab_t4.ipynb` lo ejecuta en Colab.

Estado de esta verificación: entrenamos el modelo principal bajo ese tope de 15 GB y en fp16 en **nuestra GPU** (que emula la memoria de una T4, no su velocidad).
**No lo ejecutamos en una T4 real**, así que no tenemos tiempos medidos en esa tarjeta. Resultado (`results/t4_emulada_s0/`): pico de 13.2 GB (frente a 19.5 GB con bf16), chrF++ de 38.6 (es→qu) y 33.3 (qu→es) frente a 38.7 y 33.3 con bf16, y 0.99 h de entrenamiento en lugar de 0.43 h.

## Modelo, servicio y aplicación

- **Pesos:** https://huggingface.co/sergiomc12/nllb600m-es-quy (público; la ficha trae el modo de uso, los resultados y las limitaciones). Es el modelo final `exp_norm15k_s0`, idéntico al de `models/` (misma suma SHA256). Para volver a subirlos: `python scripts/subir_pesos_hf.py`. Licencia: NLLB-200 es CC-BY-NC 4.0, por lo que el modelo ajustado es solo para uso no comercial.
- **Demo sin entrenar:** `notebooks/demo_colab.ipynb` carga los pesos y traduce (Colab con GPU T4 o CPU).
- **Servicio:**

  ```bash
  MODELO_DIR=models/exp_norm15k_s0/mejor uvicorn src.servicio.api:app --port 8088
  curl -s localhost:8088/traducir -H 'content-type: application/json' -d '{"texto":"Buenos días","origen":"es","destino":"qu"}'
  ```

  El servicio prohíbe repetir 3-gramas al decodificar. Está desplegado en `https://quechua.racinghorse.net` (CPU, unos 6 s por oración).
- **Despliegue:** `bash deploy/desplegar.sh quechua.racinghorse.net <correo>` (AlmaLinux: systemd, nginx y certbot; ver `deploy/`).
- **Aplicación Android:** `cd app/android && gradle assembleRelease -PAPI_URL=https://quechua.racinghorse.net`. La dirección del servicio queda fija al compilar.

## Limitaciones y uso responsable

- Prueba de la misma fuente que el entrenamiento, sin evaluación de hablantes: los números no miden el uso real.
- El filtro de dominio es una lista de palabras: conservador, descarta también oraciones generales y puede dejar pasar texto de esas publicaciones sin esas palabras.
- La prueba contiene pares desalineados y encabezados (2 %); no se auditó con un hablante.
- Una prueba pequeña con quechua central (21 pares) indica que el rendimiento cae fuera de la variedad de entrenamiento.
- El traductor puede inventar contenido (repeticiones, pares desalineados): no usar en textos médicos, legales o administrativos sin revisión humana.
- Licencias: el conjunto de Ayacucho no declara licencia y parte proviene de publicaciones con derechos de autor; NLLB-200 es CC-BY-NC. Uso académico y no comercial.

## Equipo y contribuciones

Todos: pruebas del modelo, generación del conjunto de datos y normalización del conjunto de datos. Además:

| Integrante | Modelos y pruebas |
|---|---|
| Flavio César Amayo Gamboa | modelo principal y ablación del tamaño; pruebas de español a quechua |
| Gustavo Adolfo Huarcaya Delgado | modelos sin 15 % de los pares y normalizado 2k; pruebas de quechua a español |
| Jhordy Joel Maximiliano Cusi | retrotraducción y ciclo automático; pruebas de rendimiento del modelo |
| Sergio Mondragon Contreras | normalizado 15k y con literatura central, modelo final; optimización de modelos |

## Uso de asistentes de IA

Se usó Claude Code (Anthropic) para escribir y depurar scripts, el servicio, la aplicación y borradores de redacción. El equipo ejecutó los experimentos y
revisó los textos; todas las cifras provienen de esas ejecuciones.
