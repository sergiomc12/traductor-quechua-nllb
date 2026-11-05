# Traductor español ↔ quechua (Ayacucho) — MIA-203, Examen Final

Fine-tuning bidireccional de **NLLB-200-distilled-600M** (`quy_Latn`) con datos limpiados y repartidos sin fuga,
evaluado contra una línea base zero-shot, con ablación del sesgo bíblico (publicaciones de una sola organización), análisis de errores,
servicio HTTP, app Android y despliegue en nginx.

Resultados y tablas: `results/tablas.md` (se regenera con un comando, ver abajo). Detalle del proceso: `AGENTS.md`.

## Estructura
```
src/            limpiar_datos.py, baseline_nllb.py, finetune.py, correr_experimentos.py, analisis.py, servicio/api.py
configs/        experimentos.json (hiperparámetros y semillas de cada experimento)
data/           NO versionado: ver "Datos"
notebooks/      demo_colab.ipynb (inferencia de principio a fin)
results/        tablas, figuras y logs que aparecen en el artículo
app/android/    cliente Android (Java) del servicio
deploy/         nginx, systemd y scripts de despliegue
paper/ slides/  artículo IEEE y diapositivas
```

## Reproducir
```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # ajusta a tu CUDA
pip install -r requirements.txt

# 1. Datos (descargar de Hugging Face a data/, ver "Datos") y limpieza/reparto sin fuga
python -X utf8 src/limpiar_datos.py
# 2. Línea base zero-shot
python -X utf8 src/baseline_nllb.py
# 3. Entrenar + ablaciones + semillas (configs/experimentos.json). ~3 h en RTX PRO 5000
python -X utf8 src/correr_experimentos.py
# 4. Tablas y figuras del artículo
python -X utf8 src/analisis.py
```
Semillas fijas (0, 1, 2). Cada corrida guarda `results/<nombre>/{historial,test}.json` y `test_predicciones.tsv`.

**Colab T4:** `finetune.py` elige fp16 + GradScaler si la GPU no soporta bf16; con 16 GB usa `--batch 16`.
Para una corrida corta: `python src/finetune.py --nombre demo --max-pasos 500 --epocas 1 --n-test 200`.
Sin entrenar: `notebooks/demo_colab.ipynb` carga los pesos y traduce.

## Datos
| Dataset (Hugging Face) | Variedad | Licencia |
|---|---|---|
| `somosnlp-hackathon-2022/spanish-to-quechua` | Ayacucho (`quy`) — **usado** | no declarada (origen mixto, ~25-34 % publicaciones de una sola organización) |
| `pollitoconpapass/cuzco-quechua-translation-spanish` | Cusco (`quz`) — descartado (no se mezclan variedades) | no declarada |

Descargar a `data/hf_somosnlp_spanish_to_quechua/` y `data/hf_pollitoconpapass_cuzco/` (parquet).
Tras la limpieza: 122 802 pares (109 807 / 6 877 / 6 118 en train/valid/test), 0 filas de fuga entre particiones.

## Pesos
Enlace: **PENDIENTE** (subir `models/principal_s0/mejor` a Hugging Face o Drive y pegar aquí). Licencia: NLLB-200 es CC-BY-NC 4.0 →
uso no comercial.

## Servicio, app y despliegue
```bash
MODELO_DIR=models/principal_s0/mejor uvicorn src.servicio.api:app --port 8088
curl -s localhost:8088/traducir -H 'content-type: application/json' -d '{"texto":"Buenos días","origen":"es","destino":"qu"}'
bash deploy/desplegar.sh traductor.<dominio> <correo>      # sube a la VM, systemd + nginx + certbot
cd app/android && gradle assembleRelease -PAPI_URL=https://traductor.<dominio>
```

## Uso de asistentes de IA
Claude Code (Anthropic) se usó para escribir/depurar scripts, el servicio, la app y borradores de redacción;
todos los resultados provienen de ejecuciones propias del equipo.
