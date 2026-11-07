# Datos

Esta carpeta **no se versiona** (los datos pesan, y varios tienen derechos que no podemos redistribuir). Aquí se explica de dónde
sale cada archivo y cómo reconstruirlo.

## Fuentes

| Fuente | Uso | Licencia | Dónde queda |
|---|---|---|---|
| [`somosnlp-hackathon-2022/spanish-to-quechua`](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua) (Hugging Face), variedad Ayacucho | entrenamiento, validación y prueba | **no declarada**; origen mixto: blogs, Tatoeba, AmericasNLP 2021 y publicaciones de una sola organización (derechos de autor) | `data/hf_somosnlp_spanish_to_quechua/` |
| [`pollitoconpapass/cuzco-quechua-translation-spanish`](https://huggingface.co/datasets/pollitoconpapass/cuzco-quechua-translation-spanish) | descartado: es otra variedad (Cusco) | no declarada | `data/hf_pollitoconpapass_cuzco/` |
| [FLORES-200](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) (es y quy_Latn) | evaluación externa (devtest) y verificación del ciclo automático (dev) | CC BY-SA 4.0 | `data/flores/flores200_dataset/` |
| Wikipedia en quechua | retrotraducción (experimentos `exp_bt30` y `exp_bt40`) | CC BY-SA 3.0 | `data/bt/` |
| CSV de oraciones quechua–español de gran tamaño (`dataset_quechua_espanol_20_millones_palabras.csv`) | origen del conjunto normalizado | **procedencia y licencia no verificadas** | `data/` |
| Textos de literatura quechua central (MINEDU) | experimento `exp_norm15k_central` y 21 pares de prueba | derechos reservados: **no se redistribuyen** | `data/` |

Los pares del conjunto de Ayacucho llevan texto con derechos de autor (publicaciones de una sola organización). Por eso el repositorio no incluye los datos y
conviene mantenerlo privado o con acceso solo para quien lo evalúa.

## Cómo reconstruir `data/`

Todos los comandos se ejecutan desde la raíz del repositorio.

```bash
# 1. Descargar el conjunto de Ayacucho (parquet) a data/hf_somosnlp_spanish_to_quechua/
#    (con la CLI de Hugging Face: hf download somosnlp-hackathon-2022/spanish-to-quechua --repo-type dataset --local-dir data/hf_somosnlp_spanish_to_quechua)
# 2. Limpieza y reparto sin fuga entre particiones  -> data/limpio/ayacucho/{train,valid,test}.tsv
python -X utf8 src/limpiar_datos.py
# 3. Marcar texto de ese dominio                       -> columna `religioso` (nombre interno)
python -X utf8 src/marcar_religioso.py
# 4. Quedarse con el texto sin marcar                   -> data/limpio/ayacucho_general/{train,valid,test}.tsv
python -X utf8 src/preparar_particiones.py
```

Resultado esperado (Tablas I y II del artículo): 122 802 pares limpios; tras excluir ese texto quedan
49 414 de entrenamiento, 3 477 de validación y 2 662 de prueba.

### Conjunto normalizado (modelo final)

```bash
python -X utf8 scripts/normalizar_dataset_quechua.py           # -> data/dataset_quechua_espanol_normalizado.csv (25 181 filas)
python -X utf8 scripts/exportar_extra_normalizado.py           # -> data/auto/extra_norm15k.tsv (15 373) y extra_norm2k.tsv (2 679)
```

Se comprobó que estos dos comandos reproducen, byte a byte, los archivos con los que se entrenaron `exp_norm15k` y `exp_norm2k`.
`data/auto/extra_norm15k_central.tsv` (modelo con literatura central) se genera con `--central`, pero el archivo que se usó al entrenar
difiere en 4 pares del que produce hoy el script, porque la literatura central se reconstruyó después con otra corrección de comillas.

### Otras particiones de los experimentos

| Archivo | Cómo se obtiene |
|---|---|
| `data/limpio/ayacucho_general/train_filtrado_15.tsv` | `python -X utf8 src/puntuar_pares.py` y luego `--quitar 15` |
| `data/limpio/ayacucho_general/train_aleatorio_15.tsv` | `python -X utf8 src/preparar_particiones.py --aleatorio` (equivalente, **no idéntico** al usado: el código original no se conservó) |
| `data/bt/bt_rt30.tsv`, `bt_rt40.tsv` | `python -X utf8 src/preparar_bt.py --n 40000` y `python -X utf8 src/seleccionar_bt.py --umbrales 30 40` |
| `data/flores/` | descargar el tar.gz de FLORES-200 y descomprimirlo en `data/flores/` |
