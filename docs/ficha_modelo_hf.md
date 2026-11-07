---
license: cc-by-nc-4.0
language:
- es
- qu
base_model: facebook/nllb-200-distilled-600M
library_name: transformers
pipeline_tag: translation
tags:
- translation
- quechua
- low-resource
- nllb
metrics:
- chrf
- bleu
---

# NLLB-200 600M ajustado para español ↔ quechua (modelo final, `exp_norm15k_s0`)

Un solo modelo para traducir **español → quechua** y **quechua → español**, obtenido por ajuste fino de
[`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) con la etiqueta `quy_Latn`.
Es el modelo final del proyecto de MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI FIIS, equipo 6).
Código, datos de partida, experimentos y artículo: https://github.com/sergiomc12/traductor-quechua-nllb (repositorio privado).

## Uso

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "sergiomc12/nllb600m-es-quy"
tok = AutoTokenizer.from_pretrained(REPO)
modelo = AutoModelForSeq2SeqLM.from_pretrained(REPO).eval()
COD = {"es": "spa_Latn", "qu": "quy_Latn"}

def traducir(texto, origen="es", destino="qu"):
    tok.src_lang = COD[origen]
    enc = tok(texto, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        gen = modelo.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(COD[destino]),
                              num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

print(traducir("Buenos días", "es", "qu"))
```

Se recomienda `no_repeat_ngram_size=3`: sin él, el modelo repite palabras en el 5 % de las traducciones es→qu de FLORES.

## Entrenamiento

- **Datos:** conjunto `somosnlp-hackathon-2022/spanish-to-quechua` (variedad Ayacucho), limpiado y repartido sin fuga entre particiones, sin el texto de tipo publicaciones de una sola organización (49 414 pares de entrenamiento), más 15 373 pares de un conjunto normalizado de verbos conjugados, sustantivos, adjetivos y adverbios generado con reglas.
- **Método:** ajuste fino de todos los parámetros, 3 épocas, lote de 32, tasa de aprendizaje 3·10⁻⁵, AdamW, bf16. Cada par genera un ejemplo en cada dirección.
- **Cómputo:** unas 0.5 h en una NVIDIA RTX PRO 5000 Blackwell de 48 GB.

## Resultados

chrF++ en una prueba de 2 662 pares (misma fuente que el entrenamiento, sin texto de tipo publicaciones de una sola organización):

| | es→qu | qu→es |
|---|---|---|
| NLLB-200 sin ajustar | 26.4 | 27.6 |
| Este modelo (3 semillas) | 38.6 ± 0.1 | 33.6 ± 0.1 |

Examen de concordancia verbal con verbos que el modelo no vio al entrenar: 57.7 ± 1.1 % (es→qu) y 79.3 ± 2.4 % (qu→es), frente a 24.8 % y 53.7 % del modelo sin el conjunto normalizado.
Más detalle en el artículo.

## Limitaciones

- Entrenado con quechua de Ayacucho. En una prueba pequeña (21 pares) con quechua central la calidad fue menor.
- Ninguna persona hablante evaluó las traducciones, y la prueba proviene de la misma fuente que el entrenamiento: los números pueden ser optimistas para otros textos.
- La referencia del examen de concordancia sale de reglas propias (paradigma de Cusco y Chanka), no de una validación con hablantes.
- Puede inventar contenido o repetir palabras. No usar en textos médicos, legales o administrativos sin revisión humana.
- Las consonantes eyectivas se escriben con apóstrofo (p'unchaw, ch'aki); omitirlo cambia el sentido de algunas palabras y el modelo es sensible a ello.

## Licencia

NLLB-200 se distribuye con licencia CC-BY-NC 4.0, por lo que este modelo es **solo para uso no comercial**. El conjunto de entrenamiento de Ayacucho no declara licencia y parte de él proviene de publicaciones de una sola organización.
