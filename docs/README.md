# Análisis de los PDFs de `docs/` y su uso como datos de entrenamiento

Siete PDFs del **Ministerio de Educación del Perú** (DIGEIBIRA/DEIB, 2021–2022), todos en **quechua central** (Quechua I: *shumaq*, *-chaw*, *-shi*). Hay una ficha por documento en [`readmes/`](readmes/).

## Conclusión corta

**Ninguno sirve para mejorar el modelo actual** (español ↔ quechua ayacuchano). La variedad es otra, y los números lo muestran:

| Medida | Palabras que ya están en el vocabulario de entrenamiento de Ayacucho |
|---|---|
| Conjunto de prueba de Ayacucho (referencia) | **87,3 %** |
| Literatura (≈ 29 000 palabras) | 27,8 % |
| Saberes de los pueblos | 29,5 % |
| Cartillas, relatos, cancionero | 27–31 % |

Además, «shumaq» aparece 0 veces en nuestros datos de Ayacucho y «sumaq» 910. Mezclar variedades empeora el resultado: en LCT-EHU 2023 añadir datos de Cusco ayudó menos que añadir datos de Ayacucho, y NLLB solo tiene la etiqueta de Ayacucho (`quy_Latn`).

**Lo que sí hay:** muy poco texto paralelo (≈ 0,3–1 mil pares, casi todo en el vocabulario) y ≈ 46 000 palabras de quechua central monolingüe. Eso alcanza para un conjunto de evaluación pequeño o para estudiar la distancia entre variedades, no para entrenar un traductor de quechua central.

## Candidatos, de mejor a peor

| # | Documento | Páginas | Texto útil | Paralelo es↔qu | Valoración como dataset (0–5) |
|---|---|---|---|---|---|
| 1 | [Vocabulario pedagógico](readmes/vocabulario-pedagogico-quechua-central.md) | 164 | ≈ 24 000 palabras | **323 pares medidos** (hasta ~830 con extractor por entrada) + ~880 glosas | **3** (para un modelo central) · 0 para Ayacucho |
| 2 | [Cartillas de animales](readmes/cartillas-animales-quechua-central.md) | 50 | ≈ 2 700 palabras | ≈ 25–30 pares de párrafo (estimado) | **2** · 0 para Ayacucho |
| 3 | [Literatura](readmes/literaratura-quechua-central.md) | 164 | ≈ 29 800 palabras en quechua | No | **2** (monolingüe) |
| 4 | [Saberes de los pueblos](readmes/saberes-pueblos-quechua-central.md) | 162 | ≈ 16 300 palabras en quechua | No | **1,5** (monolingüe) |
| 5 | [Manual de escritura](readmes/manual-escritura-lengua-quechua-central.md) | 162 | ≈ 11 500 palabras en español, dañadas | No | **1** (documentación ortográfica) |
| 6 | [Historia y relatos](readmes/historia-relatos-quechua-central.md) | 44 | ≈ 1 400 palabras | No | **0,5** |
| 7 | [Cancionero](readmes/cancionero-quechua-central.md) | 36 | ≈ 1 000 palabras | No | **0,5** |

Medido: páginas, palabras, pares del vocabulario y cobertura de vocabulario. Estimado: pares de las cartillas y el techo de ~830 del vocabulario.

## Comparación de escala

| Fuente | Pares es↔qu |
|---|---|
| Datos actuales (Ayacucho, sin texto religioso) | **49 414** de entrenamiento |
| Mejor fuente de `docs/` | ≈ 323–830 |

Aunque se sumaran todos, serían ~1–2 % del volumen actual, y de otra variedad.

## Qué haría con ellos

1. **No añadirlos** al entrenamiento del modelo de Ayacucho.
2. Si el equipo quiere cubrir el quechua central, usarlos como **conjunto de evaluación** (vocabulario + cartillas, ≈ 0,3–0,9 mil pares) y comparar el modelo de Ayacucho sin entrenar con ellos. Sería un resultado útil para el paper: cuantifica la pérdida entre variedades.
3. Buscar la misma serie del MINEDU (DIGEIBIRA/DEIB) en **quechua sureño o chanka**, que es la variedad de nuestro modelo. Es una sugerencia que no verifiqué: habría que comprobar si esas ediciones existen y si son bilingües.
4. **Pedir autorización** antes de redistribuir o usar en un producto: el vocabulario y las cartillas declaran «todos los derechos reservados»; los demás no declaran licencia y llevan «prohibida su venta».

## Notas técnicas de extracción
- Usar `pdftotext -enc UTF-8`; sin esa opción en Windows salen «�» en las ñ y acentos.
- El vocabulario va a dos columnas: `-layout` mezcla líneas; el orden de lectura (sin `-layout`) las separa.
- El manual pierde las letras «j» y «v» en la capa de texto: requeriría OCR o reglas de reparación.
- Todos los PDFs tienen capa de texto (no son escaneos), pero tres son sobre todo ilustraciones: cartillas, relatos y cancionero.
