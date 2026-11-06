# Literatura VI ciclo — quechua central (*Markanchikpa willakuyninkuna*)

**Archivo:** `docs/literaratura-quechua-central.pdf` · 164 páginas · 9,8 MB · MINEDU (DIGEIBIRA/DEIB), 1.ª ed. 2021, impreso en enero de 2022 (tiraje 11 628). Cuaderno de lectura (*Ñawinchana maytu*) para primaria.

## Qué contiene
Antología de cuentos, relatos y textos literarios **íntegramente en quechua** (por ejemplo «Atuqwan yukis»). Una pieza lleva la nota «Qichwaman tikrachiq» («traducido al quechua») con su original en español, «El zorro del cielo» de César Itier, pero el original no está en el PDF.

## Variedad e idioma
Quechua central, con el alfabeto unificado (*shumaq*, *-shi*, *-chaw*, *ts*). Casi no hay español: 41 líneas, que son créditos y paratexto.

## Lo que medí
- **≈ 29 800 palabras**, el mayor volumen de quechua del lote. Texto limpio: 0 caracteres rotos y 21 letras sueltas en 30 000 tokens.
- Solo ≈ 28 % de sus palabras están en el vocabulario de nuestros datos de Ayacucho (frente al 87 % del conjunto de prueba de Ayacucho): **la distancia de variedad es enorme**.

## Utilidad como dataset
| Uso | Valoración |
|---|---|
| Corpus paralelo | **Nulo**: no incluye el original en español |
| Quechua central monolingüe (modelo de lenguaje, preentrenamiento continuo) | Media |
| Retrotraducción (traducir qu→es con un modelo y entrenar con eso) | Posible, pero LCT-EHU 2023 reporta que los datos sintéticos de baja calidad empeoran el resultado |
| Entrenamiento del modelo actual (Ayacucho) | Nula o negativa |

## Riesgos
- Sin licencia declarada; marca «PROHIBIDA SU VENTA» (distribución gratuita del Estado, no es una licencia abierta).
- Registro literario/narrativo, distinto del uso cotidiano.
