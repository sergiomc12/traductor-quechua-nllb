# Ciclo automatico: resumen

Actualizado: 2026-11-06 06:26:02

Mejor modelo actual: `D:\Backend\traductorquechua\models\auto_it1\mejor` (seleccion 34.4)

Seleccion = media de chrF++ en {valid propia, FLORES dev-check} x {es→qu, qu→es}. FLORES devtest y la prueba propia solo se registran.

| Vuelta | Config | Seleccion | Valid propia | FLORES devtest es→qu / qu→es | Prueba propia es→qu / qu→es | Decision |
|---|---|---|---|---|---|---|
| base (principal_s0) | modelo actual | 32.6 | 37.7 | 26.3 / 29.9 | 39.0 / 33.3 | referencia |
| 1 | FLORES dev + RunaSimi + retrotraduccion dirigida a los vacios | 34.4 | 37.8 | 30.1 / 32.2 | 38.8 / 33.4 | ACEPTADO |
| 2 | se acepto la vuelta previa: mas retrotraduccion dirigida (lote nuevo) | 34.03 | 37.45 | 29.8 / 31.8 | 38.7 / 33.4 | rechazado |
| 3 | la retrotraduccion no ayudo: solo FLORES dev + RunaSimi (ablacion) | 34.4 | 37.8 | 30.1 / 32.2 | 38.8 / 33.4 | rechazado |

Para volver atras: el modelo original esta intacto en `models/principal_s0` y respaldado en `backup/v1_principal_s0` (etiqueta git v1.0-baseline-5nov).
