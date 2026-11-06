# Vocabulario pedagógico quechua central (*Yachachinapaq shimikuna – Chawpin Qichwa*)

**Archivo:** `docs/vocabulario-pedagogico-quechua central.pdf` · 164 páginas · 14,9 MB · MINEDU (DIGEIBIRA/DEIB), 1.ª ed. 2021 (tiraje 6 900).

## Qué contiene
Diccionario escolar bilingüe en tres partes, más una guía de escritura:
- **Parte I, quechua central → castellano**, por áreas (Comunicación, Matemática, Personal social y Ciencia y tecnología). Cada entrada trae la palabra, la categoría gramatical (`sh.` sustantivo, `r.` verbo, `pat.`…), la glosa en español y **una oración de ejemplo en quechua con su traducción al español entre comillas** (`'…'`).
- **Parte II, castellano → quechua central** (índice inverso, p. 111–140).
- **Parte III, palabras por campos semánticos** (animales, plantas, anatomía; p. 141–158).
- Anotaciones sobre el alfabeto unificado y la escritura del quechua central.

Formato de una entrada: `paqchiy r. aplaudir.` / *Llapanchik paqchirishun.* / `'Aplaudamos todos'.`

## Variedad e idioma
Quechua **central (Quechua I)**, no ayacuchano: marcadores como *shumaq*, *-chaw*, *-shi*. Español peruano escolar.

## Lo que medí
- ≈ 24 000 palabras; ≈ 830 traducciones entre comillas; ≈ 880 cabeceras de entrada.
- Extrayendo en orden de lectura (`pdftotext -enc UTF-8`) salen **323 pares oración quechua–oración español** a la primera, con una mediana de 6 palabras en quechua y 7 en español. Con un extractor por entrada se podría llegar hacia ~830, y las glosas aportan unos ~880 pares palabra–traducción más.
- El diseño a dos columnas mezcla líneas con `pdftotext -layout`; hay que usar el orden de lectura o recortar por columna.
- Ruido típico: la glosa del encabezado queda pegada a la oración quechua (`bocio, papera. Shullka wawqiimi…`) y hay algunas traducciones truncadas.

## Utilidad como dataset
| Uso | Valoración |
|---|---|
| Entrenamiento supervisado es↔qu (central) | **La mejor fuente de este lote**, pero solo ~0,3–0,8 mil pares |
| Entrenamiento del modelo actual (Ayacucho) | **Nula o negativa** (otra variedad) |
| Léxico bilingüe | Alta (≈ 880 palabras con ejemplos) |
| Conjunto de evaluación | Posible para un modelo central; muy pequeño |

## Riesgos
- **Derechos:** «Todos los derechos reservados. Prohibida la reproducción de este libro por cualquier medio». Hace falta autorización del MINEDU para redistribuirlo o usarlo en un producto.
- **Dominio:** frases escolares cortas, de registro simple; no sirven para texto largo.
