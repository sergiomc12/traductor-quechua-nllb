## Resultados en test (media ± desv. est. entre semillas)

| Sistema | Direccion | chrF++ | BLEU |
|---|---|---|---|
| zero-shot NLLB-600M (linea base) | es->qu | 26.4 | 3.2 |
| zero-shot NLLB-600M (linea base) | qu->es | 27.6 | 9.7 |
| principal (100 % de los datos) (n=3) | es->qu | 38.4 ± 0.4 | 9.8 ± 0.2 |
| principal (100 % de los datos) (n=3) | qu->es | 33.3 ± 0.1 | 14.7 ± 0.0 |
| ablacion: 50 % de los datos (n=1) | es->qu | 36.3 | 8.6 |
| ablacion: 50 % de los datos (n=1) | qu->es | 32.2 | 13.7 |
| ablacion: 25 % de los datos (n=1) | es->qu | 34.9 | 7.9 |
| ablacion: 25 % de los datos (n=1) | qu->es | 31.5 | 13.0 |

## Costo computacional

| Experimento | GPU | Precision | Pico VRAM (GB) | Horas GPU (entren.) | Mejor chrF++ valid |
|---|---|---|---|---|---|
| principal_s0 | NVIDIA RTX PRO 5000 Blackwell | torch.bfloat16 | 19.5 | 0.428 | 36.8 |
| principal_s1 | NVIDIA RTX PRO 5000 Blackwell | torch.bfloat16 | 19.5 | 0.411 | 36.5 |
| principal_s2 | NVIDIA RTX PRO 5000 Blackwell | torch.bfloat16 | 19.4 | 0.423 | 37.1 |
| ablacion_datos50_s0 | NVIDIA RTX PRO 5000 Blackwell | torch.bfloat16 | 19.5 | 0.203 | 35.4 |
| ablacion_datos25_s0 | NVIDIA RTX PRO 5000 Blackwell | torch.bfloat16 | 18.7 | 0.112 | 33.9 |

Total horas GPU (entrenamiento + test): **1.72 h**

## Analisis de errores (principal_s0)

### chrF++ por longitud de la referencia (palabras)

| Direccion | 1-5 | 6-10 | 11-20 | 21-40 | >40 |
|---|---|---|---|---|---|
| es->qu | 40.2 (n=1103) | 39.0 (n=1020) | 37.3 (n=493) | 40.5 (n=44) | 26.2 (n=2) |
| qu->es | 38.1 (n=621) | 36.3 (n=665) | 32.7 (n=996) | 31.7 (n=369) | 35.4 (n=11) |

### Ratio de longitud hipotesis/referencia (mediana) y % de salidas vacias o identicas a la fuente

| Direccion | Mediana ratio | % vacias | % copia de la fuente |
|---|---|---|---|
| es->qu | 0.99 | 0.0 | 0.5 |
| qu->es | 0.89 | 0.0 | 0.2 |

### es->qu: recall de sufijos (palabras de la referencia que terminan en el sufijo y aparecen exactas en la hipotesis)

| Sufijo | Apariciones en ref. | Recall |
|---|---|---|
| -pa | 889 | 44.2% |
| -taq | 228 | 32.5% |
| -sqa | 219 | 27.4% |
| -pi | 696 | 23.7% |
| -kuna | 341 | 23.5% |
| -chik | 325 | 23.1% |
| -man | 341 | 17.3% |
| -paq | 535 | 16.3% |
| -mi | 817 | 15.8% |
| -ku | 710 | 15.2% |
| -wan | 431 | 13.0% |
| -manta | 430 | 10.7% |

### Ejemplos de error (chrF++ por frase mas bajo con ref. de 6-30 palabras)

- **es->qu** (chrF++ 0)
  - src: [Reconocimiento de la página 2]
  - ref: 29 DE DICIEMBRE 2008 – ​ 4 DE ENERO 2009
  - hip: [23 kaq paginapi suti rikuylla kaq willakuy]
- **qu->es** (chrF++ 1)
  - src: 16 - 22 DE JUNIO 2008
  - ref: 26 “Me guía por los senderos trillados de la justicia ”
  - hip: 16 - 22 DE JUNIO DE 2008
- **es->qu** (chrF++ 1)
  - src: [Reconocimiento de la página 2]
  - ref: 28 DE JULIO 2008 – 3 DE AGOSTO 2008
  - hip: [23 kaq paginapi suti rikuylla kaq willakuy]
- **es->qu** (chrF++ 1)
  - src: 26 “Me guía por los senderos trillados de la justicia ”
  - ref: 16 - 22 DE JUNIO 2008
  - hip: 26  Allin rurayniyoq ñannintam pusani 
- **es->qu** (chrF++ 2)
  - src: PÁGINA 13
  - ref: 21 - 27 DE JULIO 2008
  - hip: 13 PAGINAPI
- **es->qu** (chrF++ 3)
  - src: Pero tiene otro motivo más poderoso para hacerlo: su propio nombre.
  - ref: 3: 14; 1 Sam. 12: 22).
  - hip: Ichaqa aswan allinmi sutiyoq kayninqa.