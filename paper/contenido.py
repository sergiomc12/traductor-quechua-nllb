"""Contenido del articulo (fuente unica para el PDF y el Word).

Bloques: ("sec", t) ("sub", t) ("p", t) ("bullets", [t]) ("eq", tex, plano) ("table", etiqueta, titulo, cabecera, filas)
         ("fig", ruta, titulo, etiqueta) ("pend", nota)
Los numeros de resultados se leen de results/ (ver resultados.py); si faltan, la seccion queda como pendiente.
"""
from __future__ import annotations

from pathlib import Path

TITULO = "Traducción neuronal bidireccional español–quechua mediante ajuste fino de NLLB-200"

AUTORES = [
    ("Flavio César Amayo Gamboa", "flavio.amayo.g@uni.pe"),
    ("Gustavo Adolfo Huarcaya Delgado", "gustavo.huarcaya.d@uni.pe"),
    ("Jhordy Joel Maximiliano Cusi", "jmaximilianoc@uni.pe"),
    ("Sergio Mondragon Contreras", "sergio.mondragon.c@uni.pe"),
]
AFILIACION = ["Maestría en Inteligencia Artificial", "FIIS – UNI", "Lima, Perú"]

PALABRAS_CLAVE = "traducción automática neuronal, quechua, lenguas de bajos recursos, NLLB, ajuste fino, chrF++"

REFERENCIAS = [
    "NLLB Team, \"No language left behind: Scaling human-centered machine translation,\" arXiv:2207.04672, 2022.",
    "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, \"Attention is all you need,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.",
    "O. Moreno, \"The REPUcs' Spanish–Quechua submission to the AmericasNLP 2021 shared task on open machine translation,\" in Proc. 1st Workshop on NLP for Indigenous Languages of the Americas, 2021, pp. 241–247.",
    "N. Ahmed, N. Flechas Manrique, and A. Petrović, \"Enhancing Spanish-Quechua machine translation with pre-trained models and diverse data sources: LCT-EHU at AmericasNLP shared task,\" in Proc. AmericasNLP Workshop, 2023, pp. 156–162.",
    "J. Asillo Congora, J. Santisteban, and R. Lazo Vasquez, \"UCSP submission to the AmericasNLP 2025 shared task,\" in Proc. 5th Workshop on NLP for Indigenous Languages of the Americas, 2025, pp. 84–91.",
    "W.-R. Chen and M. Abdul-Mageed, \"Improving neural machine translation of indigenous languages with multilingual transfer learning,\" arXiv:2205.06993, 2022.",
    "M. Mager, R. Bhatnagar, G. Neubig, N. T. Vu, and K. Kann, \"Neural machine translation for the indigenous languages of the Americas: An introduction,\" arXiv:2306.06804, 2023.",
    "A. L. Tonja, F. Balouchzahi, S. Butt, O. Kolesnikova, H. Ceballos, A. Gelbukh, and T. Solorio, \"NLP progress in indigenous Latin American languages,\" arXiv:2404.05365, 2024.",
    "B. Cueva Medina, G. F. Tuco Casquino, and J. Sulla-Torres, \"Development of a neural machine translation model optimized with BERT for translation from Quechua to Spanish,\" in 22nd LACCEI Int. Multi-Conf. for Engineering, Education, and Technology, 2024.",
    "J. E. Ortega, R. Castro Mamani, and K. Cho, \"Neural machine translation with a polysynthetic low resource language,\" Machine Translation, 2020, doi: 10.1007/s10590-020-09255-9.",
    "M. Popović, \"chrF++: words helping character n-grams,\" in Proc. 2nd Conf. on Machine Translation (WMT), 2017, pp. 612–618.",
    "M. Post, \"A call for clarity in reporting BLEU scores,\" in Proc. 3rd Conf. on Machine Translation (WMT), 2018, pp. 186–191.",
    "I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in Proc. Int. Conf. on Learning Representations (ICLR), 2019.",
]

# Descartes de la limpieza (reporte_limpieza.json, variedad Ayacucho)
DESCARTES = [
    ("Filas brutas (dos particiones originales unidas)", "128 434"),
    ("Vacía o de menos de 3 caracteres", "62"),
    ("Español idéntico al quechua", "22"),
    ("Más de 400 caracteres", "77"),
    ("Razón de longitud mayor a 4", "605"),
    ("Sin letras", "316"),
    ("Duplicado exacto", "4 550"),
    ("Filas limpias", "122 802"),
]


def secciones_previas() -> list[tuple]:
    """Introduccion, trabajos relacionados y metodologia: no dependen de los resultados."""
    return [
        ("sec", "Introducción"),
        ("p", "El quechua es una familia de lenguas con varios millones de hablantes en los Andes y es oficial en el Perú [3]. "
              "La variedad de Ayacucho (código ISO quy) es la base del quechua sureño estándar, que es el que usan las tareas compartidas de AmericasNLP [3], [4]. "
              "Es una lengua aglutinante: una palabra suele llevar varios sufijos de caso, persona y tiempo, de modo que el vocabulario observado crece rápido y cada forma aparece pocas veces. "
              "Hay pocos textos paralelos con el español, y el recurso más usado procede de publicaciones de una sola organización."),
        ("p", "Este trabajo aborda la traducción de oraciones en ambos sentidos, español a quechua y quechua a español. "
              "La entrada es una oración en un idioma y la salida es su traducción al otro. "
              "Medimos el éxito con chrF++ y BLEU sobre un conjunto de prueba que el modelo no vio ni en el entrenamiento ni en la selección de hiperparámetros."),
        ("p", "Usamos aprendizaje profundo porque existe un modelo multilingüe preentrenado, NLLB-200, que ya incluye el quechua ayacuchano (quy_Latn) [1]. "
              "Es un Transformer codificador-decodificador [2], y ajustarlo con unas cien mil oraciones es viable en una sola GPU. "
              "No entrenamos desde cero porque unas cien mil oraciones no bastan para ello."),
        ("p", "Los conjuntos públicos disponibles en Hugging Face tienen problemas que cambian los números reportados. "
              "En el conjunto de Ayacucho, el 7.3 % de las oraciones en español de validación y el 6.9 % de las de prueba ya estaban en entrenamiento. "
              "Más de la mitad de las filas tienen el estilo de esas publicaciones. "
              "También hay duplicados y pares mal alineados. Un traductor entrenado con ese material aprende sobre todo el estilo de esas publicaciones, "
              "y un promedio global sobre su conjunto de prueba no dice cuánto traduce fuera de él."),
        ("p", "Decidimos excluir ese material del entrenamiento, la validación y la prueba, y entrenar solo con lo que queda: 49 414 pares de entrenamiento. "
              "Es menos de la mitad de los datos, así que medimos también cuánto aporta el tamaño del conjunto."),
        ("p", "Las contribuciones son:"),
        ("bullets", [
            "Un conjunto de 122 802 pares español–quechua ayacuchano, limpiado y sin fuga entre particiones, y su subconjunto sin esas publicaciones, de 55 553 pares (49 414, 3 477 y 2 662 en entrenamiento, validación y prueba).",
            "Un modelo bidireccional, NLLB-200 de 600 millones de parámetros ajustado con tres semillas y comparado con el mismo modelo sin ajustar, y una ablación sobre el tamaño del entrenamiento (25 %, 50 % y 100 %).",
            "La comparación de diez modelos (selección de datos, retrotraducción, ciclo automático y conjunto normalizado) en la prueba propia, en FLORES-200 y en un examen de concordancia verbal.",
            "Un conjunto normalizado de verbos y léxico y el examen de concordancia, con el que el modelo final conjuga mejor verbos nuevos sin perder en la prueba.",
            "Un análisis de errores por longitud y sufijos, la dificultad del apóstrofo y de la voz, y el servicio y la aplicación Android que usan el modelo.",
        ]),
        ("p", "La Sección II resume trabajos previos, la III describe datos y método, la IV presenta los resultados, la V los discute y la VI concluye."),

        ("sec", "Trabajos relacionados"),
        ("p", "Sistemas para la tarea compartida de AmericasNLP. Moreno [3] ganó la pista sin datos de desarrollo en 2021: reunió corpus nuevos de poemas, canciones, léxicos y manuales, "
              "y preentrenó con español–inglés antes de ajustar para español–quechua. Su chrF de 34.6 es la referencia que cita el equipo de 2023 [4]. "
              "Ahmed et al. [4] añadieron textos literarios y legales y compararon un modelo preentrenado español–inglés con uno español–finés; el finés se eligió por ser aglutinante, "
              "y ese modelo rindió mejor, mientras que los datos sintéticos de baja calidad empeoraron los resultados. "
              "Asillo et al. [5] estudiaron si frases sintéticas generadas con un modelo de lenguaje grande, junto a un modelo preentrenado, mejoran la traducción del quechua."),
        ("p", "Transferencia multilingüe. Chen y Abdul-Mageed [6] ajustaron modelos preentrenados bilingües y multilingües para traducir del español a diez lenguas indígenas sudamericanas, "
              "y fijaron el estado del arte en cinco de los diez pares sin ampliar los datos de entrenamiento. Mager et al. [7] y Tonja et al. [8] resumen el estado del área y sus carencias de datos."),
        ("p", "Quechua a español. Ortega et al. [10] tradujeron del quechua al español con un modelo neuronal para una lengua polisintética y segmentación BPE guiada por sufijos. "
              "Cueva et al. [9] usaron un Transformer con LMfusion y segmentación PRPE, y construyeron una interfaz web de traducción."),
        ("p", "Este trabajo se diferencia de los anteriores. Parte de una auditoría de los datos abiertos y reparte las particiones sin fuga. "
              "Excluye esas publicaciones del entrenamiento y de la evaluación, y mide cuánto aporta el tamaño del conjunto que queda. "
              "Además entrega un modelo en ambos sentidos, con servicio y aplicación."),

        ("sec", "Metodología"),
        ("sub", "Datos"),
        ("p", "Usamos el conjunto spanish-to-quechua del hackathon SomosNLP 2022 en Hugging Face, variedad Ayacucho, con columnas es y qu. "
              "Reúne blogs, Tatoeba, datos de AmericasNLP 2021 y publicaciones de una sola organización. No declara licencia; la parte que proviene de esas publicaciones tiene derechos de autor, y lo que queda tras el filtro puede seguir incluyendo texto de esas publicaciones, como se discute en la Sección V. "
              "Descartamos el conjunto de Cusco, también en Hugging Face, porque el quechua es una familia de variedades con inteligibilidad limitada entre las lejanas, en [4] añadir datos de Cusco ayudó menos que añadir datos de Ayacucho, y NLLB solo tiene la etiqueta de Ayacucho."),
        ("p", "La limpieza une las particiones originales, aplica los filtros de la Tabla I y reparte de nuevo por grupos: "
              "todas las filas que comparten una oración en español o en quechua van a la misma partición, elegida con un hash del grupo. "
              "Después de esto, ninguna oración en español ni en quechua aparece en dos particiones. "
              "Cada fila se marca como proveniente de esas publicaciones o no. Un primer filtro de cinco palabras (Dios, Jehová, Biblia, Jesús, Cristo) marcó el 34.0 % de las filas, pero una revisión de muestras mostró que dejaba pasar material de las mismas publicaciones: vocabulario de las revistas (congregación, predicar, adoración), citas como (Is. 55: 9), nombres bíblicos y formas quechuas con sufijos como Diospa. "
              "El filtro final amplía la lista con ese vocabulario y con patrones de cita, y busca en ambos idiomas. Es conservador: ante la duda marca la fila como proveniente de ellas, y por eso descarta también algunas oraciones generales con palabras como alma o Padre, y puede dejar algo sin detectar. "
              "Marca el 55.0 % del entrenamiento, el 49.4 % de la validación y el 56.5 % de la prueba. Esas filas se excluyen de todas las particiones."),
        ("table", "tab:limpieza", "Resumen de la limpieza (variedad Ayacucho)", ["Paso", "Filas"], [list(f) for f in DESCARTES]),
        ("p", "Antes del filtro las particiones tenían 109 807, 6 877 y 6 118 pares. Después quedan 49 414 de entrenamiento, 3 477 de validación y 2 662 de prueba (Tabla II)."),
        ("table", "tab:particion", "Particiones antes y después del filtro de dominio",
         ["Partición", "Pares limpios", "Tras el filtro", "Excluido"],
         [["Entrenamiento", "109 807", "49 414", "55.0 %"], ["Validación", "6 877", "3 477", "49.4 %"],
          ["Prueba", "6 118", "2 662", "56.5 %"], ["Total", "122 802", "55 553", "54.8 %"]]),
        ("sub", "Modelo"),
        ("p", "Partimos de NLLB-200-distilled-600M [1], un Transformer [2] de codificador y decodificador con etiquetas de idioma en la entrada. "
              "Un solo modelo traduce en ambos sentidos: cada par genera dos ejemplos, uno de es a qu (etiquetas spa_Latn y quy_Latn) y otro de qu a es. "
              "Todos los parámetros se ajustan; no se congela ninguna capa ni se añaden adaptadores. "
              "La Figura 1 resume el flujo: la etiqueta de idioma y la tokenización entran al codificador de 12 capas, y el decodificador de 12 capas genera la traducción con la etiqueta del idioma de destino forzada como primer token."),
        ("fig", Path(__file__).resolve().parent.parent / "results" / "fig_arquitectura.png",
         "Arquitectura de NLLB-200-distilled-600M usada para ambas direcciones.", "fig:arquitectura"),
        ("sub", "Función de pérdida y entrenamiento"),
        ("p", "La pérdida es la entropía cruzada con forzado del maestro. Para un lote B de pares (x, y), con x la oración fuente e y la traducción de T tokens:"),
        ("eq", r"\mathcal{L}(\theta) = -\frac{1}{|B|}\sum_{(x,y)\in B}\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(y_t \mid y_{<t}, x\right).",
               "L(θ) = −(1/|B|) Σ(x,y)∈B (1/T) Σt=1..T log pθ(yt | y<t, x)"),
        ("p", "Optimizamos con AdamW [13], tasa de aprendizaje 3·10⁻⁵ y decaimiento de pesos 0.01, calentamiento lineal del 5 % de los pasos y decaimiento lineal después, "
              "recorte de gradiente a norma 1, lotes de 32 oraciones de una sola dirección y 3 épocas. La longitud máxima es de 128 tokens. "
              "Los pesos se mantienen en fp32 y el cálculo usa bf16 con autocast (fp16 con escalado de gradiente en GPU sin bf16, como la T4 de Colab). Cada época evalúa 500 oraciones de validación del conjunto filtrado con chrF++ medio de ambas direcciones, "
              "y se conservan los pesos de la mejor época. La prueba se evalúa una sola vez con esos pesos, con búsqueda en haz de ancho 4. "
              "Las semillas son 0, 1 y 2; cambian el orden de los lotes y las 500 oraciones de validación. La ablación usa una fracción aleatoria del entrenamiento (25 % y 50 %) con la semilla 0."),
        ("sub", "Línea base y métricas"),
        ("p", "La línea base es NLLB-200-distilled-600M sin ajustar (zero-shot) sobre el mismo conjunto de prueba filtrado. "
              "Como referencia publicada, el sistema ganador de 2021 obtuvo 34.6 de chrF en la prueba de AmericasNLP [3], [4]; no es comparable de forma directa porque el conjunto de prueba es otro. "
              "La métrica principal es chrF++ [11], más adecuada que BLEU para una lengua aglutinante, y reportamos también BLEU de sacreBLEU [12]. "
              "Los resultados se separan por dirección y se promedian entre semillas con su desviación estándar."),
        ("sub", "Modelos comparados"),
        ("p", "Además del modelo principal entrenamos variantes sobre la misma partición y con los mismos hiperparámetros, todas con la semilla 0. "
              "Dos quitan el 15 % de los pares de entrenamiento: una al azar y otra los de mayor pérdida bajo NLLB-200 sin ajustar, que sirve de juez independiente porque no vio nuestros datos y suele dar pérdida alta a los pares desalineados. "
              "Otras dos añaden pares retrotraducidos (español sintético y quechua real, solo en es→qu) y conservan los que, al traducirse de vuelta, alcanzan un chrF++ de al menos 30 o de al menos 40 frente al original. "
              "El ciclo automático (vuelta 1) añade pares de FLORES-200 dev, de RunaSimi y retrotraducciones dirigidas a los vacíos que detectó el diagnóstico. "
              "Las tres últimas variantes añaden el conjunto normalizado descrito abajo, completo (15k), reducido a verbos y léxico (2k), o completo junto con pares de literatura quechua central."),
        ("sub", "Conjunto normalizado y examen de concordancia"),
        ("p", "El conjunto normalizado parte de un archivo CSV de gran tamaño con oraciones quechua–español cuya procedencia y licencia no verificamos. "
              "Lo regeneramos con reglas: diez verbos conjugados en siete personas y tres tiempos, sustantivos, adjetivos y adverbios con glosas revisadas por el equipo, "
              "variantes de escritura tratadas como equivalentes (por ejemplo chaki y ch'aki) y sinónimos en español. "
              "Las filas dudosas o sin sentido se marcaron para revisión manual y no entran al entrenamiento. Resultan 25 181 filas, de las que usamos 15 373 (15k) o 2 679 (2k). "
              "Las reglas siguen el paradigma de Cusco y Chanka, que no validó ninguna persona hablante."),
        ("p", "Para medir si el modelo aprende la conjugación, y no solo memoriza, el examen de concordancia usa cinco verbos que no están en el conjunto normalizado (raíces llamka, waqa, sama, kawsa y qillqa) "
              "en las siete personas y tres tiempos. Cuenta un acierto cuando la forma verbal esperada aparece en la salida: 105 casos en es→qu y 95 en qu→es. "
              "La forma esperada sale de las mismas reglas, así que el examen mide si el modelo aprendió ese paradigma, no si coincide con el quechua ayacuchano estándar."),
    ]


TEXTO_DISPONIBILIDAD = (
    "Código, configuraciones y resultados: repositorio del proyecto, https://github.com/sergiomc12/traductor-quechua-nllb (etiqueta v1.0-final). Los pesos del modelo final están en https://huggingface.co/sergiomc12/nllb600m-es-quy. "
    "El modelo base NLLB-200 tiene licencia CC-BY-NC 4.0, por lo que el modelo ajustado se limita a uso no comercial. "
    "Se usó Claude Code (Anthropic) para escribir y depurar los scripts de limpieza, entrenamiento y análisis, el servicio, la aplicación Android y los borradores de redacción. "
    "El equipo ejecutó los experimentos y revisó los textos; todas las cifras provienen de esas ejecuciones."
)
