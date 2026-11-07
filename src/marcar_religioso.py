"""Agrega la columna `religioso` a data/limpio/ayacucho/*.tsv (conserva `biblico`).

El filtro original de 5 palabras (`biblico`) dejaba pasar ~15 puntos porcentuales de texto de las revistas
de una sola organización: vocabulario propio, citas biblicas (Is. 55: 9) y formas quechuas con sufijos (Diospa, Jehovaq).
`religioso` es deliberadamente conservador: ante la duda marca la fila como religiosa, de modo que el
subconjunto NO religioso sea lo mas limpio posible.

  python -X utf8 src/marcar_religioso.py
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
NOMBRES = r"Dios\w*|Jehov\w*|Jes[uú]s\w*|Crist\w*|Bibli\w*|b[ií]blic\w*|Testigos?|Reino|Se[nñ]or|Creador|Padre|Satan\w*|Esp[ií]ritu"
VOCAB = (r"ora(?:r|ci[oó]n|ciones)|orad|adora(?:r|ci[oó]n)|yupaycha\w*|alaban\w*|alabar|congregaci[oó]n|ancianos?|precursor\w*|predica\w*|"
         r"ministerio|bautis\w+|espiritual\w*|profec\w+|ap[oó]stol\w*|salmo\w*|pecad\w*|demonios?|resurrecci[oó]n|Armaged[oó]n|siervos?|"
         r"sacerdote|misionero|sagrad\w*|santo|santa|cristian\w*|fieles|fidelidad|verdadera adoraci[oó]n|mensaje del Reino|"
         r"No[eé]|Ad[aá]n|Abrah[aá]n|Mois[eé]s|Salom[oó]n|Jerusal[eé]n|Apocalipsis|Revelaci[oó]n|G[eé]nesis|Isa[ií]as|Jerem[ií]as|Daniel|Job|"
         r"Mateo|Corintios|Romanos|Hebreos|Filipenses|Proverbios|Hechos|Gehena|infierno|alma|eterna|paraíso|para[ií]so")
CITA = (r"\((?:lea|vea)\b|\b(?:Is|Sal|Mat|Luc|Juan|Hech|Rom|Heb|G[eé]n|[EÉ]x|Dan|Prov|Mar|G[aá]l|Efes|Filip|Col|Sant|Apoc|Rev|Cor|Tes|Tim|Sant|Pedro)\.?\s*\d+\s*:\s*\d+")
RE = re.compile(rf"\b(?:{NOMBRES}|{VOCAB})\b|{CITA}", re.I)


def marcar(df: pd.DataFrame) -> pd.Series:
    return df["es"].str.contains(RE) | df["qu"].str.contains(RE) | df["biblico"].astype(str).str.lower().eq("true")


if __name__ == "__main__":
    for sp in ("train", "valid", "test"):
        f = RAIZ / "data" / "limpio" / "ayacucho" / f"{sp}.tsv"
        df = pd.read_csv(f, sep="\t", keep_default_na=False)
        df["religioso"] = marcar(df)
        df.to_csv(f, sep="\t", index=False)
        print(f"{sp}: {int(df['religioso'].sum())} religiosas de {len(df)} ({100 * df['religioso'].mean():.1f} %)")
