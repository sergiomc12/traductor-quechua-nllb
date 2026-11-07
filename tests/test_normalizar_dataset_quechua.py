import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import normalizar_dataset_quechua as n  # noqa: E402


class TestConjugacionQuechua(unittest.TestCase):
    def test_presente_todas_las_personas(self):
        formas = [n.verbo_quechua("miku", "presente", p) for p in n.PERSONAS]
        self.assertEqual(
            formas,
            ["mikuni", "mikunki", "mikun", "mikunchik", "mikuniku", "mikunkichik", "mikunku"],
        )

    def test_pasado_segunda_persona_no_usa_terminacion_de_yo(self):
        self.assertEqual(n.verbo_quechua("miku", "pasado", "2s"), "mikurqanki")
        self.assertEqual(n.verbo_quechua("miku", "pasado", "2p"), "mikurqankichik")

    def test_futuro(self):
        formas = [n.verbo_quechua("rima", "futuro", p) for p in n.PERSONAS]
        self.assertEqual(
            formas,
            ["rimasaq", "rimanki", "rimanqa", "rimasunchik", "rimasaqku", "rimankichik", "rimanqaku"],
        )


class TestConjugacionEspanol(unittest.TestCase):
    def test_regular_y_irregulares(self):
        self.assertEqual(n.verbo_espanol("comer", "pasado", "1s"), "comí")
        self.assertEqual(n.verbo_espanol("comer", "futuro", "1s"), "comeré")
        self.assertEqual(n.verbo_espanol("hacer", "pasado", "3s"), "hizo")
        self.assertEqual(n.verbo_espanol("hacer", "futuro", "2s"), "harás")
        self.assertEqual(n.verbo_espanol("saber", "presente", "1s"), "sé")
        self.assertEqual(n.verbo_espanol("dormir", "pasado", "3p"), "durmieron")

    def test_ustedes_y_nosotros_comparten_forma(self):
        self.assertEqual(n.verbo_espanol("bailar", "presente", "2p"), "bailan")
        self.assertEqual(n.verbo_espanol("bailar", "presente", "1pi"), "bailamos")
        self.assertEqual(n.verbo_espanol("bailar", "presente", "1pe"), "bailamos")


class TestFrasesCompletas(unittest.TestCase):
    def test_oracion_pasado_primera_persona(self):
        q, e = n.oracion("miku", "mikhuna", "ñuqa", "pasado", adverbio=None)
        self.assertEqual(q, "ñuqa mikhunata mikurqani")
        self.assertEqual(e, "Yo comí la comida")

    def test_articulo_de_agua_es_el(self):
        _, e = n.oracion("upya", "yaku", "pay", "presente", adverbio=None)
        self.assertEqual(e, "Él/ella bebe el agua")

    def test_concordancia_de_adjetivo_con_genero(self):
        self.assertEqual(n.frase_nominal("puka", "wasi", "base")[1], "casa roja")
        self.assertEqual(n.frase_nominal("puka", "mayu", "base")[1], "río rojo")

    def test_caso_gramatical(self):
        q, e = n.caso("hatun", "wasi", "pi", "base")
        self.assertEqual((q, e), ("hatun wasi-pi", "en la casa grande"))


class TestRevisionSemantica(unittest.TestCase):
    def test_sugerencias(self):
        self.assertEqual(n.sugerencia_par("puka", "pacha")[0], "sin_sentido")
        self.assertEqual(n.sugerencia_par("puka", "mama")[0], "dudoso")
        self.assertEqual(n.sugerencia_par("hatun", "wasi")[0], "ok")

    def test_decision_del_usuario_prevalece(self):
        original = n._DECISIONES
        try:
            n._DECISIONES = {("puka", "pacha"): "no", ("ch'aki", "yaku"): "si"}
            self.assertFalse(n._permitido("puka", "pacha"))
            self.assertTrue(n._permitido("ch'aki", "yaku"))  # 'si' pisa el bloqueo por regla
            self.assertEqual(n._motivo("ch'aki", "yaku"), "")
        finally:
            n._DECISIONES = original

    def test_saber_sin_complemento_queda_marcado(self):
        self.assertIn("saber", n._motivo_oracion("yacha", None))
        self.assertEqual(n._motivo_oracion("miku", "papa"), "")


class TestGeneracion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filas = n.generar()

    def test_sin_duplicados(self):
        claves = [(f["quechua"], f["espanol"]) for f in self.filas]
        self.assertEqual(len(claves), len(set(claves)))

    def test_quechua_no_tiene_dos_traducciones_distintas(self):
        vistos = {}
        for f in self.filas:
            vistos.setdefault(f["quechua"], set()).add(f["espanol"])
        ambiguos = {q: v for q, v in vistos.items() if len(v) > 1}
        self.assertEqual(ambiguos, {})

    def test_no_quedan_formas_agramaticales(self):
        texto = " ".join(f["espanol"] for f in self.filas)
        for mal in ("comeró", "comerré", "hacerré", "el/la", "(incl)"):
            self.assertNotIn(mal, texto)

    def test_la_segunda_persona_nunca_lleva_terminacion_de_yo(self):
        for f in self.filas:
            t = f["quechua"].split()
            if "qam" in t or "qamkuna" in t:
                self.assertFalse(t[-1].endswith(("ni", "rqani", "saq")), f["quechua"])

    def _buscar(self, quechua):
        return [f for f in self.filas if f["quechua"] == quechua]

    def test_futuro_de_segunda_persona_solo_con_paqarin(self):
        futuros = [f for f in self.filas if f["categoria"] == "Oración futuro" and f["persona"] in ("2s", "2p")]
        self.assertTrue(futuros)
        for f in futuros:
            self.assertTrue(f["quechua"].startswith("Paqarin "), f["quechua"])
        self.assertEqual(self._buscar("Paqarin qam takinki")[0]["espanol"], "Mañana tú cantarás")

    def test_adverbio_de_tiempo_concuerda_con_el_tiempo(self):
        esperado = {"Kunan": "presente", "Qayna": "pasado", "Paqarin": "futuro"}
        for f in self.filas:
            primera = f["quechua"].split()[0]
            if f["categoria"].startswith("Oración") and primera in esperado:
                self.assertEqual(f["tiempo"], esperado[primera], f["quechua"])

    def test_lexico_aclarado_por_el_usuario(self):
        self.assertEqual(self._buscar("Pasaq")[0]["espanol"], "Pasaré")
        self.assertEqual(self._buscar("Maypichus")[0]["espanol"], "¿Dónde está?")
        self.assertEqual(self._buscar("Allin puni")[0]["espanol"], "Muy bueno")
        self.assertEqual(self._buscar("Nisqanman hina")[0]["espanol"], "Según lo que dijo")
        self.assertEqual(self._buscar("Cheqapta")[0]["espanol"], "Verdad")

    def test_prefijos_validados_por_el_usuario(self):
        self.assertEqual(self._buscar("Maypichus puka allpa")[0]["espanol"], "¿Dónde está la tierra roja?")
        self.assertEqual(self._buscar("Maypichus huch'uy alpaka")[0]["espanol"], "¿Dónde está la alpaca pequeña?")
        self.assertEqual(self._buscar("Allin puni pay t'ikata qawan")[0]["espanol"], "Muy bueno él/ella mira la flor")
        self.assertEqual(self._buscar("Allin puni sumaq papa")[0]["espanol"], "Muy bueno papa hermosa")
        self.assertFalse([f for f in self.filas if f["quechua"].startswith("Maypichus ") and f["categoria"] != "Frase nominal"])

    def test_kinoa_y_chaki_con_apostrofe(self):
        self.assertTrue(any("kinoa" in f["quechua"] for f in self.filas))
        self.assertEqual(self._buscar("ch'aki")[0]["espanol"], "seco/seca; pie")
        self.assertEqual(self._buscar("chaki")[0]["espanol"], "pie")

    def test_sinonimos_en_columna_alternativas(self):
        fila = self._buscar("hoq'o wasi")[0]
        self.assertEqual((fila["espanol"], fila["espanol_alternativas"]), ("casa mojada", "casa húmeda"))
        self.assertIn("Verdaderamente", self._buscar("Cheqapta ñuqa takisaq")[0]["espanol_alternativas"])

    def test_variante_ayacucho_presente(self):
        self.assertTrue(any(f["dialecto"] == "ayacucho" and "mikuna" in f["quechua"] for f in self.filas))


if __name__ == "__main__":
    unittest.main()
