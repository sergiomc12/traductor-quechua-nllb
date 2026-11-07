import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import construir_dataset_literatura_central as c  # noqa: E402


class TestSegmentacion(unittest.TestCase):
    def test_divide_oraciones(self):
        self.assertEqual(c.oraciones("Hola amigo. ¿Cómo estás? Bien."), ["Hola amigo.", "¿Cómo estás?", "Bien."])

    def test_no_divide_en_abreviatura_minuscula(self):
        self.assertEqual(len(c.oraciones("Fue a la casa grande, y luego volvió.")), 1)


class TestAlineacion(unittest.TestCase):
    def test_igual_numero_alinea_uno_a_uno(self):
        modo, pares = c.pares_de_texto("Una dos. Tres cuatro.", "Uno dos. Tres cuatro.")
        self.assertEqual(modo, "exacta")
        self.assertEqual(len(pares), 2)

    def test_dp_une_oraciones_cuando_cuentas_difieren(self):
        a = ["Kay chaymi.", "Huk kutipaq riqsinchik."]
        b = ["Esto es eso, y lo conocemos una vez más."]
        modo, pares = c.pares_de_texto(" ".join(a), " ".join(b))
        self.assertEqual(modo, "dp")
        self.assertEqual(len(pares), 1)
        self.assertEqual(pares[0][1], b[0])


class TestFiltros(unittest.TestCase):
    def test_descarta_muy_cortos_y_desproporcionados(self):
        self.assertFalse(c.valido("Hola", "Hola amigo"))
        self.assertFalse(c.valido("a" * 100 + " b", "uno dos"))
        self.assertTrue(c.valido("Kay wasi hatun", "Esta casa es grande"))

    def test_quita_puntos_suspensivos_finales(self):
        self.assertEqual(c._PUNTOS_FINALES.sub("", "se fue..."), "se fue")

    def test_division_por_texto_es_estable(self):
        self.assertEqual(c.es_test("Mama"), c.es_test("Mama"))


if __name__ == "__main__":
    unittest.main()
