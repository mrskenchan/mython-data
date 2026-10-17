import unittest
import pandas as pd
from analisis.association import (
    preparar_datos_transaccionales,
    minar_reglas_asociacion,
    generar_reglas_mock
)

class TestAssociationRules(unittest.TestCase):

    def setUp(self):
        # Dataset sintético de transacciones de compras
        self.df_compras = pd.DataFrame({
            "pan": [True, True, True, False, True, True, False, True, True, True],
            "leche": [True, False, True, True, True, False, True, True, False, True],
            "mantequilla": [True, False, True, False, True, False, False, True, False, True],
            "categoria": ["A", "A", "B", "B", "A", "A", "B", "A", "B", "A"],
            "monto": [10.5, 5.0, 15.2, 4.0, 12.0, 6.5, 3.2, 14.0, 7.1, 11.8]
        })

    def test_preparar_datos_transaccionales(self):
        df_bool = preparar_datos_transaccionales(self.df_compras)
        self.assertFalse(df_bool.empty)
        # Verificar que todos los tipos sean booleanos
        for col in df_bool.columns:
            self.assertEqual(df_bool[col].dtype, bool)

    def test_minar_reglas_fpgrowth(self):
        res = minar_reglas_asociacion(
            self.df_compras,
            min_support=0.2,
            min_confidence=0.5,
            min_lift=1.0,
            algorithm="fpgrowth"
        )
        self.assertTrue(res["exito"])
        self.assertIsInstance(res["reglas"], list)
        self.assertGreater(len(res["reglas"]), 0)

        primera = res["reglas"][0]
        self.assertIn("antecedents", primera)
        self.assertIn("consequents", primera)
        self.assertIn("support", primera)
        self.assertIn("confidence", primera)
        self.assertIn("lift", primera)

    def test_minar_reglas_apriori(self):
        res = minar_reglas_asociacion(
            self.df_compras,
            min_support=0.2,
            min_confidence=0.5,
            min_lift=1.0,
            algorithm="apriori"
        )
        self.assertTrue(res["exito"])
        self.assertIsInstance(res["reglas"], list)

    def test_filtro_mock_cuando_df_none(self):
        res = minar_reglas_asociacion(
            None,
            min_support=0.05,
            min_confidence=0.60
        )
        self.assertTrue(res["exito"])
        self.assertGreater(len(res["reglas"]), 0)


if __name__ == "__main__":
    unittest.main()
