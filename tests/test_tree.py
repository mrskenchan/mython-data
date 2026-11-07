import unittest
import pandas as pd
import numpy as np
from analisis.tree_engine import (
    entrenar_arbol,
    listar_modelos_guardados,
    eliminar_modelo_guardado
)


class TestTreeEngine(unittest.TestCase):

    def setUp(self):
        # Dataset de prueba sintético con 20 filas balanceadas
        self.df = pd.DataFrame({
            "edad": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49, 51, 53, 57, 58, 63],
            "ingreso": [1500, 1800, 4500, 5200, 4800, 6100, 5900, 7000, 7200, 6800, 1200, 2200, 2100, 2400, 5000, 5300, 5500, 6200, 6400, 7500],
            "hijos": [0, 0, 2, 3, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 2, 2, 3, 1, 1, 0],
            "educacion": ["Media", "Media", "Superior", "Superior", "Superior", "Postgrado", "Superior", "Postgrado", "Postgrado", "Superior",
                          "Media", "Media", "Media", "Superior", "Superior", "Superior", "Superior", "Postgrado", "Postgrado", "Postgrado"],
            "compra": ["No", "No", "Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si",
                       "No", "No", "No", "No", "Si", "Si", "Si", "Si", "Si", "Si"]
        })

    def test_entrenar_arbol_exito(self):
        res = entrenar_arbol(
            df=self.df,
            target="compra",
            features=["edad", "ingreso", "hijos", "educacion"],
            max_depth=3,
            criterion="gini"
        )
        self.assertIn("accuracy", res)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1_score", res)
        self.assertIn("confusion_matrix", res)
        self.assertIn("tree_svg", res)
        self.assertIn("importancia_variables", res)

        self.assertGreaterEqual(res["accuracy"], 0.70)
        self.assertTrue(res["tree_svg"].startswith("<svg") or "svg" in res["tree_svg"].lower())
        self.assertEqual(len(res["classes"]), 2)

    def test_entrenar_arbol_criterio_entropia(self):
        res = entrenar_arbol(
            df=self.df,
            target="compra",
            features=["edad", "ingreso"],
            max_depth=4,
            criterion="entropy"
        )
        self.assertGreater(len(res["tree_svg"]), 100)
        self.assertIsInstance(res["importancia_variables"], dict)

    def test_target_invalido(self):
        with self.assertRaises(ValueError):
            entrenar_arbol(self.df, target="columna_inexistente", features=["edad", "ingreso"])

    def test_features_vacias(self):
        with self.assertRaises(ValueError):
            entrenar_arbol(self.df, target="compra", features=[])

    def test_persistencia_modelo(self):
        res = entrenar_arbol(
            df=self.df,
            target="compra",
            features=["edad", "ingreso"],
            max_depth=2
        )
        model_id = res["model_id"]
        lista = listar_modelos_guardados()
        ids = [m["model_id"] for m in lista]
        self.assertIn(model_id, ids)

        # Prueba de eliminación
        eliminado = eliminar_modelo_guardado(model_id)
        self.assertTrue(eliminado)


if __name__ == "__main__":
    unittest.main()
