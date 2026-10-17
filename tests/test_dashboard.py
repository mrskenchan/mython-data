import unittest
import pandas as pd
from analisis.dashboard_engine import obtener_datos_dashboard

class TestDashboardEngine(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "edad": [20, 25, 30, 35, 40, 45, 50],
            "salario": [1000, 1500, 2000, 2500, 3000, 3500, 4000],
            "puntuacion": [5.5, 6.2, 7.1, 7.8, 8.5, 9.0, 9.8],
            "ciudad": ["Santiago", "Valparaíso", "Santiago", "Concepción", "Santiago", "Valparaíso", "Concepción"],
            "activo": [True, True, False, True, False, True, True]
        })

    def test_obtener_datos_dashboard(self):
        res = obtener_datos_dashboard(self.df)
        self.assertTrue(res["valido"])
        self.assertEqual(res["kpis"]["total_filas"], 7)
        self.assertEqual(res["kpis"]["total_columnas"], 5)
        self.assertEqual(res["kpis"]["completitud_pct"], 100.0)

        # Verificar clasificación de columnas
        self.assertIn("edad", res["columnas_numericas"])
        self.assertIn("salario", res["columnas_numericas"])
        self.assertIn("puntuacion", res["columnas_numericas"])
        self.assertIn("ciudad", res["columnas_categoricas"])
        self.assertIn("activo", res["columnas_categoricas"])

        # Verificar correlación
        corr = res["matriz_correlacion"]
        self.assertGreaterEqual(len(corr["variables"]), 2)
        self.assertEqual(len(corr["valores"]), len(corr["variables"]))

        # Verificar estadísticas
        self.assertIn("salario", res["estadisticas"])
        self.assertEqual(res["estadisticas"]["salario"]["min"], 1000.0)
        self.assertEqual(res["estadisticas"]["salario"]["max"], 4000.0)

    def test_df_vacio(self):
        res = obtener_datos_dashboard(None)
        self.assertFalse(res["valido"])
        self.assertEqual(res["kpis"]["total_filas"], 0)


if __name__ == "__main__":
    unittest.main()
