import unittest
import pandas as pd
from analisis.reporter import generar_dossier_ejecutivo

class TestReporter(unittest.TestCase):

    def setUp(self):
        self.meta = {
            "filename": "ventas.csv",
            "row_count": 500,
            "col_count": 8,
            "encoding": "utf-8",
            "separator": ",",
            "filesize_kb": 45.2,
            "filesize_mb": 0.04,
            "columns": ["id", "producto", "precio", "cantidad", "categoria", "cliente", "descuento", "recompra"]
        }
        self.df = pd.DataFrame({
            "id": [1, 2, 3],
            "producto": ["A", "B", "C"]
        })

    def test_generar_dossier_ejecutivo(self):
        dossier = generar_dossier_ejecutivo(self.meta, self.df)
        self.assertIn("ficha_tecnica", dossier)
        self.assertEqual(dossier["ficha_tecnica"]["filename"], "ventas.csv")
        self.assertEqual(dossier["ficha_tecnica"]["total_columnas"], 8)
        self.assertIn("auditoria_limpieza", dossier)
        self.assertIn("metricas_arbol", dossier)
        self.assertIn("top_reglas", dossier)
        self.assertIn("conclusiones", dossier)
        self.assertGreaterEqual(len(dossier["conclusiones"]), 3)


if __name__ == "__main__":
    unittest.main()
