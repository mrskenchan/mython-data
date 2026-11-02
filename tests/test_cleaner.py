import unittest
import pandas as pd
import numpy as np
from analisis.cleaner import diagnosticar_dataset, ejecutar_limpieza


class TestCleaner(unittest.TestCase):

    def setUp(self):
        self.df_crudo = pd.DataFrame({
            "ID Cliente": [1, 2, 3, 4, 5, 6, 7, 1],
            "Nombre y Apellido": ["Ana Gómez", "Carlos Ruiz", "Beatriz Soto", "David Díaz", "Elena Paz", "Fernando Gil", "Gloria Rosas", "Ana Gómez"],
            "Edad (Años)": ["25", "28", "30", np.nan, "35", "40", "42", "25"],
            "Ingreso Mensual $": [1500.0, 1800.0, 2000.0, 2200.0, 2500.0, 2800.0, 100000.0, 1500.0], # 100k es outlier
            "Código Postal": ["01234", "05678", "09999", "01111", "02222", "03333", "04444", "01234"] # Leading zeros
        })

    def test_diagnosticar_dataset(self):
        diag = diagnosticar_dataset(self.df_crudo)
        self.assertEqual(diag["total_filas"], 8)
        self.assertEqual(diag["total_columnas"], 5)
        self.assertEqual(diag["filas_duplicadas"], 1)
        self.assertGreater(diag["total_nulos"], 0)
        self.assertIn("Ingreso Mensual $", diag["columnas_numericas"])
        self.assertGreater(diag["outliers_estimados"], 0)

    def test_normalizacion_columnas_y_tipos(self):
        config = {
            "normalizar_columnas": True,
            "eliminar_duplicados": False,
            "imputacion_nulos": "mediana",
            "manejo_outliers": False
        }
        df_clean, bitacora = ejecutar_limpieza(self.df_crudo, config)
        
        columnas = list(df_clean.columns)
        self.assertIn("id_cliente", columnas)
        self.assertIn("nombre_y_apellido", columnas)
        self.assertIn("edad_anos", columnas)
        self.assertIn("ingreso_mensual", columnas)
        self.assertIn("codigo_postal", columnas)

        # Edad debe haberse convertido a numérico
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean["edad_anos"]))
        # Código postal NO debe convertirse a numérico para preservar el cero inicial
        self.assertFalse(pd.api.types.is_numeric_dtype(df_clean["codigo_postal"]))

    def test_eliminacion_duplicados(self):
        config = {
            "normalizar_columnas": False,
            "eliminar_duplicados": True,
            "imputacion_nulos": "mediana",
            "manejo_outliers": False
        }
        df_clean, bitacora = ejecutar_limpieza(self.df_crudo, config)
        self.assertEqual(len(df_clean), 7)
        self.assertEqual(bitacora["duplicados_removidos"], 1)

    def test_imputacion_nulos_mediana(self):
        config = {
            "normalizar_columnas": True,
            "eliminar_duplicados": True,
            "imputacion_nulos": "mediana",
            "manejo_outliers": False
        }
        df_clean, bitacora = ejecutar_limpieza(self.df_crudo, config)
        self.assertEqual(int(df_clean.isnull().sum().sum()), 0)
        self.assertIn("edad_anos", bitacora["nulos_imputados"])

    def test_imputacion_nulos_drop(self):
        config = {
            "normalizar_columnas": True,
            "eliminar_duplicados": False,
            "imputacion_nulos": "drop",
            "manejo_outliers": False
        }
        df_clean, bitacora = ejecutar_limpieza(self.df_crudo, config)
        self.assertEqual(bitacora["filas_eliminadas_por_nulos"], 1)
        self.assertEqual(int(df_clean.isnull().sum().sum()), 0)

    def test_manejo_outliers_excluyendo_ids(self):
        config = {
            "normalizar_columnas": True,
            "eliminar_duplicados": False,
            "imputacion_nulos": "mediana",
            "manejo_outliers": True
        }
        df_clean, bitacora = ejecutar_limpieza(self.df_crudo, config)
        # El ID Cliente no debe haber sido alterado
        self.assertEqual(list(df_clean["id_cliente"]), [1, 2, 3, 4, 5, 6, 7, 1])
        # Los outliers en ingreso deben haber sido recortados
        self.assertLess(df_clean["ingreso_mensual"].max(), 100000.0)
        self.assertGreater(bitacora["outliers_tratados"], 0)


if __name__ == "__main__":
    unittest.main()
