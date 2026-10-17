import unittest
import io
import pandas as pd
from analisis.csv_loader import (
    cargar_y_validar_csv,
    inferir_tipo_columna,
    MAX_FILE_SIZE_BYTES
)

class TestCSVLoader(unittest.TestCase):

    def test_csv_valido_coma_utf8(self):
        contenido = "id,nombre,edad,salario\n1,Ana,28,1500.50\n2,Carlos,35,2300.00\n3,Beatriz,42,3100.75\n"
        data_bytes = contenido.encode("utf-8")
        res = cargar_y_validar_csv(data_bytes, "empleados.csv")
        self.assertTrue(res["exito"])
        self.assertEqual(res["metadata"]["separator"], ",")
        self.assertEqual(res["metadata"]["row_count"], 3)
        self.assertEqual(res["metadata"]["col_count"], 4)
        self.assertIn("edad", res["column_types"])

    def test_csv_valido_punto_coma_latin1(self):
        contenido = "id;ciudad;clasificación\n101;Valparaíso;Alta\n102;Concepción;Media\n"
        data_bytes = contenido.encode("latin-1")
        res = cargar_y_validar_csv(data_bytes, "ciudades.csv")
        self.assertTrue(res["exito"])
        self.assertEqual(res["metadata"]["separator"], ";")
        self.assertEqual(res["metadata"]["row_count"], 2)

    def test_extension_invalida(self):
        res = cargar_y_validar_csv(b"dummy,content\n1,2\n", "archivo.xlsx")
        self.assertFalse(res["exito"])
        self.assertIn("extensión '.csv'", res["error"])

    def test_archivo_vacio(self):
        res = cargar_y_validar_csv(b"", "vacio.csv")
        self.assertFalse(res["exito"])
        self.assertIn("está vacío", res["error"])

    def test_archivo_sobrepasa_limite(self):
        # Simular tamaño mayor a 25MB
        bytes_grandes = b"a" * (MAX_FILE_SIZE_BYTES + 1024)
        res = cargar_y_validar_csv(bytes_grandes, "pesado.csv")
        self.assertFalse(res["exito"])
        self.assertIn("supera el límite permitido", res["error"])

    def test_inferencia_tipos_columna(self):
        s_int = pd.Series([1, 2, 3, 4])
        self.assertEqual(inferir_tipo_columna(s_int)["tipo"], "Entero")

        s_float = pd.Series([1.5, 2.3, 3.8])
        self.assertEqual(inferir_tipo_columna(s_float)["tipo"], "Decimal")

        s_cat = pd.Series(["A", "B", "A", "B", "C"])
        self.assertIn("Categórica", inferir_tipo_columna(s_cat)["tipo"])


if __name__ == "__main__":
    unittest.main()
