"""
Pruebas de Integración de Endpoints y Routers (FastAPI TestClient)
==================================================================
Valida el ciclo de vida HTTP completo de OrangeWeb para todos los requisitos
funcionales (RF1 a RF6), asegurando la correcta inyección de dependencias,
el renderizado de plantillas Jinja2 y el patrón Router / Controller.
"""

import unittest
from fastapi.testclient import TestClient
from web.main import app
from web.services.state_manager import state_manager


class TestWebRouters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Resetear estado a dataset de muestra antes de cada prueba (GET /sample/load/...)
        self.client.get("/sample/load/telecom_churn", follow_redirects=True)

    def test_root_redirect(self):
        """Valida que la raíz redirija a /upload."""
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/upload")

    def test_get_upload_page(self):
        """Valida la carga y renderizado de la interfaz de carga de datos (RF1)."""
        response = self.client.get("/upload")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Mython App", response.text)
        self.assertIn("Carga de Datos", response.text)

    def test_load_sample_dataset(self):
        """Valida la carga de datasets de muestra incorporados."""
        response = self.client.get("/sample/load/canasta_supermercado", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        active = state_manager.get_active_dataset()
        self.assertIn("canasta_supermercado", active["filename"])

    def test_clean_page_get_and_post(self):
        """Valida el diagnóstico y la ejecución de limpieza asistida (RF2)."""
        # GET diagnóstico
        get_resp = self.client.get("/clean")
        self.assertEqual(get_resp.status_code, 200)
        self.assertIn("Limpieza & Diagnóstico", get_resp.text)

        # POST ejecución de limpieza con opciones
        post_resp = self.client.post("/clean", data={
            "normalize_columns": "true",
            "drop_duplicates": "true",
            "impute_strategy": "mediana",
            "outlier_method": "true"
        }, follow_redirects=True)
        self.assertEqual(post_resp.status_code, 200)
        self.assertIsNotNone(state_manager.get_last_clean_audit())

    def test_tree_page_get_and_post(self):
        """Valida el entrenamiento del Árbol de Decisión con Scikit-Learn (RF3)."""
        # Cargar telecom_churn para tener variable objetivo categórica clara
        self.client.get("/sample/load/telecom_churn", follow_redirects=True)

        # GET vista árbol
        get_resp = self.client.get("/tree")
        self.assertEqual(get_resp.status_code, 200)
        self.assertIn("Árbol de Decisión", get_resp.text)

        # POST entrenamiento
        post_resp = self.client.post("/tree", data={
            "target_column": "churn",
            "features": ["antiguedad_meses", "cargo_mensual"],
            "criterion": "gini",
            "max_depth": "3",
            "test_size": "25"
        })
        self.assertEqual(post_resp.status_code, 200)
        self.assertIn("Exactitud", post_resp.text)
        self.assertIn("F1-Score", post_resp.text)
        self.assertIsNotNone(state_manager.get_last_tree_model())

    def test_rules_page_get_and_post(self):
        """Valida la minería de Reglas de Asociación con FP-Growth / Apriori (RF4)."""
        # Cargar canasta_supermercado
        self.client.get("/sample/load/canasta_supermercado", follow_redirects=True)

        # GET vista reglas
        get_resp = self.client.get("/rules")
        self.assertEqual(get_resp.status_code, 200)
        self.assertIn("Reglas de Asociación", get_resp.text)

        # POST minería
        post_resp = self.client.post("/rules", data={
            "algorithm": "fpgrowth",
            "min_support": "0.05",
            "min_confidence": "0.30",
            "min_lift": "1.0"
        })
        self.assertEqual(post_resp.status_code, 200)
        self.assertIn("Reglas Extraídas", post_resp.text)

    def test_dashboard_page(self):
        """Valida el generador dinámico de analítica y gráficos Plotly (RF6)."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Dashboard Analítico", response.text)
        self.assertIn("Matriz de Correlación", response.text)

    def test_reports_page_and_download(self):
        """Valida el dossier ejecutivo consolidado y descarga HTML (RF5)."""
        # Vista web
        response = self.client.get("/reports")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Dossier", response.text)

        # Descarga adjunta HTML
        dl_resp = self.client.get("/reports?download=html")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn("attachment; filename=", dl_resp.headers.get("content-disposition", ""))


if __name__ == "__main__":
    unittest.main()
