"""
Servicio de Gestión de Estado y Sesión de Datos (Patrón Singleton / State Manager)
=================================================================================
Centraliza de manera atómica y controlada el dataset actualmente activo,
las bitácoras de auditoría, las métricas del último modelo de árbol entrenado,
y la entrega de mensajes de retroalimentación para las plantillas Jinja2.
"""

from typing import Dict, Any, Optional
import pandas as pd
from starlette.requests import Request

from analisis.csv_loader import inferir_tipo_columna, obtener_catalogo_datasets


class DatasetStateManager:
    """Gestor singleton del estado en memoria de la sesión analítica."""
    
    _instance: Optional["DatasetStateManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inicializar()
        return cls._instance

    def _inicializar(self):
        # Dataset de demostración inicial
        self.demo_df = pd.DataFrame({
            "id": range(1, 11),
            "edad": [25, 34, 45, 52, 23, 40, 60, 29, 38, 48],
            "ingreso_anual": [28000, 49000, 71000, 85000, 21000, 52000, 98000, 31000, 68000, 81000],
            "hijos": [0, 2, 1, 3, 0, 1, 2, 0, 2, 1],
            "estado_civil": ["Soltero", "Casado", "Casado", "Divorciado", "Soltero", "Casado", "Casado", "Soltero", "Casado", "Casado"],
            "compra_seguro": ["No", "Si", "Si", "Si", "No", "Si", "Si", "No", "Si", "Si"]
        })

        self.current_dataset = {
            "id": None,
            "filename": "clientes_churn.csv",
            "row_count": len(self.demo_df),
            "col_count": len(self.demo_df.columns),
            "encoding": "utf-8",
            "separator": ",",
            "filesize_kb": 142.5,
            "filesize_mb": 0.14,
            "columns": list(self.demo_df.columns),
            "df": self.demo_df,
            "raw_df": self.demo_df.copy(),
            "is_cleaned": False,
            "column_types": {str(c): inferir_tipo_columna(self.demo_df[c]) for c in self.demo_df.columns}
        }

        self.flash_messages = {"success": None, "error": None}
        self.last_clean_audit: Optional[Dict[str, Any]] = None
        self.last_clean_config: Optional[Dict[str, Any]] = None
        self.last_tree_model: Optional[Dict[str, Any]] = None

    def get_active_dataset(self) -> Dict[str, Any]:
        """Retorna el dataset actualmente activo."""
        return self.current_dataset

    def set_active_dataset(self, metadata: Dict[str, Any], df: pd.DataFrame, col_types: Optional[Dict[str, Any]] = None):
        """Actualiza el dataset activo en memoria y recalcula sus tipos de columnas."""
        if col_types is None and df is not None:
            col_types = {str(c): inferir_tipo_columna(df[c]) for c in df.columns}
        
        self.current_dataset = {
            **metadata,
            "df": df,
            "raw_df": df.copy() if df is not None else None,
            "is_cleaned": False,
            "columns": list(df.columns) if df is not None else metadata.get("columns", []),
            "row_count": len(df) if df is not None else metadata.get("row_count", 0),
            "col_count": len(df.columns) if df is not None else metadata.get("col_count", 0),
            "column_types": col_types or {}
        }
        # Limpiar auditorías anteriores para evitar inconsistencias
        self.last_clean_audit = None
        self.last_clean_config = None
        self.last_tree_model = None

    def update_cleaned_dataset(self, df_clean: pd.DataFrame, audit_log: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Persiste el resultado de una operación de limpieza asistida (RF2)."""
        if df_clean is None or len(df_clean) == 0:
            raise ValueError("La operación de limpieza resultó en un dataset vacío (0 filas). Operación abortada para evitar pérdida de datos.")
            
        self.current_dataset["df"] = df_clean
        self.current_dataset["columns"] = list(df_clean.columns)
        self.current_dataset["row_count"] = len(df_clean)
        self.current_dataset["col_count"] = len(df_clean.columns)
        self.current_dataset["column_types"] = {str(c): inferir_tipo_columna(df_clean[c]) for c in df_clean.columns}
        self.current_dataset["is_cleaned"] = True
        self.last_clean_audit = audit_log
        if config:
            self.last_clean_config = config
        self.last_tree_model = None

    def restore_raw_dataset(self) -> bool:
        """Restaura el dataset a su versión cruda original antes de la limpieza."""
        raw = self.current_dataset.get("raw_df")
        if raw is None or raw.empty:
            return False
        self.current_dataset["df"] = raw.copy()
        self.current_dataset["columns"] = list(raw.columns)
        self.current_dataset["row_count"] = len(raw)
        self.current_dataset["col_count"] = len(raw.columns)
        self.current_dataset["column_types"] = {str(c): inferir_tipo_columna(raw[c]) for c in raw.columns}
        self.current_dataset["is_cleaned"] = False
        self.last_clean_audit = None
        self.last_clean_config = None
        self.last_tree_model = None
        return True

    def get_last_clean_config(self) -> Dict[str, Any]:
        """Retorna la última configuración de limpieza ejecutada o defaults."""
        return self.last_clean_config or {
            "normalize_columns": True,
            "drop_duplicates": True,
            "impute_strategy": "median",
            "outlier_method": True,
            "order_chronological": False
        }

    def get_last_clean_audit(self) -> Optional[Dict[str, Any]]:
        """Retorna la bitácora del último pipeline de limpieza ejecutado."""
        return self.last_clean_audit

    def get_last_tree_model(self) -> Optional[Dict[str, Any]]:
        """Retorna las métricas y estructura del último árbol entrenado."""
        return self.last_tree_model

    def set_last_tree_model(self, model_result: Dict[str, Any]):
        """Registra el modelo de árbol recientemente entrenado."""
        self.last_tree_model = model_result

    def set_flash_success(self, message: str):
        """Registra un mensaje temporal de éxito."""
        self.flash_messages["success"] = message

    def set_flash_error(self, message: str):
        """Registra un mensaje temporal de error."""
        self.flash_messages["error"] = message

    def get_base_context(self, request: Request, active_page: str) -> Dict[str, Any]:
        """
        Construye el contexto estándar para renderizar plantillas Jinja2,
        consumiendo los mensajes flash de forma atómica.
        """
        catalogo = obtener_catalogo_datasets()
        succ = self.flash_messages.get("success")
        err = self.flash_messages.get("error")
        # Limpiar tras consumir
        self.flash_messages["success"] = None
        self.flash_messages["error"] = None

        return {
            "request": request,
            "active_page": active_page,
            "active_dataset": self.current_dataset,
            "dataset_catalog": catalogo,
            "alert_success": succ,
            "alert_error": err
        }


# Instancia singleton lista para usar en todos los routers
state_manager = DatasetStateManager()
