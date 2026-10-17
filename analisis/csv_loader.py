"""
Módulo de Carga, Validación e Inferencia de Datasets CSV (RF1)
=============================================================
Encargado: Integrante 1 (Frontend, Validación & Ingesta)

Implementa la validación de formato (.csv), límite de tamaño (≤ 25 MB),
detección automática de codificación (chardet), inferencia inteligente
de separadores (Sniffer) y persistencia física en storage/.
"""

import os
import io
import csv
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
import chardet
import pandas as pd
import numpy as np

# Rutas de almacenamiento
STORAGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "datasets")
CATALOG_PATH = os.path.join(STORAGE_BASE_DIR, "catalog.json")

# Límite máximo de tamaño en bytes (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


def _asegurar_directorio_storage():
    """Crea los directorios de almacenamiento si no existen."""
    os.makedirs(STORAGE_BASE_DIR, exist_ok=True)
    if not os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def obtener_catalogo_datasets() -> List[Dict[str, Any]]:
    """Lee y retorna la lista de datasets registrados en el catálogo local."""
    _asegurar_directorio_storage()
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def guardar_en_catalogo(dataset_info: Dict[str, Any]):
    """Registra o actualiza un dataset en el catálogo."""
    catalogo = obtener_catalogo_datasets()
    # Eliminar si ya existía con el mismo ID
    catalogo = [d for d in catalogo if d["id"] != dataset_info["id"]]
    catalogo.insert(0, dataset_info)  # El más reciente primero
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, indent=2)


def inferir_tipo_columna(serie: pd.Series) -> Dict[str, str]:
    """
    Infiere el tipo de dato semántico de una columna y retorna una etiqueta y estilo.
    """
    dtype_str = str(serie.dtype)
    if pd.api.types.is_bool_dtype(serie):
        return {"tipo": "Booleano", "badge": "bg-purple text-white", "color": "#8b5cf6"}
    elif pd.api.types.is_integer_dtype(serie):
        return {"tipo": "Entero", "badge": "bg-primary text-white", "color": "#2563eb"}
    elif pd.api.types.is_float_dtype(serie):
        return {"tipo": "Decimal", "badge": "bg-info text-dark", "color": "#0ea5e9"}
    elif pd.api.types.is_datetime64_any_dtype(serie):
        return {"tipo": "Fecha", "badge": "bg-success text-white", "color": "#10b981"}
    else:
        # Texto o Categórica
        n_unicos = serie.nunique(dropna=True)
        if n_unicos <= 10:
            return {"tipo": f"Categórica ({n_unicos})", "badge": "bg-warning text-dark", "color": "#f59e0b"}
        return {"tipo": "Texto", "badge": "bg-secondary text-white", "color": "#64748b"}


def cargar_y_validar_csv(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Valida y procesa un archivo CSV subido por el usuario.
    
    Retorna un diccionario con:
    - éxito o error
    - metadatos (filas, columnas, encoding, separador, tamaño)
    - DataFrame procesado
    - preview de registros
    - información semántica de columnas
    """
    _asegurar_directorio_storage()

    # 1. Validación de extensión
    if not filename.lower().endswith(".csv"):
        return {
            "exito": False,
            "error": "Formato de archivo inválido. Solo se admiten archivos con extensión '.csv'."
        }

    # 2. Validación de archivo vacío
    file_size = len(file_bytes)
    if file_size == 0:
        return {
            "exito": False,
            "error": "El archivo subido está vacío (0 bytes)."
        }

    # 3. Validación de tamaño máximo (25 MB)
    if file_size > MAX_FILE_SIZE_BYTES:
        mb = file_size / (1024 * 1024)
        return {
            "exito": False,
            "error": f"El archivo supera el límite permitido de 25 MB (tamaño actual: {mb:.1f} MB)."
        }

    # 4. Detección automática de codificación (Encoding)
    sample_bytes = file_bytes[:64 * 1024]  # Muestra inicial de hasta 64 KB
    detect_result = chardet.detect(sample_bytes)
    encoding = detect_result.get("encoding") or "utf-8"

    # Fallback si chardet falla o detecta con baja confianza
    sample_text = ""
    for enc in [encoding, "utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            sample_text = sample_bytes.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if not sample_text:
        return {
            "exito": False,
            "error": "No fue posible decodificar el archivo. Asegúrate de que sea un texto plano CSV válido."
        }

    # 5. Detección automática del separador
    separador = ","
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=[",", ";", "\t", "|"])
        separador = dialect.delimiter
    except Exception:
        # Heurística de conteo si el sniffer falla
        conteo_comas = sample_text.count(",")
        conteo_puntos_coma = sample_text.count(";")
        conteo_tabs = sample_text.count("\t")
        if conteo_puntos_coma > conteo_comas and conteo_puntos_coma > conteo_tabs:
            separador = ";"
        elif conteo_tabs > conteo_comas and conteo_tabs > conteo_puntos_coma:
            separador = "\t"

    # 6. Lectura con Pandas
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=separador, encoding=encoding)
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al parsear el CSV con separador '{separador}' y codificación '{encoding}': {str(e)}"
        }

    if df.empty or len(df.columns) == 0:
        return {
            "exito": False,
            "error": "El archivo CSV no contiene filas o columnas legibles."
        }

    # 7. Persistencia física en storage/datasets/<uuid>/raw.csv
    dataset_id = str(uuid.uuid4())
    dataset_dir = os.path.join(STORAGE_BASE_DIR, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    raw_path = os.path.join(dataset_dir, "raw.csv")

    with open(raw_path, "wb") as f:
        f.write(file_bytes)

    # 8. Metadatos de columnas y tipos inferidos
    col_info = {}
    for col in df.columns:
        col_info[str(col)] = inferir_tipo_columna(df[col])

    # 9. Registro en catálogo
    dataset_meta = {
        "id": dataset_id,
        "filename": filename,
        "raw_path": raw_path,
        "clean_path": None,
        "filesize_kb": round(file_size / 1024, 1),
        "filesize_mb": round(file_size / (1024 * 1024), 2),
        "encoding": encoding,
        "separator": separador,
        "row_count": int(len(df)),
        "col_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    guardar_en_catalogo(dataset_meta)

    return {
        "exito": True,
        "dataset_id": dataset_id,
        "metadata": dataset_meta,
        "df": df,
        "column_types": col_info,
        "preview_rows": df.head(50).to_dict(orient="records")
    }
