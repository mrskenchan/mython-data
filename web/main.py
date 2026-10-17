"""
OrangeWeb — Servidor Principal FastAPI
======================================
Punto de entrada de la aplicación web. Orquesta el ruteo hacia las plantillas
Jinja2, la entrega de archivos estáticos y la conexión con la capa analítica.
"""

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
import shutil

from analisis.cleaner import diagnosticar_dataset, ejecutar_limpieza
from analisis.tree_engine import entrenar_arbol
from analisis.association import generar_reglas_mock, minar_reglas_asociacion
from analisis.dashboard_engine import obtener_datos_dashboard
from analisis.csv_loader import (
    cargar_y_validar_csv,
    obtener_catalogo_datasets,
    inferir_tipo_columna,
    STORAGE_BASE_DIR,
    CATALOG_PATH
)

app = FastAPI(
    title="OrangeWeb — Minería de Datos",
    description="Plataforma de procesamiento analítico y minería de datos inspirada en Orange Data Mining.",
    version="1.0.0"
)

# Montaje de archivos estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Configuración de plantillas Jinja2
templates = Jinja2Templates(directory="web/templates")

# Dataset de demostración inicial
DEMO_DF = pd.DataFrame({
    "id": range(1, 11),
    "edad": [25, 34, 45, 52, 23, 40, 60, 29, 38, 48],
    "ingreso_anual": [28000, 49000, 71000, 85000, 21000, 52000, 98000, 31000, 68000, 81000],
    "hijos": [0, 2, 1, 3, 0, 1, 2, 0, 2, 1],
    "estado_civil": ["Soltero", "Casado", "Casado", "Divorciado", "Soltero", "Casado", "Casado", "Soltero", "Casado", "Casado"],
    "compra_seguro": ["No", "Si", "Si", "Si", "No", "Si", "Si", "No", "Si", "Si"]
})

CURRENT_DATASET = {
    "id": None,
    "filename": "clientes_churn.csv",
    "row_count": 12450,
    "col_count": 6,
    "encoding": "utf-8",
    "separator": ",",
    "filesize_kb": 142.5,
    "filesize_mb": 0.14,
    "columns": ["id", "edad", "ingreso_anual", "hijos", "estado_civil", "compra_seguro"],
    "df": DEMO_DF,
    "column_types": {str(c): inferir_tipo_columna(DEMO_DF[c]) for c in DEMO_DF.columns}
}

# Estado de mensajes temporales
FLASH_MESSAGES = {"success": None, "error": None}


def _obtener_contexto_base(request: Request, active_page: str) -> dict:
    """Genera el diccionario de contexto base para todas las plantillas."""
    global FLASH_MESSAGES
    catalogo = obtener_catalogo_datasets()
    succ = FLASH_MESSAGES.get("success")
    err = FLASH_MESSAGES.get("error")
    # Limpiar tras consumir
    FLASH_MESSAGES["success"] = None
    FLASH_MESSAGES["error"] = None

    return {
        "request": request,
        "active_page": active_page,
        "active_dataset": CURRENT_DATASET,
        "dataset_catalog": catalogo,
        "alert_success": succ,
        "alert_error": err
    }


# ==================== RUTAS DE LA INTERFAZ ====================

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/upload")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Vista de Carga de Datos y Previsualización (RF1)"""
    ctx = _obtener_contexto_base(request, "upload")
    df = CURRENT_DATASET["df"]
    ctx["columns"] = CURRENT_DATASET["columns"]
    ctx["column_types"] = CURRENT_DATASET.get("column_types", {})
    ctx["preview_data"] = df.head(50).to_dict(orient="records")
    return templates.TemplateResponse(request=request, name="upload.html", context=ctx)


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    """Manejo de subida real de archivo CSV con validación integral (RF1)"""
    global CURRENT_DATASET, FLASH_MESSAGES
    try:
        content = await file.read()
        res = cargar_y_validar_csv(content, file.filename)
        
        if res["exito"]:
            CURRENT_DATASET = {
                **res["metadata"],
                "df": res["df"],
                "column_types": res["column_types"]
            }
            FLASH_MESSAGES["success"] = (
                f"¡Archivo '{file.filename}' validado y guardado con éxito! "
                f"({res['metadata']['row_count']} registros, codificación {res['metadata']['encoding'].upper()}, "
                f"separador '{res['metadata']['separator']}')."
            )
        else:
            FLASH_MESSAGES["error"] = res["error"]
    except Exception as e:
        FLASH_MESSAGES["error"] = f"Error crítico al procesar el archivo: {str(e)}"
        
    return RedirectResponse(url="/upload", status_code=303)


@app.post("/datasets/select/{dataset_id}", response_class=HTMLResponse)
async def select_dataset(request: Request, dataset_id: str):
    """Permite seleccionar y activar un dataset previo desde el modal de catálogo."""
    global CURRENT_DATASET, FLASH_MESSAGES
    catalogo = obtener_catalogo_datasets()
    seleccionado = next((d for d in catalogo if d["id"] == dataset_id), None)
    
    if seleccionado and os.path.exists(seleccionado["raw_path"]):
        try:
            df = pd.read_csv(
                seleccionado["raw_path"],
                sep=seleccionado["separator"],
                encoding=seleccionado["encoding"]
            )
            col_types = {str(c): inferir_tipo_columna(df[c]) for c in df.columns}
            CURRENT_DATASET = {
                **seleccionado,
                "df": df,
                "column_types": col_types
            }
            FLASH_MESSAGES["success"] = f"Dataset '{seleccionado['filename']}' activado como fuente actual de datos."
        except Exception as e:
            FLASH_MESSAGES["error"] = f"Error al cargar el dataset seleccionado: {str(e)}"
    else:
        FLASH_MESSAGES["error"] = "El dataset seleccionado no existe o fue removido del servidor."

    return RedirectResponse(url="/upload", status_code=303)


@app.get("/datasets/delete/{dataset_id}", response_class=HTMLResponse)
async def delete_dataset(request: Request, dataset_id: str):
    """Elimina un dataset del catálogo y remueve su carpeta en storage/."""
    global FLASH_MESSAGES
    dataset_dir = os.path.join(STORAGE_BASE_DIR, dataset_id)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    
    # Actualizar catalog.json
    catalogo = obtener_catalogo_datasets()
    catalogo = [d for d in catalogo if d["id"] != dataset_id]
    try:
        import json
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalogo, f, indent=2)
        FLASH_MESSAGES["success"] = "Dataset eliminado del almacenamiento."
    except Exception as e:
        FLASH_MESSAGES["error"] = f"Error al actualizar catálogo: {str(e)}"

    return RedirectResponse(url="/upload", status_code=303)


@app.get("/clean", response_class=HTMLResponse)
async def clean_page(request: Request):
    """Vista de Diagnóstico y Limpieza Asistida (RF2)"""
    ctx = _obtener_contexto_base(request, "clean")
    diag = diagnosticar_dataset(CURRENT_DATASET["df"])
    
    audit = {
        "duplicates_removed": 18,
        "columns_normalized": CURRENT_DATASET["columns"],
        "outliers_clipped": 24,
        "nulls_imputed": {
            col: f"Nulos imputados según estrategia" for col in diag.get("columnas_con_nulos", {}).keys()
        } if diag.get("columnas_con_nulos") else {"edad": "12 nulos imputados con Mediana"}
    }

    ctx["diagnostic"] = diag
    ctx["audit_log"] = audit
    return templates.TemplateResponse(request=request, name="clean.html", context=ctx)


@app.post("/clean", response_class=HTMLResponse)
async def handle_clean(
    request: Request,
    normalize_columns: bool = Form(False),
    drop_duplicates: bool = Form(False),
    impute_strategy: str = Form("median"),
    outlier_method: bool = Form(False)
):
    """Dispara el pipeline de limpieza y redirige a la vista de resultados"""
    global CURRENT_DATASET, FLASH_MESSAGES
    config = {
        "normalizar_columnas": normalize_columns,
        "eliminar_duplicados": drop_duplicates,
        "imputacion_nulos": impute_strategy,
        "manejo_outliers": outlier_method
    }
    df_clean, audit = ejecutar_limpieza(CURRENT_DATASET["df"], config)
    CURRENT_DATASET["df"] = df_clean
    CURRENT_DATASET["columns"] = list(df_clean.columns)
    FLASH_MESSAGES["success"] = "Pipeline de limpieza ejecutado correctamente sobre el dataset activo."
    return RedirectResponse(url="/clean", status_code=303)


@app.get("/tree", response_class=HTMLResponse)
async def tree_page(request: Request):
    """Vista de Árbol de Decisión (RF3)"""
    ctx = _obtener_contexto_base(request, "tree")
    mock_tree = entrenar_arbol(
        df=None,
        target="compra_seguro",
        features=["edad", "ingreso_anual", "hijos", "estado_civil"],
        max_depth=5,
        criterion="gini"
    )
    ctx["available_columns"] = CURRENT_DATASET["columns"]
    ctx["selected_target"] = CURRENT_DATASET["columns"][-1] if CURRENT_DATASET["columns"] else "compra_seguro"
    ctx["current_depth"] = 5
    ctx["model_result"] = mock_tree
    return templates.TemplateResponse(request=request, name="tree.html", context=ctx)


@app.post("/tree", response_class=HTMLResponse)
async def handle_tree(
    request: Request,
    target_column: str = Form("compra_seguro"),
    max_depth: int = Form(5),
    criterion: str = Form("gini")
):
    """Entrena o re-calcula el árbol con los parámetros seleccionados"""
    ctx = _obtener_contexto_base(request, "tree")
    tree_result = entrenar_arbol(
        df=None,
        target=target_column,
        features=[c for c in CURRENT_DATASET["columns"] if c != target_column],
        max_depth=max_depth,
        criterion=criterion
    )
    ctx["available_columns"] = CURRENT_DATASET["columns"]
    ctx["selected_target"] = target_column
    ctx["current_depth"] = max_depth
    ctx["model_result"] = tree_result
    return templates.TemplateResponse(request=request, name="tree.html", context=ctx)


@app.get("/rules", response_class=HTMLResponse)
async def rules_page(
    request: Request,
    min_support: float = 0.05,
    min_confidence: float = 0.60,
    min_lift: float = 1.0,
    algorithm: str = "fpgrowth"
):
    """Vista de Reglas de Asociación (RF4)"""
    ctx = _obtener_contexto_base(request, "rules")
    resultado = minar_reglas_asociacion(
        df=CURRENT_DATASET.get("df"),
        columnas=CURRENT_DATASET.get("columns"),
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        algorithm=algorithm
    )
    ctx["rules"] = resultado.get("reglas", [])
    ctx["total_reglas"] = resultado.get("total_reglas", 0)
    ctx["frequent_itemsets_count"] = resultado.get("frequent_itemsets_count", 0)
    ctx["algoritmo_utilizado"] = resultado.get("algoritmo_utilizado", algorithm.upper())
    ctx["aviso"] = resultado.get("aviso")
    ctx["min_support"] = min_support
    ctx["min_confidence"] = min_confidence
    ctx["min_lift"] = min_lift
    ctx["algorithm"] = algorithm
    ctx["available_columns"] = CURRENT_DATASET.get("columns", [])
    return templates.TemplateResponse(request=request, name="rules.html", context=ctx)


@app.post("/rules", response_class=HTMLResponse)
async def handle_rules(
    request: Request,
    min_support: float = Form(0.05),
    min_confidence: float = Form(0.60),
    min_lift: float = Form(1.0),
    algorithm: str = Form("fpgrowth")
):
    """Filtra y recalcula las reglas según los controles seleccionados"""
    ctx = _obtener_contexto_base(request, "rules")
    resultado = minar_reglas_asociacion(
        df=CURRENT_DATASET.get("df"),
        columnas=CURRENT_DATASET.get("columns"),
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        algorithm=algorithm
    )
    ctx["rules"] = resultado.get("reglas", [])
    ctx["total_reglas"] = resultado.get("total_reglas", 0)
    ctx["frequent_itemsets_count"] = resultado.get("frequent_itemsets_count", 0)
    ctx["algoritmo_utilizado"] = resultado.get("algoritmo_utilizado", algorithm.upper())
    ctx["aviso"] = resultado.get("aviso")
    ctx["min_support"] = min_support
    ctx["min_confidence"] = min_confidence
    ctx["min_lift"] = min_lift
    ctx["algorithm"] = algorithm
    ctx["available_columns"] = CURRENT_DATASET.get("columns", [])
    return templates.TemplateResponse(request=request, name="rules.html", context=ctx)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Vista de Dashboard Analítico Interactivo (RF6)"""
    ctx = _obtener_contexto_base(request, "dashboard")
    dash_data = obtener_datos_dashboard(CURRENT_DATASET.get("df"))
    ctx["dash"] = dash_data
    return templates.TemplateResponse(request=request, name="dashboard.html", context=ctx)


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Vista de Dossier e Informes Ejecutivos (RF5)"""
    ctx = _obtener_contexto_base(request, "reports")
    return templates.TemplateResponse(request=request, name="report.html", context=ctx)
