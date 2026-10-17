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
import io
import os

from analisis.cleaner import diagnosticar_dataset, ejecutar_limpieza
from analisis.tree_engine import entrenar_arbol
from analisis.association import generar_reglas_mock

app = FastAPI(
    title="OrangeWeb — Minería de Datos",
    description="Plataforma de procesamiento analítico y minería de datos inspirada en Orange Data Mining.",
    version="1.0.0"
)

# Montaje de archivos estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Configuración de plantillas Jinja2
templates = Jinja2Templates(directory="web/templates")

# Estado simulado en memoria para la maqueta interactiva
MOCK_DF = pd.DataFrame({
    "id": range(1, 11),
    "edad": [25, 34, 45, 52, 23, 40, 60, 29, 38, 48],
    "ingreso_anual": [28000, 49000, 71000, 85000, 21000, 52000, 98000, 31000, 68000, 81000],
    "hijos": [0, 2, 1, 3, 0, 1, 2, 0, 2, 1],
    "estado_civil": ["Soltero", "Casado", "Casado", "Divorciado", "Soltero", "Casado", "Casado", "Soltero", "Casado", "Casado"],
    "compra_seguro": ["No", "Si", "Si", "Si", "No", "Si", "Si", "No", "Si", "Si"]
})

CURRENT_DATASET = {
    "filename": "clientes_churn.csv",
    "rows": 12450,
    "columns": ["id", "edad", "ingreso_anual", "hijos", "estado_civil", "compra_seguro"],
    "df": MOCK_DF
}


# ==================== RUTAS DE LA INTERFAZ ====================

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/upload")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Vista de Carga de Datos y Previsualización (RF1)"""
    preview_rows = CURRENT_DATASET["df"].to_dict(orient="records")
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "active_page": "upload",
            "active_dataset": CURRENT_DATASET,
            "columns": CURRENT_DATASET["columns"],
            "preview_data": preview_rows
        }
    )


@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    """Manejo de subida real de archivo CSV con validación inicial (RF1)"""
    global CURRENT_DATASET
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        CURRENT_DATASET = {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "df": df
        }
    except Exception as e:
        pass  # En caso de error se conserva el dataset mock
        
    return RedirectResponse(url="/clean", status_code=303)


@app.get("/clean", response_class=HTMLResponse)
async def clean_page(request: Request):
    """Vista de Diagnóstico y Limpieza Asistida (RF2)"""
    diag = diagnosticar_dataset(CURRENT_DATASET["df"])
    
    # Resumen de auditoría pre-cargado para la maqueta
    audit = {
        "duplicates_removed": 18,
        "columns_normalized": CURRENT_DATASET["columns"],
        "outliers_clipped": 24,
        "nulls_imputed": {
            "edad": "12 nulos imputados con Mediana (35.0)",
            "ingreso_anual": "40 nulos imputados con Mediana ($48,500)",
            "estado_civil": "90 nulos imputados con Moda ('Casado')"
        }
    }

    return templates.TemplateResponse(
        "clean.html",
        {
            "request": request,
            "active_page": "clean",
            "active_dataset": CURRENT_DATASET,
            "diagnostic": diag,
            "audit_log": audit
        }
    )


@app.post("/clean", response_class=HTMLResponse)
async def handle_clean(request: Request):
    """Dispara el pipeline de limpieza y redirige a la vista de resultados"""
    return RedirectResponse(url="/clean", status_code=303)


@app.get("/tree", response_class=HTMLResponse)
async def tree_page(request: Request):
    """Vista de Árbol de Decisión (RF3)"""
    mock_tree = entrenar_arbol(
        df=None,
        target="compra_seguro",
        features=["edad", "ingreso_anual", "hijos", "estado_civil"],
        max_depth=5,
        criterion="gini"
    )

    return templates.TemplateResponse(
        "tree.html",
        {
            "request": request,
            "active_page": "tree",
            "active_dataset": CURRENT_DATASET,
            "available_columns": CURRENT_DATASET["columns"],
            "selected_target": "compra_seguro",
            "current_depth": 5,
            "model_result": mock_tree
        }
    )


@app.post("/tree", response_class=HTMLResponse)
async def handle_tree(
    request: Request,
    target_column: str = Form("compra_seguro"),
    max_depth: int = Form(5),
    criterion: str = Form("gini")
):
    """Entrena o re-calcula el árbol con los parámetros seleccionados"""
    tree_result = entrenar_arbol(
        df=None,
        target=target_column,
        features=["edad", "ingreso_anual", "hijos", "estado_civil"],
        max_depth=max_depth,
        criterion=criterion
    )
    return templates.TemplateResponse(
        "tree.html",
        {
            "request": request,
            "active_page": "tree",
            "active_dataset": CURRENT_DATASET,
            "available_columns": CURRENT_DATASET["columns"],
            "selected_target": target_column,
            "current_depth": max_depth,
            "model_result": tree_result
        }
    )


@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    """Vista de Reglas de Asociación (RF4)"""
    rules = generar_reglas_mock(min_support=0.05, min_confidence=0.60, min_lift=1.0)
    return templates.TemplateResponse(
        "rules.html",
        {
            "request": request,
            "active_page": "rules",
            "active_dataset": CURRENT_DATASET,
            "rules": rules
        }
    )


@app.post("/rules", response_class=HTMLResponse)
async def handle_rules(
    request: Request,
    min_support: float = Form(0.05),
    min_confidence: float = Form(0.60)
):
    """Filtra y recalcula las reglas según los sliders seleccionados"""
    rules = generar_reglas_mock(min_support=min_support, min_confidence=min_confidence, min_lift=1.0)
    return templates.TemplateResponse(
        "rules.html",
        {
            "request": request,
            "active_page": "rules",
            "active_dataset": CURRENT_DATASET,
            "rules": rules
        }
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Vista de Dashboard Analítico Interactivo (RF6)"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "active_page": "dashboard",
            "active_dataset": CURRENT_DATASET
        }
    )


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Vista de Dossier e Informes Ejecutivos (RF5)"""
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "active_page": "reports",
            "active_dataset": CURRENT_DATASET
        }
    )
