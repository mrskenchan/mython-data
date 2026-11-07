"""
Router de Carga de Datos, Validación y Gestión de Catálogo (RF1)
===============================================================
Encargado: Integrante 1 (Frontend, Validación & Ingesta)
"""

import os
import shutil
import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.csv_loader import (
    cargar_y_validar_csv,
    obtener_catalogo_datasets,
    inferir_tipo_columna,
    STORAGE_BASE_DIR,
    CATALOG_PATH
)

router = APIRouter(tags=["RF1 - Carga y Validación de Datasets"])


@router.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/upload")


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Vista de Carga de Datos y Previsualización (RF1)"""
    ctx = state_manager.get_base_context(request, "upload")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    
    ctx["columns"] = active.get("columns", [])
    ctx["column_types"] = active.get("column_types", {})
    ctx["preview_data"] = df.head(50).to_dict(orient="records") if df is not None else []
    return templates.TemplateResponse(request=request, name="upload.html", context=ctx)


@router.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    """Manejo de subida real de archivo CSV con validación integral (RF1)"""
    try:
        content = await file.read()
        res = cargar_y_validar_csv(content, file.filename)
        
        if res["exito"]:
            state_manager.set_active_dataset(
                metadata=res["metadata"],
                df=res["df"],
                col_types=res["column_types"]
            )
            state_manager.set_flash_success(
                f"¡Archivo '{file.filename}' validado y guardado con éxito! "
                f"({res['metadata']['row_count']} registros, codificación {res['metadata']['encoding'].upper()}, "
                f"separador '{res['metadata']['separator']}')."
            )
        else:
            state_manager.set_flash_error(res["error"])
    except Exception as e:
        state_manager.set_flash_error(f"Error crítico al procesar el archivo: {str(e)}")
        
    return RedirectResponse(url="/upload", status_code=303)


@router.get("/sample/load/{sample_name}", response_class=HTMLResponse)
async def load_sample_dataset(request: Request, sample_name: str):
    """Carga de forma inmediata uno de los datasets de demostración preconfigurados."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filename = f"{sample_name}.csv"
    sample_path = os.path.join(base_dir, "sample_data", filename)
    
    if not os.path.exists(sample_path):
        state_manager.set_flash_error(f"El dataset de muestra '{filename}' no fue encontrado en el servidor.")
        return RedirectResponse(url="/upload", status_code=303)
        
    try:
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
            
        res = cargar_y_validar_csv(file_bytes, filename)
        if res["exito"]:
            state_manager.set_active_dataset(
                metadata=res["metadata"],
                df=res["df"],
                col_types=res["column_types"]
            )
            state_manager.set_flash_success(
                f"¡Dataset de muestra '{filename}' cargado y activado con éxito! "
                f"({res['metadata']['row_count']} filas, {res['metadata']['col_count']} columnas listas)."
            )
        else:
            state_manager.set_flash_error(res["error"])
    except Exception as e:
        state_manager.set_flash_error(f"Error al cargar el dataset de muestra: {str(e)}")
        
    return RedirectResponse(url="/upload", status_code=303)


@router.post("/datasets/select/{dataset_id}", response_class=HTMLResponse)
async def select_dataset(request: Request, dataset_id: str):
    """Permite seleccionar y activar un dataset previo desde el modal de catálogo."""
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
            state_manager.set_active_dataset(
                metadata=seleccionado,
                df=df,
                col_types=col_types
            )
            state_manager.set_flash_success(f"Dataset '{seleccionado['filename']}' activado como fuente actual de datos.")
        except Exception as e:
            state_manager.set_flash_error(f"Error al cargar el dataset seleccionado: {str(e)}")
    else:
        state_manager.set_flash_error("El dataset seleccionado no existe o fue removido del servidor.")

    return RedirectResponse(url="/upload", status_code=303)


@router.get("/datasets/delete/{dataset_id}", response_class=HTMLResponse)
async def delete_dataset(request: Request, dataset_id: str):
    """Elimina un dataset del catálogo y remueve su carpeta en storage/."""
    dataset_dir = os.path.join(STORAGE_BASE_DIR, dataset_id)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    
    catalogo = obtener_catalogo_datasets()
    catalogo = [d for d in catalogo if d["id"] != dataset_id]
    try:
        import json
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalogo, f, indent=2)
        state_manager.set_flash_success("Dataset eliminado del almacenamiento.")
    except Exception as e:
        state_manager.set_flash_error(f"Error al actualizar catálogo: {str(e)}")

    return RedirectResponse(url="/upload", status_code=303)
