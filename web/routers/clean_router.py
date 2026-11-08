"""
Router de Limpieza Asistida y Auditoría de Datos (RF2)
======================================================
Encargado: Integrante 2 (Backend / Machine Learning)
"""

import io
from fastapi import APIRouter, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.cleaner import diagnosticar_dataset, ejecutar_limpieza

router = APIRouter(tags=["RF2 - Limpieza y Diagnóstico"])


@router.get("/clean", response_class=HTMLResponse)
async def clean_page(request: Request):
    """Vista de Diagnóstico y Limpieza Asistida (RF2)"""
    ctx = state_manager.get_base_context(request, "clean")
    active = state_manager.get_active_dataset()
    active_df = active.get("df")
    
    diag = diagnosticar_dataset(active_df) if active_df is not None and not active_df.empty else {
        "total_filas": 0,
        "total_columnas": 0,
        "total_nulos": 0,
        "columnas_con_nulos": {},
        "filas_duplicadas": 0,
        "columnas_sucias": [],
        "columnas_numericas": [],
        "outliers_estimados": 0
    }
    ctx["diagnostic"] = diag
    
    last_audit = state_manager.get_last_clean_audit()
    ctx["cleaning_executed"] = last_audit is not None
    ctx["audit_log"] = {
        "duplicates_removed": last_audit.get("duplicados_removidos", 0) if last_audit else 0,
        "columns_normalized": last_audit.get("columnas_normalizadas", []) if last_audit else [],
        "outliers_clipped": last_audit.get("outliers_tratados", 0) if last_audit else 0,
        "nulls_imputed": last_audit.get("nulos_imputados", {}) if last_audit else {},
        "rows_dropped_nulls": last_audit.get("filas_eliminadas_por_nulos", 0) if last_audit else 0,
        "chronological_ordered": last_audit.get("ordenamiento_aplicado") if last_audit else None,
        "advertencias": last_audit.get("advertencias", []) if last_audit else []
    }
    
    ctx["is_cleaned"] = active.get("is_cleaned", False)
    ctx["has_raw_backup"] = active.get("raw_df") is not None
    ctx["last_config"] = state_manager.get_last_clean_config()
    ctx["preview_data"] = active_df.head(20).to_dict(orient="records") if active_df is not None and not active_df.empty else []
    ctx["preview_columns"] = list(active_df.columns) if active_df is not None else []
    ctx["column_types"] = active.get("column_types", {})
    
    return templates.TemplateResponse(request=request, name="clean.html", context=ctx)


@router.post("/clean", response_class=HTMLResponse)
async def handle_clean(
    request: Request,
    normalize_columns: bool = Form(False),
    drop_duplicates: bool = Form(False),
    impute_strategy: str = Form("median"),
    outlier_method: bool = Form(False),
    order_chronological: bool = Form(False)
):
    """Dispara el pipeline de limpieza y redirige a la vista de resultados"""
    active_df = state_manager.get_active_dataset().get("df")
    if active_df is None or active_df.empty:
        state_manager.set_flash_error("No hay un dataset activo con datos para limpiar.")
        return RedirectResponse(url="/clean", status_code=303)

    config = {
        "normalizar_columnas": normalize_columns,
        "eliminar_duplicados": drop_duplicates,
        "imputacion_nulos": impute_strategy,
        "manejo_outliers": outlier_method,
        "ordenar_cronologicamente": order_chronological
    }
    
    ui_config = {
        "normalize_columns": normalize_columns,
        "drop_duplicates": drop_duplicates,
        "impute_strategy": impute_strategy,
        "outlier_method": outlier_method,
        "order_chronological": order_chronological
    }
    
    try:
        df_clean, audit = ejecutar_limpieza(active_df, config)
        state_manager.update_cleaned_dataset(df_clean, audit, config=ui_config)
        
        if audit.get("advertencias"):
            advertencias_txt = " ".join(audit["advertencias"])
            state_manager.set_flash_success(f"Pipeline ejecutado con salvaguarda preventiva: {advertencias_txt}")
        else:
            state_manager.set_flash_success(
                f"¡Limpieza ejecutada exitosamente! Se procesaron {len(df_clean)} registros y {len(df_clean.columns)} variables."
            )
    except Exception as e:
        state_manager.set_flash_error(f"Error al ejecutar limpieza: {str(e)}")

    return RedirectResponse(url="/clean", status_code=303)


@router.post("/clean/restore", response_class=HTMLResponse)
async def restore_raw_data(request: Request):
    """Restaura el dataset a su estado crudo original."""
    exito = state_manager.restore_raw_dataset()
    if exito:
        state_manager.set_flash_success("Se ha restaurado el dataset a su versión original sin modificaciones de limpieza.")
    else:
        state_manager.set_flash_error("No hay un dataset original respaldado para restaurar.")
    return RedirectResponse(url="/clean", status_code=303)


@router.get("/clean/download")
async def download_cleaned_csv():
    """Descarga el dataset activo limpio en formato CSV."""
    active = state_manager.get_active_dataset()
    df = active.get("df")
    if df is None or df.empty:
        return Response(content="Dataset vacio", status_code=400)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    filename = active.get("filename", "dataset.csv")
    if not filename.endswith("_limpio.csv"):
        base_name = filename[:-4] if filename.endswith(".csv") else filename
        filename = f"{base_name}_limpio.csv"
        
    return Response(
        content=csv_buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
