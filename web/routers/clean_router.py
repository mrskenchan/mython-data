"""
Router de Limpieza Asistida y Auditoría de Datos (RF2)
======================================================
Encargado: Integrante 2 (Backend / Machine Learning)
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.cleaner import diagnosticar_dataset, ejecutar_limpieza

router = APIRouter(tags=["RF2 - Limpieza y Diagnóstico"])


@router.get("/clean", response_class=HTMLResponse)
async def clean_page(request: Request):
    """Vista de Diagnóstico y Limpieza Asistida (RF2)"""
    ctx = state_manager.get_base_context(request, "clean")
    active_df = state_manager.get_active_dataset().get("df")
    
    diag = diagnosticar_dataset(active_df) if active_df is not None else {}
    ctx["diagnostic"] = diag
    
    last_audit = state_manager.get_last_clean_audit()
    ctx["cleaning_executed"] = last_audit is not None
    ctx["audit_log"] = {
        "duplicates_removed": last_audit.get("duplicados_removidos", 0) if last_audit else 0,
        "columns_normalized": last_audit.get("columnas_normalizadas", []) if last_audit else [],
        "outliers_clipped": last_audit.get("outliers_tratados", 0) if last_audit else 0,
        "nulls_imputed": last_audit.get("nulos_imputados", {}) if last_audit else {}
    }
    return templates.TemplateResponse(request=request, name="clean.html", context=ctx)


@router.post("/clean", response_class=HTMLResponse)
async def handle_clean(
    request: Request,
    normalize_columns: bool = Form(False),
    drop_duplicates: bool = Form(False),
    impute_strategy: str = Form("median"),
    outlier_method: bool = Form(False)
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
        "manejo_outliers": outlier_method
    }
    try:
        df_clean, audit = ejecutar_limpieza(active_df, config)
        state_manager.update_cleaned_dataset(df_clean, audit)
        state_manager.set_flash_success("Pipeline de limpieza ejecutado correctamente sobre el dataset activo.")
    except Exception as e:
        state_manager.set_flash_error(f"Error al ejecutar limpieza: {str(e)}")

    return RedirectResponse(url="/clean", status_code=303)
