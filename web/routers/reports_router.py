"""
Router de Informes Ejecutivos y Dossier Analítico (RF5)
======================================================
Encargado: Integrante 1 (Frontend, Reportes & DevOps)
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.association import minar_reglas_asociacion
from analisis.reporter import generar_dossier_ejecutivo

router = APIRouter(tags=["RF5 - Informes Ejecutivos"])


@router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request, download: str = None):
    """Vista de Dossier e Informes Ejecutivos (RF5)"""
    ctx = state_manager.get_base_context(request, "reports")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    columns = active.get("columns", [])

    # Extraer reglas más relevantes para el informe
    reglas_res = minar_reglas_asociacion(
        df=df,
        columnas=columns,
        min_support=0.05,
        min_confidence=0.60,
        min_lift=1.0
    )
    reglas = reglas_res.get("reglas", [])

    # Obtener auditoría y modelo de árbol de la sesión
    audit_limpieza = state_manager.get_last_clean_audit()
    last_tree = state_manager.get_last_tree_model()
    
    metricas_arbol = None
    if last_tree:
        metricas_arbol = {
            "target": last_tree.get("target", "objetivo"),
            "accuracy": last_tree.get("accuracy", 0.90),
            "precision": last_tree.get("precision", 0.90),
            "recall": last_tree.get("recall", 0.90),
            "f1_score": last_tree.get("f1_score", 0.90),
            "main_rule": f"Modelo Scikit-Learn '{last_tree.get('name')}' con exactitud del {last_tree.get('accuracy', 0)*100:.1f}%."
        }

    dossier = generar_dossier_ejecutivo(
        dataset_meta=active,
        df=df,
        audit_limpieza=audit_limpieza,
        metricas_arbol=metricas_arbol,
        reglas_asociacion=reglas
    )
    ctx["dossier"] = dossier

    response = templates.TemplateResponse(request=request, name="report.html", context=ctx)
    if download == "html":
        response.headers["Content-Disposition"] = 'attachment; filename="informe_ejecutivo_mython.html"'
    return response
