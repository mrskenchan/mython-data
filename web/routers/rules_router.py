"""
Router de Minería de Reglas de Asociación y Patrones Frecuentes (RF4)
=====================================================================
Encargado: Integrante 1 (Frontend, Validación & Minería Descriptiva)
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.association import minar_reglas_asociacion

router = APIRouter(tags=["RF4 - Reglas de Asociación"])


@router.get("/rules", response_class=HTMLResponse)
async def rules_page(
    request: Request,
    min_support: float = 0.05,
    min_confidence: float = 0.60,
    min_lift: float = 1.0,
    algorithm: str = "fpgrowth"
):
    """Vista de Reglas de Asociación (RF4)"""
    ctx = state_manager.get_base_context(request, "rules")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    columns = active.get("columns", [])

    resultado = minar_reglas_asociacion(
        df=df,
        columnas=columns,
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
    ctx["available_columns"] = columns
    return templates.TemplateResponse(request=request, name="rules.html", context=ctx)


@router.post("/rules", response_class=HTMLResponse)
async def handle_rules(
    request: Request,
    min_support: float = Form(0.05),
    min_confidence: float = Form(0.60),
    min_lift: float = Form(1.0),
    algorithm: str = Form("fpgrowth")
):
    """Filtra y recalcula las reglas según los controles seleccionados"""
    ctx = state_manager.get_base_context(request, "rules")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    columns = active.get("columns", [])

    resultado = minar_reglas_asociacion(
        df=df,
        columnas=columns,
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
    ctx["available_columns"] = columns
    return templates.TemplateResponse(request=request, name="rules.html", context=ctx)
