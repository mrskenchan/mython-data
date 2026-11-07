"""
Router del Dashboard Analítico Dinámico (RF6)
==============================================
Encargado: Integrante 1 (Frontend, Visualización & Experiencia de Usuario)
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.dashboard_engine import obtener_datos_dashboard

router = APIRouter(tags=["RF6 - Dashboard Analítico"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Vista de Dashboard Analítico Interactivo (RF6)"""
    ctx = state_manager.get_base_context(request, "dashboard")
    active_df = state_manager.get_active_dataset().get("df")
    
    dash_data = obtener_datos_dashboard(active_df)
    ctx["dash"] = dash_data
    return templates.TemplateResponse(request=request, name="dashboard.html", context=ctx)
