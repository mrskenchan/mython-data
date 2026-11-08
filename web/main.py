"""
Mython App — Servidor Principal FastAPI
=======================================
Punto de entrada de la plataforma de minería de datos y analítica interactiva (RF1 a RF6).
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.routers import (
    upload_router,
    clean_router,
    tree_router,
    rules_router,
    dashboard_router,
    reports_router
)

app = FastAPI(
    title="Mython App — Minería de Datos",
    description="Plataforma web de minería de datos y procesamiento analítico interactivo (RF1 - RF6).",
    version="1.0.0"
)

# Montaje de archivos estáticos (CSS, JS, Favicon)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Entrega el favicon de Python en formato SVG."""
    fav_path = os.path.join(STATIC_DIR, "favicon.svg")
    return FileResponse(fav_path, media_type="image/svg+xml")


# Inclusión de Routers Modulares
app.include_router(upload_router)
app.include_router(clean_router)
app.include_router(tree_router)
app.include_router(rules_router)
app.include_router(dashboard_router)
app.include_router(reports_router)
