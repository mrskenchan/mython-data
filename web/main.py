"""
OrangeWeb — Servidor Principal FastAPI
======================================
Punto de entrada de la aplicación web estructurado según arquitectura modular
y patrón Router/Controller. Orquesta el montaje de archivos estáticos, la inyección
de dependencias y el registro de módulos analíticos (RF1 a RF6).
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.routers import (
    upload_router,
    clean_router,
    tree_router,
    rules_router,
    dashboard_router,
    reports_router
)

app = FastAPI(
    title="OrangeWeb — Minería de Datos",
    description="Plataforma web de minería de datos y procesamiento analítico interactivo (RF1 - RF6).",
    version="1.0.0"
)

# Montaje de archivos estáticos (CSS, JS)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Inclusión de Routers Modulares (Patrón Router / Controller)
app.include_router(upload_router)
app.include_router(clean_router)
app.include_router(tree_router)
app.include_router(rules_router)
app.include_router(dashboard_router)
app.include_router(reports_router)
