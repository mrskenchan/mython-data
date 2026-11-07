"""
Paquete de Routers Modulares (Patrón Router / Controller)
=========================================================
"""

from web.routers.upload_router import router as upload_router
from web.routers.clean_router import router as clean_router
from web.routers.tree_router import router as tree_router
from web.routers.rules_router import router as rules_router
from web.routers.dashboard_router import router as dashboard_router
from web.routers.reports_router import router as reports_router

__all__ = [
    "upload_router",
    "clean_router",
    "tree_router",
    "rules_router",
    "dashboard_router",
    "reports_router"
]
