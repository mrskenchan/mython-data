"""
passenger_wsgi.py — Archivo Puente para Despliegue en DirectAdmin (CloudLinux)
==============================================================================
Convierte la aplicación ASGI de FastAPI en un estándar WSGI compatible con
Phusion Passenger para ejecutarse sobre el servidor web en https://mineria.aerys.cl
"""

import sys
import os

# 1. Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 2. Importar el middleware adaptador ASGI a WSGI y la aplicación FastAPI
from a2wsgi import ASGIMiddleware
from web.main import app

# 3. Phusion Passenger busca obligatoriamente la variable 'application'
application = ASGIMiddleware(app)
