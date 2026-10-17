# 🍊 OrangeWeb — Software de Administración de Minería de Datos

> **Plataforma web analítica con procesamiento y minería de datos en Python**, inspirada en los flujos de trabajo de **Orange Data Mining**. Permite cargar archivos CSV, limpiarlos de forma interactiva y automática, entrenar árboles de decisión supervisados, descubrir reglas de asociación, visualizar métricas en un dashboard dinámico y exportar informes ejecutivos.

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![MLxtend](https://img.shields.io/badge/Mining-MLxtend-green.svg)](https://rasbt.github.io/mlxtend/)
[![Production URL](https://img.shields.io/badge/Deployment-mineria.aerys.cl-orange.svg)](https://mineria.aerys.cl)

---

## 📋 Tabla de Contenidos
1. [Objetivos del Sistema](#-objetivos-del-sistema)
2. [Arquitectura y Comunicación entre Capas](#-arquitectura-y-comunicación-entre-capas)
3. [Requerimientos Funcionales (RF)](#-requerimientos-funcionales-rf)
4. [Estructura del Repositorio](#-estructura-del-repositorio)
5. [Instalación y Ejecución Local](#-instalación-y-ejecución-local)
6. [Despliegue en Producción (DirectAdmin / aerys.cl)](#-despliegue-en-producción-directadmin--aeryscl)
7. [Estrategia de Trabajo con Git](#-estrategia-de-trabajo-con-git)
8. [Equipo y Asignación de Roles](#-equipo-y-asignación-de-roles)

---

## 🎯 Objetivos del Sistema
* **General:** Diseñar y desarrollar una aplicación web modular que reproduzca las funcionalidades principales de Orange: carga de datos, limpieza automática/asistida, modelado predictivo, minería de asociación, informes y dashboards.
* **Procesamiento analítico obligatorio en Python:** Validación y limpieza con Pandas/NumPy, árboles de decisión con Scikit-Learn y reglas de asociación con MLxtend.

---

## 🏛 Arquitectura y Comunicación entre Capas

La aplicación está construida bajo una **Arquitectura en Capas Desacopladas (Clean Layered Architecture)** implementada en Python con **FastAPI**:

```
[ Navegador Web / Cliente HTTP ]
              │ (HTTPS / TLS 1.3)
              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CAPA WEB & PRESENTACIÓN (/web)                           │
│    • FastAPI Routers (Upload, Clean, Tree, Rules, Reports)   │
│    • Jinja2 Server-Side Templates (HTML5 + Bootstrap 5)     │
│    • Componentes interactivos y gráficos Plotly.js          │
│    • Validación de contratos de entrada con esquemas DTO    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Invocación directa en memoria
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CAPA DE PROCESAMIENTO Y MINERÍA (/analisis)              │
│    • Pipeline de Limpieza: Deduplicar, Imputar, Tratar IQR   │
│    • Árboles de Decisión: Scikit-Learn + Visualizador SVG   │
│    • Reglas de Asociación: FP-Growth & Apriori (MLxtend)    │
│    • Generador de Informes (Dossier HTML / PDF Print)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Lectura / Escritura
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. CAPA DE PERSISTENCIA Y ARCHIVOS (/storage)               │
│    • Base de Datos SQLite: Datasets, logs, modelos y reglas │
│    • File Storage: Versiones físicas raw.csv y cleaned.csv  │
│    • Modelos Serializados: Archivos binarios .joblib        │
└─────────────────────────────────────────────────────────────┘
```

### Justificación de Comunicación entre Capas
Al elegir **Python unificado para ambas capas**, la comunicación es **nativa y en memoria RAM** a través de DataFrames de Pandas y arrays de NumPy. Esto elimina la sobrecarga de serializar grandes archivos CSV o JSON entre procesos independientes, garantizando tiempos de respuesta mínimos.

---

## ⚙️ Requerimientos Funcionales (RF)

| Requerimiento | Descripción | Tecnología Principal |
| :--- | :--- | :--- |
| **RF1: Carga de Datos** | Subida CSV drag & drop, validación de MIME/extensión/tamaño (≤25MB), detección de encoding (`chardet`), separador (`csv.Sniffer`) y vista previa interactiva. | `FastAPI`, `python-multipart` |
| **RF2: Limpieza Automática** | Diagnóstico inicial + selector *multiple-choice*. Normalización `snake_case`, eliminación de duplicados, tipado, imputación nulos (mediana/moda) y recorte IQR de outliers con bitácora auditable. | `pandas`, `numpy` |
| **RF3: Árbol de Decisión** | Selección de variables ($Y$ y $X$), hiperparámetros (profundidad, split 70/30, Gini/Entropía), métricas (Accuracy, Precision, Recall, Matriz de Confusión), visualizador SVG y comparador de modelos. | `scikit-learn`, `joblib` |
| **RF4: Reglas de Asociación** | Algoritmos **FP-Growth** y **Apriori** con filtros de soporte, confianza mínima y ordenamiento por *Lift*. Visualización en tabla interactiva y scatter plot. | `mlxtend` |
| **RF5: Informes** | Generación de dossier consolidado (datos, auditoría, árbol, reglas y conclusiones). Exportación autónoma a **HTML** y botón de descarga en **PDF** vía `@media print`. | `Jinja2`, `CSS3 Print` |
| **RF6: Dashboard** | Panel de analítica interactiva con histogramas, diagramas de dispersión, matriz de correlación *Heatmap* y filtros dinámicos. | `Plotly.js` |
| **RF7: Cobertura Orange** | Módulos integrados con flujo ordenado: Datos ➔ Preprocesamiento ➔ Visualización ➔ Árbol de Decisión ➔ Asociación. | `Bootstrap 5`, UI Orange-like |

---

## 📁 Estructura del Repositorio

El proyecto cumple estrictamente con la separación física de carpetas exigida en la pauta:

```text
mython-data/
├── web/                       # Capa de presentación y ruteo
│   ├── routers/               # Endpoints REST (upload, clean, tree, rules, reports)
│   ├── templates/             # Plantillas Jinja2 (HTML5)
│   │   ├── base.html          # Layout principal (Sidebar estilo Orange)
│   │   ├── upload.html        # Vista de carga y previsualización (RF1)
│   │   ├── clean.html         # Asistente de limpieza multiple-choice (RF2)
│   │   ├── tree.html          # Árbol de decisión y comparador (RF3)
│   │   ├── rules.html         # Reglas de asociación (RF4)
│   │   ├── dashboard.html     # Dashboard interactivo Plotly (RF6)
│   │   └── report.html        # Dossier de informes HTML/PDF (RF5)
│   ├── static/                # Archivos estáticos (CSS, JS, iconos)
│   └── main.py                # Instancia de FastAPI y configuración
├── analisis/                  # Capa científica y de minería de datos (Python)
│   ├── cleaner.py             # Pipeline de limpieza automática y diagnóstico (RF2)
│   ├── tree_engine.py         # Motor de entrenamiento Scikit-Learn y métricas (RF3)
│   ├── association.py         # Minería de patrones frecuentes FP-Growth/Apriori (RF4)
│   └── reporter.py            # Generador del dossier ejecutivo (RF5)
├── storage/                   # Capa de persistencia local (Ignorada en Git)
│   ├── database.sqlite3       # Base de datos relacional de metadatos y modelos
│   ├── datasets/              # Archivos físicos raw.csv y cleaned.csv versionados
│   └── models/                # Modelos Scikit-Learn serializados (.joblib)
├── passenger_wsgi.py          # Puente WSGI para despliegue en DirectAdmin
├── requirements.txt           # Dependencias oficiales del proyecto
├── .gitignore                 # Exclusión de entornos, temporales, PDFs y datasets
└── README.md                  # Documentación oficial del proyecto
```

---

## 🚀 Instalación y Ejecución Local

### 1. Prerrequisitos
* **Python 3.11** o superior instalado.
* Git configurado en tu terminal.

### 2. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/mython-data.git
cd mython-data
```

### 3. Crear y Activar Entorno Virtual
```bash
# En Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# En Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Iniciar Servidor de Desarrollo
```bash
uvicorn web.main:app --reload --host 127.0.0.1 --port 8000
```
Abre tu navegador en [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 🌐 Despliegue en Producción (DirectAdmin / aerys.cl)

El proyecto está configurado para ejecutarse sobre **CloudLinux Setup Python App** en el servidor de hosting **DirectAdmin** bajo el dominio `aerys.cl`:

* **Versión de Python en Producción:** Python 3.11.16 en entorno virtual aislado.
* **URL Pública:** `https://mineria.aerys.cl` (con SSL Let's Encrypt).
* **Pasarela WSGI:** Utiliza `passenger_wsgi.py` con el middleware `a2wsgi` para servir la aplicación FastAPI de forma continua:

```python
# passenger_wsgi.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from a2wsgi import ASGIMiddleware
from web.main import app

application = ASGIMiddleware(app)
```

---

## 🌿 Estrategia de Trabajo con Git

Para garantizar el cumplimiento de la rúbrica de evaluación (25% en Nota 1 y 10% en Nota 2):
1. **Ramas Principales:**
   * `main`: Código estable de producción y entregas oficiales.
   * `develop`: Rama base para integración continua.
2. **Ramas por Funcionalidad:**
   * `feature/carga-csv-rf1`
   * `feature/limpieza-datos-rf2`
   * `feature/arbol-scikit-rf3`
   * `feature/reglas-asociacion-rf4`
   * `feature/informes-dashboard`
3. **Commits Equitativos:** Cada integrante comitea exclusivamente desde su cuenta en sus ramas asignadas antes de fusionar a `develop` mediante Pull Request.

---

## 👥 Equipo y Asignación de Roles

| Integrante | Rol Principal | Requerimientos Asignados |
| :--- | :--- | :--- |
| **Integrante 1** | Frontend, Arquitectura Web & DevOps | **RF1** (Carga CSV), **RF3** (Gestión de modelos), **RF4** (Asociación `mlxtend`), **RF5** (Informes), **RF6** (Dashboard Plotly), **RF7** (UI Orange), Servidor FastAPI y Despliegue. |
| **Integrante 2** | Backend & Motor Científico de ML | **RF2** (Pipeline de limpieza, imputación nulos y outliers IQR con bitácora auditable) y **RF3** (Entrenamiento Scikit-Learn, métricas de evaluación, matriz de confusión y visualizador SVG del árbol). |

---

## 📚 Documentación Adicional en PDF
Para una revisión en profundidad, el proyecto cuenta con tomos formales descargables:
* 📄 **Documento Maestro:** `Documento_Arquitectura_Mineria_Datos.pdf`
* 📄 **Tomo I (Diagramas):** `Documento_Diagramas_Funcionales.pdf` (Casos de Uso, Flujogramas, Secuencia y ERD).
* 📄 **Tomo II (Servidores):** `Documento_Arquitectura_Servidores_Comunicacion.pdf` (BPMN, DirectAdmin y Modelo C4).
* 📄 **Tomo III (Especificación):** `Documento_Especificacion_Requerimientos_y_API.pdf` (SRS IEEE 830 y Catálogo OpenAPI).