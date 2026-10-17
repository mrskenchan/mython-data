"""
Módulo del Dashboard Analítico Dinámico (RF6)
==============================================
Encargado: Integrante 1 (Frontend, Visualización & Experiencia de Usuario)

Genera las estructuras de datos, correlaciones de Pearson, resúmenes descriptivos
y muestras optimizadas para renderizado reactivo con Plotly.js en el cliente.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def obtener_datos_dashboard(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Calcula los datos analíticos para alimentar el Dashboard interactivo Plotly.
    
    Retorna:
    - KPIs del dataset (registros, variables, completitud %, memoria aproximada).
    - Lista clasificada de columnas numéricas y categóricas.
    - Matriz de correlación lineal de Pearson (Heatmap).
    - Estadísticas descriptivas de variables numéricas.
    - Muestra representativa de registros (hasta 500 filas) en formato JSON.
    """
    if df is None or df.empty:
        return {
            "valido": False,
            "kpis": {
                "total_filas": 0,
                "total_columnas": 0,
                "completitud_pct": 100.0,
                "num_numericas": 0,
                "num_categoricas": 0
            },
            "columnas_numericas": [],
            "columnas_categoricas": [],
            "matriz_correlacion": {"variables": [], "valores": []},
            "estadisticas": {},
            "registros_muestra": []
        }

    # 1. Clasificación de Columnas
    columnas_numericas = []
    columnas_categoricas = []

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            columnas_categoricas.append(str(col))
        elif pd.api.types.is_numeric_dtype(df[col]):
            columnas_numericas.append(str(col))
        else:
            columnas_categoricas.append(str(col))

    # 2. KPIs
    total_filas = len(df)
    total_columnas = len(df.columns)
    tasa_nulos = float(df.isna().mean().mean())
    completitud_pct = round((1.0 - tasa_nulos) * 100, 1)

    kpis = {
        "total_filas": total_filas,
        "total_columnas": total_columnas,
        "completitud_pct": completitud_pct,
        "num_numericas": len(columnas_numericas),
        "num_categoricas": len(columnas_categoricas)
    }

    # 3. Matriz de Correlación de Pearson
    matriz_correlacion = {"variables": [], "valores": []}
    if len(columnas_numericas) >= 2:
        df_num = df[columnas_numericas].dropna()
        if not df_num.empty:
            corr_df = df_num.corr(method="pearson").round(2).fillna(0)
            matriz_correlacion = {
                "variables": list(corr_df.columns),
                "valores": corr_df.values.tolist()
            }

    # 4. Estadísticas Descriptivas de Numéricas
    estadisticas = {}
    for col in columnas_numericas:
        s = df[col].dropna()
        if not s.empty:
            estadisticas[col] = {
                "media": round(float(s.mean()), 2),
                "mediana": round(float(s.median()), 2),
                "desv_std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2)
            }

    # 5. Muestra de Registros para el Scatter Plot (máximo 500 registros para no sobrecargar el DOM)
    if total_filas > 500:
        df_muestra = df.sample(500, random_state=42)
    else:
        df_muestra = df.copy()

    # Convertir NaN a None / null seguro para JSON
    df_muestra_clean = df_muestra.where(pd.notnull(df_muestra), None)
    registros_muestra = df_muestra_clean.to_dict(orient="records")

    return {
        "valido": True,
        "kpis": kpis,
        "columnas_numericas": columnas_numericas,
        "columnas_categoricas": columnas_categoricas,
        "matriz_correlacion": matriz_correlacion,
        "estadisticas": estadisticas,
        "registros_muestra": registros_muestra
    }
