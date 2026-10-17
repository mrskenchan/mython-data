"""
Módulo de Limpieza Automática y Diagnóstico de Datos (RF2)
==========================================================
Encargado: Integrante 2 (Backend / Machine Learning)

Este archivo define el pipeline de limpieza para tratar valores nulos,
duplicados, normalización de nombres, tipado y valores atípicos (outliers IQR).
Actualmente contiene implementaciones mock funcionales listas para conectar
con el frontend mientras el encargado de backend desarrolla la lógica completa.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def diagnosticar_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Escanea el DataFrame crudo sin modificarlo y retorna un diagnóstico preliminar
    para desplegar en el asistente de limpieza interactiva.
    """
    total_filas = len(df)
    total_columnas = len(df.columns)
    
    # Conteo de nulos por columna
    nulos_por_col = df.isnull().sum().to_dict()
    columnas_con_nulos = {col: int(cant) for col, cant in nulos_por_col.items() if cant > 0}
    total_nulos = sum(columnas_con_nulos.values())
    
    # Filas duplicadas
    duplicados = int(df.duplicated().sum())
    
    # Nombres de columnas sucias (espacios, mayúsculas, caracteres especiales)
    cols_sucias = [col for col in df.columns if ' ' in str(col) or not str(col).islower()]
    
    # Diagnóstico preliminar de outliers en columnas numéricas (usando IQR)
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers_detectados = 0
    for col in cols_numericas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers_detectados += int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())

    return {
        "total_filas": total_filas,
        "total_columnas": total_columnas,
        "total_nulos": total_nulos,
        "columnas_con_nulos": columnas_con_nulos,
        "filas_duplicadas": duplicados,
        "columnas_sucias": cols_sucias,
        "columnas_numericas": cols_numericas,
        "outliers_estimados": outliers_detectados,
    }


def ejecutar_limpieza(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta el pipeline modular de limpieza según las opciones seleccionadas en la UI.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original con los datos crudos.
    config : dict
        Opciones de limpieza (normalizar, drop_duplicates, imputar_nulos, outliers_iqr).
        
    Retorna:
    --------
    tuple(pd.DataFrame, dict)
        1. DataFrame limpio y procesado.
        2. Diccionario con la bitácora auditable de cambios realizados.
    """
    df_clean = df.copy()
    bitacora = {
        "duplicados_removidos": 0,
        "nulos_imputados": {},
        "columnas_normalizadas": [],
        "outliers_tratados": 0,
    }

    # 1. Normalización de nombres de columnas
    if config.get("normalizar_columnas", True):
        nombres_originales = list(df_clean.columns)
        df_clean.columns = (
            df_clean.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w\s]", "", regex=True)
        )
        bitacora["columnas_normalizadas"] = list(df_clean.columns)

    # 2. Eliminación de filas duplicadas
    if config.get("eliminar_duplicados", True):
        antes = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        bitacora["duplicados_removidos"] = int(antes - len(df_clean))

    # 3. Tratamiento de nulos
    metodo_imputacion = config.get("imputacion_nulos", "mediana")
    for col in df_clean.columns:
        nulos = int(df_clean[col].isnull().sum())
        if nulos > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                if metodo_imputacion == "mediana":
                    val = df_clean[col].median()
                elif metodo_imputacion == "media":
                    val = df_clean[col].mean()
                else:
                    val = 0
                df_clean[col] = df_clean[col].fillna(val)
                bitacora["nulos_imputados"][col] = f"{nulos} imputados ({metodo_imputacion}: {val:.2f})"
            else:
                # Categórico: moda o 'Desconocido'
                moda = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Desconocido"
                df_clean[col] = df_clean[col].fillna(moda)
                bitacora["nulos_imputados"][col] = f"{nulos} imputados (moda: '{moda}')"

    # 4. Manejo de Outliers (Recorte IQR - Winsorization)
    if config.get("manejo_outliers", True):
        cols_num = df_clean.select_dtypes(include=[np.number]).columns
        for col in cols_num:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lim_inf = q1 - 1.5 * iqr
                lim_sup = q3 + 1.5 * iqr
                outliers = ((df_clean[col] < lim_inf) | (df_clean[col] > lim_sup)).sum()
                if outliers > 0:
                    df_clean[col] = df_clean[col].clip(lower=lim_inf, upper=lim_sup)
                    bitacora["outliers_tratados"] += int(outliers)

    return df_clean, bitacora
