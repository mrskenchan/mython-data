"""
Módulo de Reglas de Asociación y Patrones Frecuentes (RF4)
==========================================================
Encargado: Integrante 1 (Frontend, Validación & Minería Descriptiva)

Implementa algoritmos de minería de reglas de asociación usando MLxtend
(FP-Growth y Apriori) sobre datasets tabulares y transaccionales, calculando
soporte, confianza, lift y métricas de calidad.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

try:
    from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


def preparar_datos_transaccionales(
    df: pd.DataFrame,
    columnas_seleccionadas: Optional[List[str]] = None,
    max_cardinalidad: int = 20
) -> pd.DataFrame:
    """
    Transforma un DataFrame tabular en una matriz booleana (One-Hot Encoded)
    apta para algoritmos de minería de reglas de asociación.
    
    Discretiza columnas numéricas continuas en 3 rangos (Bajo, Medio, Alto)
    y preserva columnas categóricas con cardinalidad controlada.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols = columnas_seleccionadas if columnas_seleccionadas else list(df.columns)
    df_sub = df[cols].copy()

    df_trans = pd.DataFrame(index=df_sub.index)

    for col in cols:
        serie = df_sub[col].dropna()
        if serie.empty:
            continue

        # Si es booleana
        if pd.api.types.is_bool_dtype(serie):
            df_trans[f"{col}: Sí"] = serie.astype(bool)
        # Si es numérica continua con más de 5 valores únicos -> discretizar
        elif pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 5:
            try:
                # Discretizar en cuantiles (Bajo, Medio, Alto)
                binned = pd.qcut(serie, q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")
                dummies = pd.get_dummies(binned, prefix=col, prefix_sep=": ")
                df_trans = pd.concat([df_trans, dummies], axis=1)
            except Exception:
                # Si qcut falla por duplicados en cuantiles, usar cut por rangos iguales
                try:
                    binned = pd.cut(serie, bins=3, labels=["Bajo", "Medio", "Alto"])
                    dummies = pd.get_dummies(binned, prefix=col, prefix_sep=": ")
                    df_trans = pd.concat([df_trans, dummies], axis=1)
                except Exception:
                    continue
        else:
            # Variable categórica o discreta con cardinalidad razonable
            if serie.nunique() <= max_cardinalidad:
                dummies = pd.get_dummies(serie, prefix=col, prefix_sep=": ")
                df_trans = pd.concat([df_trans, dummies], axis=1)

    return df_trans.astype(bool)


def minar_reglas_asociacion(
    df: Optional[pd.DataFrame] = None,
    columnas: Optional[List[str]] = None,
    min_support: float = 0.05,
    min_confidence: float = 0.60,
    min_lift: float = 1.0,
    algorithm: str = "fpgrowth",
    max_len: int = 3
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline de minería de reglas de asociación con FP-Growth o Apriori.
    
    Retorna un diccionario con:
    - 'reglas': Lista de reglas con antecedentes, consecuentes, soporte, confianza, lift.
    - 'total_reglas': Cantidad de reglas encontradas.
    - 'frequent_itemsets_count': Cantidad de itemsets frecuentes extraídos.
    - 'algoritmo_utilizado': Algoritmo ejecutado.
    - 'aviso': Mensaje informativo o sugerencia de parámetros.
    """
    # Si no se pasó DataFrame o está vacío, usar dataset sintético representativo de demostración
    if df is None or df.empty or not HAS_MLXTEND:
        reglas_mock = generar_reglas_mock(min_support, min_confidence, min_lift, algorithm)
        return {
            "exito": True,
            "reglas": reglas_mock,
            "total_reglas": len(reglas_mock),
            "frequent_itemsets_count": len(reglas_mock) * 2,
            "algoritmo_utilizado": f"{algorithm.upper()} (Dataset de demostración)",
            "aviso": "Mostrando reglas del dataset demostrativo de ejemplo."
        }

    # 1. Transformación One-Hot Booleana
    df_bool = preparar_datos_transaccionales(df, columnas)
    if df_bool.empty or df_bool.shape[1] < 2:
        reglas_mock = generar_reglas_mock(min_support, min_confidence, min_lift, algorithm)
        return {
            "exito": True,
            "reglas": reglas_mock,
            "total_reglas": len(reglas_mock),
            "frequent_itemsets_count": len(reglas_mock) * 2,
            "algoritmo_utilizado": f"{algorithm.upper()} (Demostración)",
            "aviso": "El dataset activo no contiene suficientes columnas categóricas o binarizables. Se muestra conjunto de referencia."
        }

    # 2. Extracción de Itemsets Frecuentes con MLxtend
    try:
        if algorithm.lower() == "apriori":
            itemsets = apriori(df_bool, min_support=min_support, use_colnames=True, max_len=max_len)
        else:
            itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True, max_len=max_len)
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al ejecutar algoritmo de itemsets frecuentes ({algorithm}): {str(e)}",
            "reglas": []
        }

    if itemsets.empty:
        # Sugerir reducir min_support
        return {
            "exito": True,
            "reglas": [],
            "total_reglas": 0,
            "frequent_itemsets_count": 0,
            "algoritmo_utilizado": algorithm.upper(),
            "aviso": f"No se encontraron patrones frecuentes con soporte >= {min_support*100:.1f}%. Intenta reducir el umbral de soporte mínimo."
        }

    # 3. Generación de Reglas de Asociación
    try:
        df_rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al generar reglas de asociación: {str(e)}",
            "reglas": []
        }

    if df_rules.empty:
        return {
            "exito": True,
            "reglas": [],
            "total_reglas": 0,
            "frequent_itemsets_count": len(itemsets),
            "algoritmo_utilizado": algorithm.upper(),
            "aviso": f"Se hallaron {len(itemsets)} itemsets frecuentes, pero ninguna regla alcanzó el {min_confidence*100:.0f}% de confianza. Reduce la confianza mínima."
        }

    # 4. Filtrar por Lift
    df_rules = df_rules[df_rules["lift"] >= min_lift].copy()
    df_rules = df_rules.sort_values(by=["lift", "confidence"], ascending=[False, False])

    # 5. Formatear para presentación
    lista_reglas = []
    for idx, row in df_rules.head(50).iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        sup = float(row["support"])
        conf = float(row["confidence"])
        lift_val = float(row["lift"])

        impacto = "Moderado"
        if lift_val >= 2.0:
            impacto = "Muy Alto"
        elif lift_val >= 1.5:
            impacto = "Alto"

        lista_reglas.append({
            "id": len(lista_reglas) + 1,
            "antecedents": ant,
            "consequents": con,
            "support": round(sup, 4),
            "confidence": round(conf, 4),
            "lift": round(lift_val, 2),
            "impact": impacto
        })

    return {
        "exito": True,
        "reglas": lista_reglas,
        "total_reglas": len(lista_reglas),
        "frequent_itemsets_count": len(itemsets),
        "algoritmo_utilizado": algorithm.upper(),
        "aviso": None
    }


def generar_reglas_mock(
    min_support: float = 0.05,
    min_confidence: float = 0.60,
    min_lift: float = 1.0,
    algorithm: str = "fpgrowth"
) -> List[Dict[str, Any]]:
    """
    Retorna un conjunto de reglas de asociación simuladas representativas para
    renderizar la tabla y el scatter plot en la maqueta visual.
    """
    reglas = [
        {
            "id": 1,
            "antecedents": ["Café", "Leche"],
            "consequents": ["Galletas"],
            "support": 0.082,
            "confidence": 0.84,
            "lift": 2.41,
            "impact": "Muy Alto"
        },
        {
            "id": 2,
            "antecedents": ["Cerveza", "Snacks"],
            "consequents": ["Papas Fritas"],
            "support": 0.065,
            "confidence": 0.78,
            "lift": 1.95,
            "impact": "Alto"
        },
        {
            "id": 3,
            "antecedents": ["Pan", "Mantequilla"],
            "consequents": ["Queso"],
            "support": 0.115,
            "confidence": 0.72,
            "lift": 1.62,
            "impact": "Alto"
        },
        {
            "id": 4,
            "antecedents": ["Yogurt"],
            "consequents": ["Cereales"],
            "support": 0.094,
            "confidence": 0.68,
            "lift": 1.45,
            "impact": "Moderado"
        },
        {
            "id": 5,
            "antecedents": ["Gaseosa"],
            "consequents": ["Hielo"],
            "support": 0.071,
            "confidence": 0.62,
            "lift": 1.25,
            "impact": "Moderado"
        },
        {
            "id": 6,
            "antecedents": ["Té Verde"],
            "consequents": ["Miel"],
            "support": 0.058,
            "confidence": 0.65,
            "lift": 1.38,
            "impact": "Moderado"
        }
    ]
    # Filtrar según los umbrales recibidos
    filtradas = [
        r for r in reglas
        if r["support"] >= min_support and r["confidence"] >= min_confidence and r["lift"] >= min_lift
    ]
    return filtradas
