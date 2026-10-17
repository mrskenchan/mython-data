"""
Módulo de Reglas de Asociación y Patrones Frecuentes (RF4)
==========================================================
Encargado: Integrante 1 (Frontend & Minería Descriptiva)

Implementa algoritmos de minería de reglas de asociación con MLxtend
(FP-Growth y Apriori), calculando soporte, confianza y lift.
"""

from typing import Dict, Any, List
import pandas as pd


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
        }
    ]
    # Filtrar según los umbrales recibidos
    filtradas = [
        r for r in reglas
        if r["support"] >= min_support and r["confidence"] >= min_confidence and r["lift"] >= min_lift
    ]
    return filtradas
