"""
Módulo de Árbol de Decisión Supervisado (RF3)
=============================================
Encargado: Integrante 2 (Backend / Machine Learning)

Permite entrenar un modelo de clasificación con Scikit-Learn,
calcular métricas de desempeño, matriz de confusión y generar la
representación visual interactiva del árbol.
"""

from typing import Dict, Any, List
import pandas as pd


def entrenar_arbol(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    max_depth: int = 5,
    criterion: str = "gini",
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Entrena un modelo de árbol de decisión y retorna métricas de evaluación,
    matriz de confusión y visualizador gráfico en formato SVG.
    """
    # Verificación de datos de prueba
    if df is None or df.empty:
        raise ValueError("No hay datos disponibles para entrenar el árbol.")

    # 2. Entrenamiento real del árbol
    raise NotImplementedError("El entrenamiento del árbol aún no está implementado.")
