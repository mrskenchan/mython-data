"""
Módulo de Árbol de Decisión Supervisado (RF3)
=============================================
Encargado: Integrante 2 (Backend / Machine Learning)

Permite entrenar un modelo de clasificación con Scikit-Learn,
calcular métricas de desempeño, matriz de confusión y generar la
representación visual interactiva del árbol.
"""

from io import StringIO
from typing import Dict, Any, List
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,recall_score, f1_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from io import StringIO




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
    
    if target not in df.columns:
        raise ValueError("La variable objetivo no existe en el CSV.")
    
# 2. Verificar que existan variables predictoras
    if not features:
        raise ValueError("Debes seleccionar al menos una variable predictora.")
    
    if not all(feature in df.columns for feature in features):
        raise ValueError("Algunas variables predictoras no existen en el CSV.")
    if target in features:
        raise ValueError("La variable objetivo no puede ser una variable predictora.")
#3. Seleccionar las variables predictoras
    X = df[features].copy()
    y = df[target].copy()
# 4. Manejo de valores faltantes
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Existen valores faltantes en las variables predictoras o en la variable objetivo.")
    #convertir variables categóricas a numéricas
    X = pd.get_dummies(X, dtype=int)
# 5. Entrenamiento real del árbol
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    modelo = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    importancia_variables = modelo.feature_importances_
    nombre_variables = X.columns.tolist()
    importancia_variables = pd.Series(importancia_variables, index=nombre_variables).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    plot_tree(modelo, feature_names=nombre_variables, class_names=modelo.classes_, filled=True)
    plt.title("Árbol de Decisión")
    plt.xlabel("Variables Predictoras")
    plt.ylabel("Clase")
    svg_buffer = StringIO()
    plt.savefig(svg_buffer, format='svg')
    svg_string = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close()

    return {
        "modelo": modelo,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matriz_confusion.tolist(), # Convertir a lista para serialización JSON
        "importancia_variables": importancia_variables.to_dict(), # Convertir a diccionario para serialización JSON
        "tree_svg": svg_string # Agregar la cadena SVG al diccionario de retorno
    }

    
