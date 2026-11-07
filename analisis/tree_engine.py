"""
Módulo de Árbol de Decisión Supervisado (RF3)
=============================================
Encargado: Integrante 2 (Backend / Machine Learning)

Permite entrenar un modelo de clasificación con Scikit-Learn (DecisionTreeClassifier),
calcular métricas de desempeño (Accuracy, Precision, Recall, F1), matriz de confusión,
ranking de importancia de variables y generación visual del árbol en formato SVG.
Incluye persistencia y gestión de modelos serializados con joblib.
"""

import os
import datetime
import uuid
from io import StringIO
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

# Configurar backend no interactivo de Matplotlib para servidores web concurrentes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

STORAGE_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "models"
)


def _asegurar_directorio_modelos():
    """Crea la carpeta de persistencia de modelos si no existe."""
    os.makedirs(STORAGE_MODELS_DIR, exist_ok=True)


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
    Entrena un modelo de árbol de decisión sobre el DataFrame provisto y retorna métricas,
    matriz de confusión, importancia de variables y visualizador SVG.
    """
    # 1. Validaciones previas
    if df is None or df.empty:
        raise ValueError("No hay datos disponibles para entrenar el árbol de decisión.")
    
    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en el dataset.")
    
    if not features:
        raise ValueError("Debes seleccionar al menos una variable predictora (Feature).")
    
    features_validas = [f for f in features if f in df.columns and f != target]
    if not features_validas:
        raise ValueError("No se encontraron variables predictoras válidas distintas del target.")

    # 2. Selección y preparación de datos
    X = df[features_validas].copy()
    y = df[target].copy()

    # Si hay nulos, imputación defensiva para garantizar convergencia
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                moda = X[col].mode().iloc[0] if not X[col].mode().empty else "Desconocido"
                X[col] = X[col].fillna(moda)

    if y.isnull().any():
        moda_y = y.mode().iloc[0] if not y.mode().empty else "Desconocido"
        y = y.fillna(moda_y)

    # Conversión One-Hot para categóricas
    X = pd.get_dummies(X, drop_first=False, dtype=int)
    nombre_variables = X.columns.tolist()

    # 3. Partición de datos Train/Test
    # Si la muestra es muy reducida (< 6 filas), entrenamos y evaluamos sobre el conjunto completo
    n_muestras = len(X)
    if n_muestras < 6:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        test_ratio = max(0.10, min(0.50, float(test_size)))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=random_state, stratify=y if y.nunique() > 1 else None
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=random_state
            )

    # 4. Entrenamiento Scikit-Learn
    modelo = DecisionTreeClassifier(
        max_depth=int(max_depth),
        criterion=criterion,
        random_state=random_state
    )
    modelo.fit(X_train, y_train)

    # 5. Predicción y Evaluación de Métricas
    y_pred = modelo.predict(X_test)
    clases_modelo = [str(c) for c in modelo.classes_]

    acc = float(accuracy_score(y_test, y_pred))
    matriz_conf = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Importancia de variables
    importancia_series = pd.Series(modelo.feature_importances_, index=nombre_variables).sort_values(ascending=False)
    importancia_dict = {str(k): round(float(v), 4) for k, v in importancia_series.items()}

    # 6. Generación del Gráfico en Formato SVG
    fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
    plot_tree(
        modelo,
        feature_names=nombre_variables,
        class_names=clases_modelo,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=9,
        ax=ax
    )
    ax.set_title(f"Árbol de Decisión ({criterion.upper()} | Profundidad: {max_depth})", fontsize=11, fontweight="bold")
    
    svg_buffer = StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_string = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close(fig)

    model_id = str(uuid.uuid4())[:8]
    resultado = {
        "model_id": model_id,
        "name": f"Árbol_{criterion.capitalize()}_D{max_depth}",
        "target": target,
        "features": features_validas,
        "classes": clases_modelo,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": matriz_conf.tolist(),
        "importancia_variables": importancia_dict,
        "tree_svg": svg_string,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hyperparameters": {
            "max_depth": max_depth,
            "criterion": criterion,
            "test_size": test_size,
            "random_state": random_state
        }
    }

    # Guardar automáticamente en disco para gestión y persistencia (RF3)
    try:
        guardar_modelo_disco(model_id, modelo, resultado)
    except Exception:
        pass

    return resultado


def guardar_modelo_disco(model_id: str, modelo: Any, metadata: Dict[str, Any]):
    """Persiste el modelo Scikit-Learn y su metadata en storage/models/."""
    _asegurar_directorio_modelos()
    model_path = os.path.join(STORAGE_MODELS_DIR, f"{model_id}.joblib")
    meta_path = os.path.join(STORAGE_MODELS_DIR, f"{model_id}.meta.json")
    
    # Serializar modelo Scikit-Learn
    joblib.dump(modelo, model_path)
    
    # Guardar metadata ligera sin el SVG para listados rápidos
    meta_copy = {k: v for k, v in metadata.items() if k != "tree_svg"}
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_copy, f, indent=2)


def listar_modelos_guardados() -> List[Dict[str, Any]]:
    """Retorna la lista de modelos entrenados y persistidos."""
    _asegurar_directorio_modelos()
    import json
    modelos = []
    for fname in os.listdir(STORAGE_MODELS_DIR):
        if fname.endswith(".meta.json"):
            meta_path = os.path.join(STORAGE_MODELS_DIR, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    modelos.append(json.load(f))
            except Exception:
                continue
    # Ordenar por fecha más reciente
    modelos.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return modelos


def eliminar_modelo_guardado(model_id: str) -> bool:
    """Elimina un modelo serializado y su metadata."""
    _asegurar_directorio_modelos()
    eliminado = False
    for ext in [".joblib", ".meta.json"]:
        path = os.path.join(STORAGE_MODELS_DIR, f"{model_id}{ext}")
        if os.path.exists(path):
            os.remove(path)
            eliminado = True
    return eliminado
