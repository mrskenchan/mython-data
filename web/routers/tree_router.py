"""
Router de Árbol de Decisión Supervisado (RF3)
============================================
Encargado: Integrante 2 (Backend / Machine Learning)
"""

from typing import List
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from web.templates_config import templates
from web.services.state_manager import state_manager
from analisis.tree_engine import (
    entrenar_arbol,
    listar_modelos_guardados,
    eliminar_modelo_guardado
)

router = APIRouter(tags=["RF3 - Árbol de Decisión"])


def _seleccionar_target_y_features_defecto(columns: List[str]):
    """Selecciona heurísticamente el target (última columna) y las demás como features."""
    if not columns:
        return None, []
    target = columns[-1]
    features = [c for c in columns if c != target and not c.lower().startswith("id")]
    if not features:
        features = [c for c in columns if c != target]
    return target, features


@router.get("/tree", response_class=HTMLResponse)
async def tree_page(request: Request):
    """Vista de Árbol de Decisión (RF3)"""
    ctx = state_manager.get_base_context(request, "tree")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    columns = active.get("columns", [])

    target_defecto, features_defecto = _seleccionar_target_y_features_defecto(columns)

    if df is None or df.empty:
        ctx["error_arbol"] = "El dataset activo no contiene registros (0 filas). No es posible entrenar un árbol de decisión sin datos. Usa el botón 'Restaurar Dataset Original' o carga un nuevo archivo CSV."

    model_result = state_manager.get_last_tree_model()
    if model_result is None and df is not None and not df.empty and target_defecto and features_defecto:
        try:
            model_result = entrenar_arbol(
                df=df,
                target=target_defecto,
                features=features_defecto,
                max_depth=5,
                criterion="gini",
                test_size=0.25
            )
            state_manager.set_last_tree_model(model_result)
        except Exception as e:
            ctx["error_arbol"] = f"No fue posible entrenar modelo inicial automático: {str(e)}"

    ctx["available_columns"] = columns
    ctx["selected_target"] = model_result.get("target") if model_result else target_defecto
    ctx["selected_features"] = model_result.get("features") if model_result else features_defecto
    ctx["current_depth"] = model_result.get("hyperparameters", {}).get("max_depth", 5) if model_result else 5
    ctx["current_test_size"] = int(model_result.get("hyperparameters", {}).get("test_size", 0.25) * 100) if model_result else 25
    ctx["current_criterion"] = model_result.get("hyperparameters", {}).get("criterion", "gini") if model_result else "gini"
    ctx["model_result"] = model_result
    ctx["saved_models"] = listar_modelos_guardados()
    return templates.TemplateResponse(request=request, name="tree.html", context=ctx)


@router.post("/tree", response_class=HTMLResponse)
async def handle_tree(
    request: Request,
    target_column: str = Form(...),
    features: List[str] = Form([]),
    max_depth: int = Form(5),
    test_size: int = Form(25),
    criterion: str = Form("gini")
):
    """Entrena o re-calcula el árbol con los parámetros seleccionados en la UI"""
    ctx = state_manager.get_base_context(request, "tree")
    active = state_manager.get_active_dataset()
    df = active.get("df")
    columns = active.get("columns", [])

    if df is None or df.empty:
        state_manager.set_flash_error("El dataset activo no contiene registros (0 filas). Por favor restaura el dataset original o carga un nuevo archivo CSV.")
        return RedirectResponse(url="/tree", status_code=303)

    # Si no se seleccionó ninguna feature, seleccionar todas las demás por defecto
    if not features:
        features = [c for c in columns if c != target_column]

    try:
        tree_result = entrenar_arbol(
            df=df,
            target=target_column,
            features=features,
            max_depth=max_depth,
            criterion=criterion,
            test_size=test_size / 100.0
        )
        state_manager.set_last_tree_model(tree_result)
        state_manager.set_flash_success(f"Modelo '{tree_result['name']}' entrenado y persistido con éxito (Accuracy: {tree_result['accuracy']*100:.1f}%).")
        ctx["model_result"] = tree_result
    except Exception as e:
        state_manager.set_flash_error(f"Error al entrenar el árbol: {str(e)}")
        ctx["model_result"] = state_manager.get_last_tree_model()

    ctx["available_columns"] = columns
    ctx["selected_target"] = target_column
    ctx["selected_features"] = features
    ctx["current_depth"] = max_depth
    ctx["current_test_size"] = test_size
    ctx["current_criterion"] = criterion
    ctx["saved_models"] = listar_modelos_guardados()
    return templates.TemplateResponse(request=request, name="tree.html", context=ctx)


@router.get("/tree/delete/{model_id}", response_class=HTMLResponse)
async def delete_saved_model(request: Request, model_id: str):
    """Elimina un modelo guardado del almacenamiento local."""
    eliminado = eliminar_modelo_guardado(model_id)
    if eliminado:
        state_manager.set_flash_success(f"Modelo '{model_id}' eliminado del almacenamiento.")
    else:
        state_manager.set_flash_error("El modelo especificado no existe o ya fue eliminado.")
    return RedirectResponse(url="/tree", status_code=303)
