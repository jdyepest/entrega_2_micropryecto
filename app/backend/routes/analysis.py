"""
Rutas de análisis: POST /api/analyze
"""

import uuid
from flask import Blueprint, request, jsonify

from services.segmentation import analyze_segments
from services.contributions import analyze_contributions
from services.models import MODELS

analysis_bp = Blueprint("analysis", __name__)

# Almacén en memoria de análisis (reemplazable por DB)
_analysis_store: dict[str, dict] = {}


@analysis_bp.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)

    text = (body.get("text") or "").strip()
    model = body.get("model", "encoder")
    tasks = body.get("tasks", ["segmentation", "contributions"])

    # Validaciones básicas
    if not text:
        return jsonify({"error": "El campo 'text' es requerido."}), 400
    if len(text) < 50:
        return jsonify({"error": "El texto es demasiado corto (mín. 50 caracteres)."}), 400
    if model not in MODELS:
        return jsonify({"error": f"Modelo '{model}' no válido. Opciones: {list(MODELS.keys())}"}), 400

    analysis_id = str(uuid.uuid4())
    result: dict = {
        "id": analysis_id,
        "model": model,
        "model_name": MODELS[model]["name"],
        "segmentation": None,
        "contributions": None,
    }

    # Tarea 1: Segmentación retórica
    segmentation_data = None
    if "segmentation" in tasks:
        segmentation_data = analyze_segments(text, model)
        result["segmentation"] = segmentation_data

    # Tarea 2: Extracción de contribuciones (requiere segmentos)
    if "contributions" in tasks:
        segments = (
            segmentation_data["segments"]
            if segmentation_data
            else analyze_segments(text, model)["segments"]
        )
        result["contributions"] = analyze_contributions(segments, model)

    # Guardar en memoria para /api/compare
    _analysis_store[analysis_id] = {
        "text": text,
        "model": model,
        "result": result,
    }

    return jsonify(result), 200


def get_analysis_store():
    """Expone el store para uso en otras rutas."""
    return _analysis_store
