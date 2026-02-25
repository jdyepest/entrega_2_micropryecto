"""
Rutas de comparación: GET /api/compare/<analysis_id>
"""

import random
from flask import Blueprint, jsonify

from routes.analysis import get_analysis_store
from services.models import MODELS

comparison_bp = Blueprint("comparison", __name__)


def _generate_comparison_metrics(text: str, seed_model: str) -> dict:
    """
    Genera métricas comparativas simuladas para los 3 modelos.
    Las métricas del modelo usado en el análisis original son las más fieles;
    las demás se simulan con variación realista.
    """
    rng = random.Random(hash(text[:120]))

    def _jitter(base: float, spread: float = 0.04) -> float:
        return round(min(max(base + rng.uniform(-spread, spread), 0.50), 0.99), 2)

    def _lat_jitter(base: float) -> float:
        return round(base + rng.uniform(-0.2, 0.4), 2)

    task1 = {}
    task2 = {}
    for m, cfg in MODELS.items():
        f1_1 = _jitter(cfg["task1_f1_base"])
        prec_1 = _jitter(f1_1, 0.03)
        rec_1 = _jitter(f1_1, 0.03)
        lat_1 = _lat_jitter(cfg["simulated_delay_s"])
        task1[m] = {
            "f1": f1_1,
            "precision": prec_1,
            "recall": rec_1,
            "latency": lat_1,
        }

        f1_2 = _jitter(cfg["task2_f1_base"])
        prec_2 = _jitter(f1_2, 0.03)
        rec_2 = _jitter(f1_2, 0.03)
        lat_2 = _lat_jitter(cfg["simulated_delay_s"] * 0.6)
        task2[m] = {
            "f1": f1_2,
            "precision": prec_2,
            "recall": rec_2,
            "latency": lat_2,
        }

    cost = {m: round(cfg["cost_per_doc"] * rng.uniform(0.9, 1.1), 4) for m, cfg in MODELS.items()}
    total_time = {
        "encoder": round(task1["encoder"]["latency"] + task2["encoder"]["latency"], 2),
        "llm": round(task1["llm"]["latency"] + task2["llm"]["latency"], 2),
        "api": round(task1["api"]["latency"] + task2["api"]["latency"], 2),
    }

    return {
        "task1_metrics": task1,
        "task2_metrics": task2,
        "cost_per_doc": cost,
        "total_time": total_time,
    }


@comparison_bp.route("/api/compare/<analysis_id>", methods=["GET"])
def compare(analysis_id: str):
    """
    Devuelve métricas comparativas simuladas para los 3 modelos.
    ---
    tags:
      - comparison
    produces:
      - application/json
    parameters:
      - name: analysis_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Métricas de comparación
        schema:
          type: object
      404:
        description: Análisis no encontrado
        schema:
          type: object
          properties:
            error: {type: string}
    """
    store = get_analysis_store()

    if analysis_id not in store:
        return jsonify({"error": "Análisis no encontrado. Realiza primero un análisis."}), 404

    entry = store[analysis_id]
    metrics = _generate_comparison_metrics(entry["text"], entry["model"])
    metrics["analysis_id"] = analysis_id
    metrics["original_model"] = entry["model"]

    return jsonify(metrics), 200
