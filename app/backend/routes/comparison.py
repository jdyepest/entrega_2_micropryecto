"""
Rutas de comparación: GET /api/compare/<analysis_id>
"""

import json
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify

from routes.analysis import get_analysis_store
from services.models import MODELS

comparison_bp = Blueprint("comparison", __name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _eval_results_dir() -> Path:
    return _repo_root() / "artifacts" / "eval_results"


def _parse_run_at(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        # Python can parse ISO with offset; fallback if needed.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _load_latest_eval_results() -> dict[tuple[str, str], dict]:
    """
    Lee artifacts/eval_results/*.json y devuelve el resultado más reciente
    por (model, task). Estructura: {(model, task): result_dict}
    """
    results_dir = _eval_results_dir()
    if not results_dir.exists():
        return {}

    best: dict[tuple[str, str], dict] = {}
    best_ts: dict[tuple[str, str], datetime] = {}

    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        meta = data.get("meta") or {}
        run_at = _parse_run_at(meta.get("run_at"))
        for res in data.get("results") or []:
            model = (res.get("model") or "").strip().lower()
            task = (res.get("task") or "").strip().lower()
            if not model or not task:
                continue
            key = (model, task)
            if key not in best_ts or run_at >= best_ts[key]:
                best_ts[key] = run_at
                best[key] = res

    return best


def _apply_eval_overrides(metrics: dict) -> dict:
    """
    Sobrescribe métricas simuladas con resultados reales si están disponibles.
    """
    eval_map = _load_latest_eval_results()
    if not eval_map:
        return metrics

    for model in ["encoder", "llm", "api"]:
        res1 = eval_map.get((model, "task1"))
        res2 = eval_map.get((model, "task2"))
        if res1:
            metrics["task1_metrics"][model] = {
                "f1": float(res1.get("f1", 0.0)),
                "precision": float(res1.get("precision", 0.0)),
                "recall": float(res1.get("recall", 0.0)),
                "latency": float(res1.get("time_per_doc_s", 0.0)),
            }
        if res2:
            metrics["task2_metrics"][model] = {
                "f1": float(res2.get("f1", 0.0)),
                "precision": float(res2.get("precision", 0.0)),
                "recall": float(res2.get("recall", 0.0)),
                "latency": float(res2.get("time_per_doc_s", 0.0)),
            }
        if res1 and res2:
            metrics["total_time"][model] = round(
                float(res1.get("total_time_s", 0.0)) + float(res2.get("total_time_s", 0.0)),
                2,
            )

    return metrics


def _generate_comparison_metrics(text: str, seed_model: str) -> dict:
    """
    Genera métricas comparativas simuladas para los 3 modelos.
    Si no hay resultados reales en artifacts/eval_results, usa valores fijos.
    """
    task1 = {}
    task2 = {}
    for m, cfg in MODELS.items():
        f1_1 = round(cfg["task1_f1_base"], 2)
        prec_1 = round(min(f1_1 + 0.02, 0.99), 2)
        rec_1 = round(min(f1_1 + 0.01, 0.99), 2)
        lat_1 = round(cfg["simulated_delay_s"], 2)
        task1[m] = {
            "f1": f1_1,
            "precision": prec_1,
            "recall": rec_1,
            "latency": lat_1,
        }

        f1_2 = round(cfg["task2_f1_base"], 2)
        prec_2 = round(min(f1_2 + 0.02, 0.99), 2)
        rec_2 = round(min(f1_2 + 0.01, 0.99), 2)
        lat_2 = round(cfg["simulated_delay_s"] * 0.6, 2)
        task2[m] = {
            "f1": f1_2,
            "precision": prec_2,
            "recall": rec_2,
            "latency": lat_2,
        }

    cost = {m: round(cfg["cost_per_doc"], 4) for m, cfg in MODELS.items()}
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
    metrics = _apply_eval_overrides(metrics)
    metrics["analysis_id"] = analysis_id
    metrics["original_model"] = entry["model"]

    return jsonify(metrics), 200
